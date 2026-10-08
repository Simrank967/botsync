from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Constants (match config/game.json; ball speed is re-read from the observation if seen)
# --------------------------------------------------------------------------------------
DIRECTION_VECTORS = {
    "STAY": (0, 0),
    "UP": (0, 1),
    "UP_RIGHT": (1, 1),
    "RIGHT": (1, 0),
    "DOWN_RIGHT": (1, -1),
    "DOWN": (0, -1),
    "DOWN_LEFT": (-1, -1),
    "LEFT": (-1, 0),
    "UP_LEFT": (-1, 1),
}
MOVES = list(DIRECTION_VECTORS)
KICK_DIRECTIONS = [name for name in MOVES if name != "STAY"]
ACTION_COUNT = len(MOVES) + len(KICK_DIRECTIONS) * 3

_R2 = math.sqrt(0.5)
UNIT = {
    name: ((0.0, 0.0) if name == "STAY" else (vx / math.hypot(vx, vy), vy / math.hypot(vx, vy)))
    for name, (vx, vy) in DIRECTION_VECTORS.items()
}

BALL_RADIUS = 1.5
BALL_SPEED = 8.0
KICK_DISTANCES = (32.0, 64.0, 96.0)
POSSESSION_RADIUS = 5.0
TACKLE_CONTACT = 6.15
SAFE_CARRY_STEPS = 3          # opponent cannot tackle until possession_steps >= 3
FORCED_RELEASE_STEPS = 10
PLAN_SUBSTEP = 2.0            # planning resolution for the ball (engine uses ~0.67)


def direction_toward(dx: float, dy: float, dead_zone: float = 1.0) -> str:
    horizontal = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    vertical = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    return f"{vertical}_{horizontal}" if vertical and horizontal else vertical or horizontal or "STAY"


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


_GRID_CACHE: dict = {}

# --------------------------------------------------------------------------------------
# Per-observation context
# --------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, observation: dict[str, Any]) -> None:
        state = observation["state"]
        field = state["field"]
        self.player_id = observation["player_id"]
        self.opponent_id = observation["opponent_id"]
        self.W = float(field["width"])
        self.H = float(field["height"])
        self.goal_w = float(field.get("goal_width", 36.0))
        self.goal_l = (self.W - self.goal_w) / 2
        self.goal_r = self.goal_l + self.goal_w
        self.radius = float(field.get("player_radius", 3.0))
        self.speed = float(field.get("player_speed", 4.0))
        self.sign = 1.0 if observation["attack_direction"] == "UP" else -1.0
        self.attack = observation["attack_direction"]
        me = state["players"][self.player_id]
        opp = state["players"][self.opponent_id]
        self.me = (float(me["x"]), float(me["y"]))
        self.opp = (float(opp["x"]), float(opp["y"]))
        ball = state["ball"]
        self.ball = ball
        self.bx, self.by = float(ball["x"]), float(ball["y"])
        self.status = ball["status"]
        self.possession = ball["possession"]
        self.poss_steps = int(ball.get("possession_steps", 0))
        velocity = ball.get("velocity", {}) or {}
        self.bvx, self.bvy = float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0))
        self.ball_remaining = float(ball.get("remaining_kick_distance", 0.0))
        speed = math.hypot(self.bvx, self.bvy)
        self.ball_speed = speed if speed > 1e-6 else BALL_SPEED
        self.obstacles = [
            (float(o["x"]), float(o["y"]), float(o["x"]) + float(o["width"]), float(o["y"]) + float(o["height"]))
            for o in state.get("obstacles", [])
        ]
        self.iteration = int(state.get("iteration", 0))
        self.score = state.get("score", {})
        self.dribble_bias = DRIBBLE_BIAS
        self.conceded = int((state.get("score") or {}).get(self.opponent_id, 0))
        self.lead = int((state.get("score") or {}).get(self.player_id, 0)) - self.conceded
        self.max_iter = max(1, int(state.get("maximum_iterations", 400)))
        self.my_goal_y = 0.0 if self.sign > 0 else self.H
        self.opp_goal_y = self.H if self.sign > 0 else 0.0
        self.my_goal = (self.W / 2, self.my_goal_y)
        self.opp_goal = (self.W / 2, self.opp_goal_y)
        self.no_kick = False
        self._field = None

    # ---- geometry helpers ----
    def circle_hits_obstacle(self, x: float, y: float, r: float) -> bool:
        for x0, y0, x1, y1 in self.obstacles:
            cx = x0 if x < x0 else x1 if x > x1 else x
            cy = y0 if y < y0 else y1 if y > y1 else y
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return True
        return False

    def player_ok(self, x: float, y: float, margin: float = 0.1) -> bool:
        r = self.radius + margin
        if x - r < 0 or x + r > self.W or y - r < 0 or y + r > self.H:
            return False
        return not self.circle_hits_obstacle(x, y, r)

    def step_pos(self, pos: tuple[float, float], move: str) -> tuple[float, float]:
        ux, uy = UNIT[move]
        return pos[0] + ux * self.speed, pos[1] + uy * self.speed

    def valid_moves(self, pos: tuple[float, float] | None = None, margin: float = 0.1) -> list[str]:
        pos = pos or self.me
        out = []
        for move in MOVES:
            if move == "STAY":
                continue
            x, y = self.step_pos(pos, move)
            if self.player_ok(x, y, margin):
                out.append(move)
        return out

    def segment_clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(dist / 2.0))
        for i in range(1, n):
            t = i / n
            if not self.player_ok(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.1):
                return False
        return True

    # ---- flood-fill navigation field ----
    def free_grid(self):
        key = tuple(self.obstacles)
        cached = _GRID_CACHE.get(key)
        if cached is not None:
            return cached
        cell = 2.0
        nx, ny = int(self.W / cell), int(self.H / cell)
        free = [[self.player_ok((ix + 0.5) * cell, (iy + 0.5) * cell, 0.6) for iy in range(ny)] for ix in range(nx)]
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = (free, cell, nx, ny)
        return _GRID_CACHE[key]

    def nav_field(self, target: tuple[float, float]):
        free, cell, nx, ny = self.free_grid()
        tx = int(_clamp(target[0] / cell, 0, nx - 1))
        ty = int(_clamp(target[1] / cell, 0, ny - 1))
        mx = int(_clamp(self.me[0] / cell, 0, nx - 1))
        my = int(_clamp(self.me[1] / cell, 0, ny - 1))
        dist = {}
        heap = []
        seeds = [(tx, ty)]
        if not free[tx][ty]:
            seeds = [
                (tx + dx, ty + dy)
                for dx in range(-4, 5)
                for dy in range(-4, 5)
                if 0 <= tx + dx < nx and 0 <= ty + dy < ny and free[tx + dx][ty + dy]
            ]
        for s in seeds:
            dist[s] = 0.0
            heapq.heappush(heap, (0.0, s))
        limit = None
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, 1e18):
                continue
            if limit is not None and d > limit:
                break
            if node == (mx, my) and limit is None:
                limit = d + 4.0
            ix, iy = node
            for dx, dy, c in steps:
                jx, jy = ix + dx, iy + dy
                if jx < 0 or jy < 0 or jx >= nx or jy >= ny or not free[jx][jy]:
                    continue
                nd = d + c
                if nd < dist.get((jx, jy), 1e18):
                    dist[(jx, jy)] = nd
                    heapq.heappush(heap, (nd, (jx, jy)))
        return dist, cell, nx, ny

    def nav_cost(self, field, pos: tuple[float, float], fallback_target: tuple[float, float]) -> float:
        dist, cell, nx, ny = field
        ix = int(_clamp(pos[0] / cell, 0, nx - 1))
        iy = int(_clamp(pos[1] / cell, 0, ny - 1))
        best = dist.get((ix, iy))
        if best is None:
            best = 1e4 + math.hypot(pos[0] - fallback_target[0], pos[1] - fallback_target[1])
        return best * cell


def navigate(ctx: Ctx, target: tuple[float, float], stop_distance: float = 0.0, allow_stay: bool = True) -> str:
    """Return the best move toward ``target`` honouring obstacles."""
    me = ctx.me
    dx, dy = target[0] - me[0], target[1] - me[1]
    distance = math.hypot(dx, dy)
    if distance <= max(stop_distance, 0.3) and allow_stay:
        return "STAY"
    direct = ctx.segment_clear(me, target) if distance < 160 else False
    candidates = []
    if direct:
        for move in MOVES:
            if move == "STAY":
                continue
            x, y = ctx.step_pos(me, move)
            if not ctx.player_ok(x, y):
                continue
            candidates.append((math.hypot(target[0] - x, target[1] - y), move))
        if candidates:
            best = min(candidates)
            if allow_stay and best[0] >= distance - 0.2:
                return "STAY"
            return best[1]
    field = ctx.nav_field(target)
    base = ctx.nav_cost(field, me, target)
    scored = []
    for move in MOVES:
        if move == "STAY":
            continue
        x, y = ctx.step_pos(me, move)
        if not ctx.player_ok(x, y):
            continue
        scored.append((ctx.nav_cost(field, (x, y), target) + 0.01 * math.hypot(target[0] - x, target[1] - y), move))
    if not scored:
        return "STAY"
    best = min(scored)
    if allow_stay and best[0] >= base and distance < 6:
        return "STAY"
    return best[1]


# --------------------------------------------------------------------------------------
# Ball physics (mirrors the engine, at a coarser planning resolution)
# --------------------------------------------------------------------------------------
def ball_trace(ctx: Ctx, x: float, y: float, ux: float, uy: float, distance: float, max_steps: int = 14,
               ball_speed: float | None = None):
    """Simulate a ball. Returns (points, goal) where points are (t, x, y, travelled) and
    goal is ('top'|'bottom', t, travelled) or None. t is the 1-based simulation step."""
    speed = ball_speed or ctx.ball_speed
    W, H, r = ctx.W, ctx.H, BALL_RADIUS
    gl, gr = ctx.goal_l, ctx.goal_r
    obstacles = ctx.obstacles
    vx, vy = ux, uy
    remaining = distance
    travelled = 0.0
    points = []
    t = 0
    while remaining > 1e-9 and t < max_steps:
        t += 1
        travel = min(speed, remaining)
        subs = max(1, math.ceil(travel / PLAN_SUBSTEP))
        sd = travel / subs
        for _ in range(subs):
            px, py = x, y
            cx, cy = px + vx * sd, py + vy * sd
            if gl <= cx <= gr:
                if cy + r >= H:
                    points.append((t, cx, cy, travelled + sd))
                    return points, ("top", t, travelled + sd)
                if cy - r <= 0:
                    points.append((t, cx, cy, travelled + sd))
                    return points, ("bottom", t, travelled + sd)
            if cx - r < 0 or cx + r > W:
                vx = -vx
                cx = _clamp(cx, r, W - r)
            if cy - r < 0 or cy + r > H:
                vy = -vy
                cy = _clamp(cy, r, H - r)
            for x0, y0, x1, y1 in obstacles:
                qx = x0 if cx < x0 else x1 if cx > x1 else cx
                qy = y0 if cy < y0 else y1 if cy > y1 else cy
                if (cx - qx) ** 2 + (cy - qy) ** 2 < r * r:
                    left, right, bottom, top = x0 - r, x1 + r, y0 - r, y1 + r
                    crossed_x = px <= left or px >= right
                    crossed_y = py <= bottom or py >= top
                    if crossed_x:
                        vx = -vx
                    if crossed_y:
                        vy = -vy
                    if not crossed_x and not crossed_y:
                        vx, vy = -vx, -vy
                    cx, cy = px + vx * 0.05, py + vy * 0.05
                    break
            x, y = cx, cy
            travelled += sd
            remaining -= sd
            points.append((t, x, y, travelled))
    return points, None


# --------------------------------------------------------------------------------------
# Kick evaluation
# --------------------------------------------------------------------------------------
GOAL_VALUE = 100.0
RACE_MARGIN = 0.0         # give up the chase for a loose ball as soon as the opponent is closer
DRIBBLE_BIAS = 10.0       # >0 prefers keeping the ball over a non-scoring kick
BIAS_CYCLE = (10.0, -1.0, 4.0)   # tactic used after 0, 1, 2, ... goals conceded (cycles)
SHOOT_NOW = 60.0          # kick immediately when the best shot is worth at least this much
LOOKAHEAD = 3             # dribble planning depth (steps)
URGENCY_SHOOT = 18.0      # urgency lowers the shoot-now threshold (behind) / raises it (ahead)
RACE_WEIGHT = 12  # value of winning/losing the race for the loose ball after a kick
BOUNCE_BACK_PENALTY = 30.0  # a kick that rebounds onto us is not progress (self-interception loop)
MOUTH_BONUS = 4.0         # reward for ending a kick in front of the goal mouth


def _race_time(ctx: Ctx, pos: tuple[float, float], target: tuple[float, float]) -> float:
    return max(0.0, math.hypot(pos[0] - target[0], pos[1] - target[1]) - POSSESSION_RADIUS) / ctx.speed


def evaluate_kicks(ctx: Ctx, pos: tuple[float, float], opp: tuple[float, float], directions: list[str],
                   danger: float = 1.0, t_offset: float = 0.0):
    """Return list of (value, direction, power, info) for every kick candidate."""
    results = []
    sign = ctx.sign
    for direction in directions:
        ux, uy = UNIT[direction]
        clearance = ctx.radius + BALL_RADIUS + 0.05
        sx, sy = pos[0] + ux * clearance, pos[1] + uy * clearance
        points, goal = ball_trace(ctx, sx, sy, ux, uy, KICK_DISTANCES[-1])
        # pre-compute opponent slack along the path: positive => reachable
        slacks = []
        first_reach = None
        running = -1e9
        for index, (t, px, py, travelled) in enumerate(points):
            slack = 4.5 + ctx.speed * (t + t_offset) - math.hypot(opp[0] - px, opp[1] - py)
            running = max(running, slack)
            slacks.append(running)
            if first_reach is None and slack >= 0.0:
                first_reach = index
        for power in (1, 2, 3):
            limit = KICK_DISTANCES[power - 1]
            # slice the path at the kick distance
            end_index = len(points) - 1
            for index, point in enumerate(points):
                if point[3] >= limit - 1e-6:
                    end_index = index
                    break
            scored_here = goal is not None and goal[2] <= limit + 1e-6
            if scored_here:
                end_index = len(points) - 1
            slack_max = slacks[end_index] if slacks else -99.0
            p_int = _clamp((slack_max + 5.0) / 10.0, 0.0, 1.0)
            if first_reach is not None and first_reach <= end_index:
                ip = points[first_reach]
                my_dist_to_goal = abs(ip[2] - ctx.my_goal_y)
                v_int = -6.0 - 14.0 * danger * (1.0 - my_dist_to_goal / ctx.H)
            else:
                v_int = -10.0
            if scored_here:
                side = goal[0]
                mine = (side == "top") == (sign > 0)
                if mine:
                    value = (1 - p_int) * GOAL_VALUE + p_int * v_int
                else:
                    value = -GOAL_VALUE
                results.append((value, direction, power, {"goal": mine, "p_int": p_int}))
                continue
            final = points[end_index] if points else (0, sx, sy, 0.0)
            fx, fy = final[1], final[2]
            t_me = _race_time(ctx, pos, (fx, fy)) + 0.5
            t_opp = _race_time(ctx, opp, (fx, fy))
            progress = sign * (fy - ctx.H / 2) / (ctx.H / 2)
            v_end = RACE_WEIGHT * _clamp((t_opp - t_me) / 3.0, -1.0, 1.0) + 9.0 * progress
            if progress > 0.4:  # finishing in front of the goal mouth beats a wide ball
                v_end += MOUTH_BONUS * (1.0 - min(1.0, abs(fx - ctx.W / 2) / (ctx.W / 2)))
            for _t, bx, by, travelled_d in points[: end_index + 1]:
                if travelled_d > 8.0 and math.hypot(bx - pos[0], by - pos[1]) < 5.5:
                    v_end -= BOUNCE_BACK_PENALTY
                    break
            value = (1 - p_int) * v_end + p_int * v_int
            results.append((value, direction, power, {"goal": False, "p_int": p_int}))
    return results


# --------------------------------------------------------------------------------------
# Tactical logic
# --------------------------------------------------------------------------------------
def _forward_dirs(ctx: Ctx, wide: bool = False) -> list[str]:
    a = ctx.attack
    base = [a, f"{a}_LEFT", f"{a}_RIGHT"]
    if wide:
        base += ["LEFT", "RIGHT"]
    return base


def _best_kick(ctx: Ctx, pos, opp, directions) -> tuple[float, str, int]:
    results = evaluate_kicks(ctx, pos, opp, directions)
    best = max(results, key=lambda item: (item[0], item[2]))
    return best[0], best[1], best[2]


def _approach(pos, target, speed):
    d = math.hypot(target[0] - pos[0], target[1] - pos[1])
    if d <= speed:
        return target
    return pos[0] + (target[0] - pos[0]) / d * speed, pos[1] + (target[1] - pos[1]) / d * speed


def urgency(ctx: Ctx) -> float:
    """>0 when we need goals (behind / tied late), <0 when protecting a lead."""
    elapsed = ctx.iteration / ctx.max_iter
    if ctx.lead < 0:
        return min(1.5, 0.6 * (-ctx.lead) + 0.6 * elapsed)
    if ctx.lead == 0:
        return max(0.0, (elapsed - 0.5) / 0.5) * 0.5
    return -min(1.0, 0.35 * ctx.lead + 0.3 * elapsed)


def _follow(ctx: Ctx, opp, pos, steps: float):
    """Where a chasing opponent ends up: closes on ``pos`` but cannot overlap us (contact distance)."""
    d = math.hypot(opp[0] - pos[0], opp[1] - pos[1])
    keep = max(2 * ctx.radius, d - ctx.speed * steps)
    if d <= keep or d < 1e-9:
        return opp
    return pos[0] + (opp[0] - pos[0]) / d * keep, pos[1] + (opp[1] - pos[1]) / d * keep


def attack_with_ball(ctx: Ctx) -> dict[str, Any]:
    me, opp, sign = ctx.me, ctx.opp, ctx.sign
    s = ctx.poss_steps
    kick_dirs = _forward_dirs(ctx, wide=True)
    now_value, now_dir, now_power = _best_kick(ctx, me, opp, kick_dirs)
    reach = ctx.speed + TACKLE_CONTACT
    opp_dist = math.hypot(opp[0] - me[0], opp[1] - me[1])
    must_kick = s >= FORCED_RELEASE_STEPS - 1 or (s >= SAFE_CARRY_STEPS and opp_dist <= reach + 4.0)

    # --- situational aggression -------------------------------------------------------
    urg = urgency(ctx)
    bias = ctx.dribble_bias                       # possession preference is NOT loosened by urgency
    progress_weight = 0.4 + 1.0 * max(0.0, urg)   # urgency drives the carry toward goal instead
    shoot_now = _clamp(SHOOT_NOW - URGENCY_SHOOT * urg, 35.0, 80.0)
    opp_behind = sign * (opp[1] - me[1]) < -6.0                      # clear run at goal
    opp_deep = abs(opp[1] - ctx.opp_goal_y) > ctx.H * 0.55           # defender out of position
    if opp_behind:
        bias += 4.0                                                   # free run: keep carrying it
    if opp_deep and s <= 1:
        shoot_now -= 10.0                                             # counterattack: shoot readily

    def kick_action(move: str) -> dict[str, Any]:
        return {"move": move, "kick": {"direction": now_dir, "power": now_power}}

    if now_value >= shoot_now and not ctx.no_kick:  # deliberate shot: direction/power come from the simulated lane
        return kick_action("STAY")

    best_move, best_value = None, -1e9
    if not must_kick:
        for move in ctx.valid_moves():
            ux, uy = UNIT[move]
            if uy * sign < -0.5:
                continue  # never carry the ball backwards
            pos, best_k = me, None
            for k in range(1, LOOKAHEAD + 1):
                pos = ctx.step_pos(pos, move)
                if not ctx.player_ok(*pos) or s + k > FORCED_RELEASE_STEPS - 2:
                    break
                opp_k = _follow(ctx, opp, pos, k)
                gap = math.hypot(opp_k[0] - pos[0], opp_k[1] - pos[1])
                if s + k >= SAFE_CARRY_STEPS and gap <= TACKLE_CONTACT + 0.5:
                    break  # a tackle becomes possible: we must have kicked by then
                value, _, _ = _best_kick(ctx, pos, opp_k, kick_dirs)
                value *= 0.93 ** k
                if best_k is None or value > best_k:
                    best_k = value
            if best_k is None:
                continue
            value = best_k + progress_weight * (sign * uy)
            if value > best_value:
                best_move, best_value = move, value
    if best_move is not None and (ctx.no_kick or best_value > now_value - bias):
        return {"move": best_move}
    if ctx.no_kick and not must_kick:
        move = navigate(ctx, ctx.opp_goal, allow_stay=False)
        if ctx.player_ok(*ctx.step_pos(me, move)):
            return {"move": move}
    # Kick now toward the best simulated lane; keep moving toward goal while kicking.
    move = navigate(ctx, ctx.opp_goal, allow_stay=False)
    return kick_action(move if ctx.player_ok(*ctx.step_pos(me, move)) else "STAY")


def defend_goal_side(ctx: Ctx) -> tuple[float, float]:
    """Target point between the opposing ball carrier and our goal."""
    ox, oy = ctx.opp
    gx, gy = ctx.my_goal
    d = math.hypot(gx - ox, gy - oy) or 1.0
    ux, uy = (gx - ox) / d, (gy - oy) / d
    gap = min(9.0, d * 0.5)
    return ox + ux * gap, oy + uy * gap


def chase_loose_ball(ctx: Ctx) -> str:
    me, opp = ctx.me, ctx.opp
    if ctx.status == "moving" and ctx.ball_remaining > 0:
        speed = ctx.ball_speed
        vx, vy = ctx.bvx / speed, ctx.bvy / speed
        points, goal = ball_trace(ctx, ctx.bx, ctx.by, vx, vy, ctx.ball_remaining, ball_speed=speed)
        best = None
        gap_best = None
        for t, px, py, travelled in points:
            gap = math.hypot(px - me[0], py - me[1]) - 4.5 - ctx.speed * t
            if gap <= 0:
                best = (px, py, t)
                break
            if gap_best is None or gap < gap_best[0]:
                gap_best = (gap, px, py, t)
        if best is not None:
            return navigate(ctx, (best[0], best[1]), stop_distance=0.5)
        final = points[-1] if points else (0, ctx.bx, ctx.by, 0)
        if gap_best is not None and goal is not None and ((goal[0] == "top") != (ctx.sign > 0)):
            return navigate(ctx, (gap_best[1], gap_best[2]), stop_distance=0.5)
        return navigate(ctx, (final[1], final[2]), stop_distance=1.0)
    ball = (ctx.bx, ctx.by)
    t_me = _race_time(ctx, me, ball)
    t_opp = _race_time(ctx, opp, ball)
    if t_opp < t_me - RACE_MARGIN:
        # opponent will win the ball: deny its shots instead of trailing it
        gx, gy = ctx.my_goal
        d = math.hypot(gx - ball[0], gy - ball[1]) or 1.0
        pull = (ball[0] + (gx - ball[0]) / d * 8.0, ball[1] + (gy - ball[1]) / d * 8.0)
        return defensive_move(ctx, ball, t_offset=t_opp, pull=pull)
    return navigate(ctx, ball, stop_distance=0.0, allow_stay=False)


ALL_DIRS = [name for name in MOVES if name != "STAY"]


def flip_ctx(ctx: Ctx) -> Ctx:
    """The same position seen from the opponent's side (it attacks our goal)."""
    f = Ctx.__new__(Ctx)
    f.__dict__.update(ctx.__dict__)
    f.sign = -ctx.sign
    f.attack = "DOWN" if ctx.attack == "UP" else "UP"
    f.me, f.opp = ctx.opp, ctx.me
    f.my_goal_y, f.opp_goal_y = ctx.opp_goal_y, ctx.my_goal_y
    f.my_goal, f.opp_goal = ctx.opp_goal, ctx.my_goal
    return f


def _threat(fctx: Ctx, shooter, defender, t_offset: float = 0.0) -> float:
    """Value of the shooter's best kick (shooter attacks our goal) given our defender position."""
    results = evaluate_kicks(fctx, shooter, defender, ALL_DIRS, t_offset=t_offset)
    return max(item[0] for item in results)


def defensive_move(ctx: Ctx, shooter, t_offset: float = 0.0, pull=None) -> str:
    """Pick the move that minimises the opponent's best shot; tie-break toward ``pull``."""
    fctx = flip_ctx(ctx)
    me = ctx.me
    best = None
    for move in MOVES:
        q = me if move == "STAY" else ctx.step_pos(me, move)
        if move != "STAY" and not ctx.player_ok(*q):
            continue
        threat = _threat(fctx, shooter, q, t_offset)
        tie = math.hypot(q[0] - pull[0], q[1] - pull[1]) if pull else 0.0
        key = (round(threat / 3.0), tie)
        if best is None or key < best[0]:
            best = (key, move)
    return best[1] if best else "STAY"


def press_carrier(ctx: Ctx) -> str:
    me, opp = ctx.me, ctx.opp
    dist = math.hypot(opp[0] - me[0], opp[1] - me[1])
    if ctx.poss_steps >= SAFE_CARRY_STEPS and dist <= ctx.speed * 2 + TACKLE_CONTACT:
        return navigate(ctx, opp, allow_stay=False)  # tackle needs a non-STAY move
    return defensive_move(ctx, opp, pull=defend_goal_side(ctx))


def tactical_action(observation: dict[str, Any], kick_power: int = 3, no_kick: bool = False) -> dict[str, Any]:
    ctx = Ctx(observation)
    ctx.no_kick = no_kick
    ctx.dribble_bias = BIAS_CYCLE[ctx.conceded % len(BIAS_CYCLE)]
    if ctx.possession == ctx.player_id:
        return attack_with_ball(ctx)
    if ctx.possession == ctx.opponent_id:
        return {"move": press_carrier(ctx)}
    return {"move": chase_loose_ball(ctx)}


class Policy:
    """Lookahead tactical policy. A model file is optional and ignored if large."""

    def __init__(self, q_table=None, visits=None, seed: int = 0, evaluation_epsilon: float = 0.0,
                 safety_margin: float = 0.6, kick_power: int = 3) -> None:
        self.kick_power = kick_power
        self.last_sig = None
        self.stuck = 0
        self.no_kick_until = -1

    @classmethod
    def load(cls, path: Path) -> "Policy":
        try:
            path = Path(path)
            if path.stat().st_size < 1_000_000:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return cls(kick_power=int(raw.get("kick_power", 3)))
        except Exception:
            pass
        return cls()

    def _stall_guard(self, observation: dict[str, Any]) -> bool:
        """True while we are stuck re-collecting our own kick in the same spot."""
        state = observation["state"]
        me = state["players"][observation["player_id"]]
        ball = state["ball"]
        iteration = int(state.get("iteration", 0))
        sig = (round(float(me["x"]), 1), round(float(me["y"]), 1), round(float(ball["x"]), 1),
               round(float(ball["y"]), 1), ball.get("possession"))
        holding = ball.get("possession") == observation["player_id"]
        self.stuck = self.stuck + 1 if (holding and sig == self.last_sig) else 0
        self.last_sig = sig
        if self.stuck >= 2:
            self.no_kick_until = iteration + 4
        return iteration <= self.no_kick_until

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        try:
            return tactical_action(observation, self.kick_power, self._stall_guard(observation))
        except Exception:
            return {"move": "STAY"}