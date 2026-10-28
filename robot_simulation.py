"""
SightSeeker - Vision-Based Autonomous Robot Simulation (2D, v3)
================================================================

Sense -> Think -> Act, every frame:
  SENSE  forward camera cone (object recognition with confidence that drops with distance)
         + 360-degree proximity ring (32 range sensors, like a cheap LiDAR / ultrasonic array)
  THINK  finite-state machine: SEARCHING -> NAVIGATING -> AVOIDING -> GOAL_REACHED,
         with target memory (keeps heading for the last place it saw the target)
  ACT    free-space planner (VFH, Vector Field Histogram): from the proximity ring, how far the robot
         can drive in each direction; it steers toward the open direction closest to the goal

Run:
  python robot_simulation.py                    # interactive demo
  python robot_simulation.py --benchmark 200    # headless randomized trials (no window)

Keys: R = reset with a new goal   SPACE = pause   Q = quit
"""

import math
import os
import random
import sys
import time

BENCHMARK = "--benchmark" in sys.argv
if BENCHMARK:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # run without opening a window
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()

# ======================== CONSTANTS ========================
W, H = 1280, 800
ARENA_W = 900
HUD_W = W - ARENA_W
FPS = 60

# Light Theme Colors
BG          = (242, 243, 247)
ARENA_BG    = (250, 250, 253)
PANEL_BG    = (232, 234, 240)
BORDER      = (200, 205, 215)
TEXT        = (30, 32, 40)
TEXT_DIM    = (120, 125, 140)
TEAL        = (0, 145, 190)
TEAL_LIGHT  = (180, 230, 245)
RED         = (220, 55, 55)
GREEN       = (35, 165, 65)
YELLOW      = (200, 150, 0)
BLUE        = (50, 100, 210)
ORANGE      = (230, 130, 30)
PURPLE      = (130, 60, 190)
GRID_COLOR  = (228, 230, 238)

# Fonts
TITLE_FNT  = pygame.font.SysFont("segoeui", 28, bold=True)
HEAD_FNT   = pygame.font.SysFont("segoeui", 16, bold=True)
MAIN_FNT   = pygame.font.SysFont("consolas", 14)
SMALL_FNT  = pygame.font.SysFont("consolas", 12)
TINY_FNT   = pygame.font.SysFont("consolas", 10)
BIG_FNT    = pygame.font.SysFont("segoeui", 20, bold=True)
ICON_FNT   = pygame.font.SysFont("segoeui", 18)

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("SightSeeker - Vision-Based Autonomous Robot (2D)")
clock = pygame.time.Clock()

STATE_COLORS = {"SEARCHING": YELLOW, "NAVIGATING": GREEN, "AVOIDING": RED, "GOAL_REACHED": TEAL}


# ======================== LANGUAGE GROUNDING ========================
# name -> (nouns, attributes). Free text such as "find the red barrel" is scored against these.
GOAL_WORDS = {
    "Red Barrel":  (["barrel", "drum", "oil"], ["red"]),
    "Blue Box":    (["box", "carton", "package"], ["blue"]),
    "Table":       (["table", "bench"], ["orange", "wooden", "wood"]),
    "Chair":       (["chair", "seat", "stool"], []),
    "Plant":       (["plant", "flower", "tree", "pot"], ["green"]),
    "Crate":       (["crate", "pallet"], ["tan", "beige"]),
    "Server Rack": (["server", "rack"], []),
    "Pillar":      (["pillar", "column", "pole", "post"], ["grey", "gray"]),
    "Desk":        (["desk", "workstation"], ["purple"]),
    "Cone":        (["cone", "traffic"], ["yellow"]),
}


def _word_match(tok, kw):
    if tok == kw or tok == kw + "s" or tok == kw + "es":
        return True
    if len(kw) >= 5 and abs(len(tok) - len(kw)) <= 1:
        # one substitution / insertion / deletion
        if len(tok) == len(kw):
            return sum(a != b for a, b in zip(tok, kw)) == 1
        short, long_ = (tok, kw) if len(tok) < len(kw) else (kw, tok)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    return False


def parse_goal(text):
    """Return (canonical object name, matched words) or (None, []) if nothing matches."""
    low = text.lower()
    toks = "".join(c if c.isalpha() else " " for c in low).split()
    best, best_score, best_hits = None, 0, []
    for name, (nouns, attrs) in GOAL_WORDS.items():
        score, hits = 0, []
        if (" " + name.lower() + " ") in (" " + " ".join(toks) + " "):
            score += 5
            hits.append(name.lower())
        for t in toks:
            if any(_word_match(t, k) for k in nouns):
                score += 3
                hits.append(t)
            elif any(_word_match(t, k) for k in attrs):
                score += 2
                hits.append(t)
        if score > best_score:
            best, best_score, best_hits = name, score, hits
    return best, best_hits


# ======================== OBSTACLE CLASS ========================
class Obstacle:
    TYPES = [
        ("Red Barrel",  "circle", RED,    (180, 50, 50)),
        ("Blue Box",    "rect",   BLUE,   (40, 80, 170)),
        ("Table",       "rect",   ORANGE, (190, 110, 25)),
        ("Chair",       "rect",   (100, 160, 80), (70, 130, 55)),
        ("Plant",       "circle", GREEN,  (25, 130, 50)),
        ("Crate",       "rect",   (180, 160, 100), (150, 130, 70)),
        ("Server Rack", "rect",   (100, 100, 120), (70, 70, 90)),
        ("Pillar",      "circle", (160, 160, 170), (130, 130, 140)),
        ("Desk",        "rect",   PURPLE, (100, 45, 150)),
        ("Cone",        "circle", YELLOW, (180, 145, 0)),
    ]

    def __init__(self, name=None, forced_pos=None, exclude=None, rng=random):
        if name:
            t = next(t for t in self.TYPES if t[0] == name)
        else:
            choices = [t for t in self.TYPES if t[0] != exclude]
            t = rng.choice(choices)
        self.name, self.shape, self.color, self.dark = t
        self.w = rng.randint(35, 65)
        self.h = self.w if self.shape == "circle" else rng.randint(30, 60)
        self.r = self.w // 2
        # radius of the circle that encloses the object (used by sensors / collisions)
        self.r_eff = self.r if self.shape == "circle" else math.hypot(self.w, self.h) / 2
        if forced_pos:
            self.x, self.y = forced_pos
        else:
            self.x = rng.randint(80, ARENA_W - 80)
            self.y = rng.randint(80, H - 80)
        self.rect = pygame.Rect(self.x - self.w // 2, self.y - self.h // 2, self.w, self.h)
        self.detected = False
        self.det_alpha = 0

    def contact_normal(self, px, py, radius):
        """If a circle (px, py, radius) overlaps this object, return (nx, ny, depth); else None."""
        if self.shape == "circle":
            dx, dy = px - self.x, py - self.y
            d = math.hypot(dx, dy)
            pen = radius + self.r - d
            if pen > 0:
                return (dx / d, dy / d, pen) if d > 1e-6 else (1.0, 0.0, pen)
            return None
        cx = min(max(px, self.rect.left), self.rect.right)
        cy = min(max(py, self.rect.top), self.rect.bottom)
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)
        if d >= radius:
            return None
        if d > 1e-6:
            return dx / d, dy / d, radius - d
        # centre inside the rectangle: push out along the shortest axis
        left, right = px - self.rect.left, self.rect.right - px
        top, bottom = py - self.rect.top, self.rect.bottom - py
        m = min(left, right, top, bottom)
        if m == left:
            return -1.0, 0.0, left + radius
        if m == right:
            return 1.0, 0.0, right + radius
        if m == top:
            return 0.0, -1.0, top + radius
        return 0.0, 1.0, bottom + radius

    def draw(self, surf, is_target, t):
        shadow_color = (200, 200, 210, 60)
        ss = pygame.Surface((self.w + 6, self.h + 6), pygame.SRCALPHA)
        if self.shape == "circle":
            pygame.draw.circle(ss, shadow_color, (self.w // 2 + 3, self.h // 2 + 3), self.r + 3)
        else:
            pygame.draw.rect(ss, shadow_color, (0, 0, self.w + 6, self.h + 6), border_radius=4)
        surf.blit(ss, (self.x - self.w // 2 - 1, self.y - self.h // 2 + 3))

        if is_target:
            pulse = abs(math.sin(t * 2.5)) * 0.6 + 0.4
            glow_r = int(self.r_eff + 18 * pulse)
            gs = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(gs, (*self.color, int(60 * pulse)), (glow_r + 2, glow_r + 2), glow_r)
            surf.blit(gs, (self.x - glow_r - 2, self.y - glow_r - 2))
            ch_size = int(self.r_eff + 22 + 4 * math.sin(t * 3))
            for angle in (0, 90, 180, 270):
                rad = math.radians(angle)
                x1 = self.x + math.cos(rad) * (ch_size - 8)
                y1 = self.y + math.sin(rad) * (ch_size - 8)
                x2 = self.x + math.cos(rad) * ch_size
                y2 = self.y + math.sin(rad) * ch_size
                pygame.draw.line(surf, self.color, (x1, y1), (x2, y2), 2)

        if self.shape == "circle":
            pygame.draw.circle(surf, self.color, (self.x, self.y), self.r)
            pygame.draw.circle(surf, self.dark, (self.x, self.y), self.r, 2)
            pygame.draw.circle(surf, tuple(min(c + 60, 255) for c in self.color), (self.x - 3, self.y - 3), max(2, self.r // 3))
        else:
            pygame.draw.rect(surf, self.color, self.rect, border_radius=4)
            pygame.draw.rect(surf, self.dark, self.rect, 2, border_radius=4)
            pygame.draw.line(surf, self.dark, (self.rect.left + 4, self.rect.centery), (self.rect.right - 4, self.rect.centery), 1)

        lbl = TINY_FNT.render(self.name, True, TEXT_DIM)
        surf.blit(lbl, (self.x - lbl.get_width() // 2, self.y + self.h // 2 + 6))

        self.det_alpha = min(self.det_alpha + 15, 200) if self.detected else max(self.det_alpha - 10, 0)
        if self.det_alpha > 0:
            det_rect = self.rect.inflate(14, 14)
            ds = pygame.Surface((det_rect.w, det_rect.h), pygame.SRCALPHA)
            color = GREEN if is_target else TEAL
            pygame.draw.rect(ds, (*color, self.det_alpha), (0, 0, det_rect.w, det_rect.h), 2, border_radius=4)
            bk = 10
            for cx, cy in ((0, 0), (det_rect.w, 0), (0, det_rect.h), (det_rect.w, det_rect.h)):
                dx = bk if cx == 0 else -bk
                dy = bk if cy == 0 else -bk
                pygame.draw.line(ds, (*color, self.det_alpha), (cx, cy), (cx + dx, cy), 2)
                pygame.draw.line(ds, (*color, self.det_alpha), (cx, cy), (cx, cy + dy), 2)
            surf.blit(ds, det_rect.topleft)


# ======================== PARTICLES ========================
class Particle:
    def __init__(self, x, y, color, speed=2, life=80):
        self.x, self.y = x, y
        angle = random.uniform(0, 2 * math.pi)
        spd = random.uniform(0.5, speed)
        self.vx, self.vy = math.cos(angle) * spd, math.sin(angle) * spd
        self.life = self.max_life = life
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.02
        self.life -= 1

    def draw(self, surf):
        if self.life > 0:
            a = int(255 * (self.life / self.max_life))
            s = max(1, self.size * (self.life / self.max_life))
            ps = pygame.Surface((int(s * 2 + 2), int(s * 2 + 2)), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*self.color, a), (int(s + 1), int(s + 1)), int(s))
            surf.blit(ps, (self.x - s, self.y - s))


# ======================== ROBOT ========================
CELL = 50            # exploration grid cell (px)
GX, GY = ARENA_W // CELL, H // CELL
N_PROX = 32          # proximity ring: 32 range sensors every 11.25 degrees
PROX_RANGE = 130     # px
SAFE_DIST = 60       # px - readings closer than this (from the robot's edge) are drawn red
SAFE_R = 24          # px - planner safety bubble around the robot centre


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def ray_circle(ox, oy, dx, dy, cx, cy, r):
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    t = -b - s
    if t < 0:
        t = 0.0 if c < 0 else -b + s
    return t if t >= 0 else None


class Robot:
    def __init__(self, rng=random):
        self.rng = rng
        self.x, self.y = 120.0, 120.0
        self.angle = 45.0
        self.speed = 0.0
        self.target_speed = 0.0
        self.max_speed = 3.2
        self.turn_speed = 2.0
        self.radius = 14

        self.vision_range = 220
        self.vision_angle = 70
        self.radar_sweep = 0

        self.state = "SEARCHING"
        self.prev_state = "SEARCHING"
        self.trail = []
        self.detected_objects = []
        self.target_detected = False
        self.target_obj = None
        self.memory = None            # (x, y, time) of the last sighting of the target
        self.prox = [PROX_RANGE] * N_PROX
        self.clear = [PROX_RANGE] * N_PROX
        self.avoiding = False
        self.plan_dir = None

        # exploration memory: which parts of the room the camera has already looked at
        self.seen = [[False] * GY for _ in range(GX)]
        self.seen_count = 0
        self.known_obstacles = {}      # id -> obstacle seen by the camera
        self.waypoint = None
        self.wp_frames = 0
        self.scan_left = 360.0         # start with a 360-degree look-around
        self.blocked = set()
        self.pos_hist = []             # recent positions, for stuck detection
        self.recoveries = 0

        self.distance_traveled = 0.0
        self.collisions = 0
        self.in_contact = set()
        self.frames = 0

    # ---------------- SENSE ----------------
    def _in_cone(self, px, py):
        d = math.hypot(px - self.x, py - self.y)
        if d > self.vision_range:
            return False, d
        a = math.degrees(math.atan2(py - self.y, px - self.x))
        diff = (a - self.angle) % 360
        if diff > 180:
            diff -= 360
        return abs(diff) <= self.vision_angle / 2, d

    def _cone_points(self, segments=20):
        pts = [(self.x, self.y)]
        sa = math.radians(self.angle - self.vision_angle / 2)
        ea = math.radians(self.angle + self.vision_angle / 2)
        for i in range(segments + 1):
            a = sa + (ea - sa) * (i / segments)
            pts.append((self.x + math.cos(a) * self.vision_range, self.y + math.sin(a) * self.vision_range))
        return pts

    def sense(self, obstacles, target_name):
        self.detected_objects = []
        self.target_detected = False
        self.target_obj = None
        for obs in obstacles:
            obs.detected = False
            in_cone, dist = self._in_cone(obs.x, obs.y)
            if in_cone:
                obs.detected = True
                conf = max(40, int(100 - (dist / self.vision_range) * 55))
                self.detected_objects.append({"name": obs.name, "dist": dist, "conf": conf, "obj": obs})
                self.known_obstacles[id(obs)] = obs
                if obs.name == target_name:
                    self.target_detected = True
                    self.target_obj = obs
                    self.memory = (obs.x, obs.y, self.frames)
        self.detected_objects.sort(key=lambda d: d["dist"])

        # 360-degree proximity ring (ray casting against objects and walls)
        # once the target has been seen it is the destination, not an obstacle
        near = [o for o in obstacles if not (self.memory is not None and o.name == target_name)
                and math.hypot(o.x - self.x, o.y - self.y) - o.r_eff < PROX_RANGE]
        for i in range(N_PROX):
            a = 2 * math.pi * i / N_PROX
            dx, dy = math.cos(a), math.sin(a)
            best = PROX_RANGE
            for o in near:
                t = ray_circle(self.x, self.y, dx, dy, o.x, o.y, o.r_eff)
                if t is not None and t < best:
                    best = t
            if dx > 1e-6:
                best = min(best, (ARENA_W - self.x) / dx)
            elif dx < -1e-6:
                best = min(best, -self.x / dx)
            if dy > 1e-6:
                best = min(best, (H - self.y) / dy)
            elif dy < -1e-6:
                best = min(best, -self.y / dy)
            self.prox[i] = best
        # free-space histogram: how far the robot's safety bubble can travel in each direction
        step = 2 * math.pi / N_PROX
        for k in range(N_PROX):
            c = PROX_RANGE
            for j, d in enumerate(self.prox):
                if d >= PROX_RANGE - 1e-6:
                    continue
                da = (j - k) * step
                proj = d * math.cos(da)
                if proj <= 0:
                    continue
                perp = d * math.sin(da)
                if perp * perp < SAFE_R * SAFE_R:
                    c = min(c, proj - math.sqrt(SAFE_R * SAFE_R - perp * perp))
            self.clear[k] = max(0.0, c)
        front = self._front_indices()
        return self.avoiding or min(self.prox[i] for i in front) - self.radius < 30

    # ---------------- THINK ----------------
    def goal_point(self):
        if self.target_obj is not None:
            return self.target_obj.x, self.target_obj.y
        if self.memory is not None:
            return self.memory[0], self.memory[1]
        return None

    def think(self, obstacle_near, target_r):
        self.prev_state = self.state
        if self.state == "GOAL_REACHED":
            return
        goal = self.goal_point()
        if goal is not None and math.hypot(goal[0] - self.x, goal[1] - self.y) < target_r + self.radius + 14:
            if self.target_detected or self.memory is not None:
                self.state = "GOAL_REACHED"
                return
        if obstacle_near:
            self.state = "AVOIDING"
        elif goal is not None:
            self.state = "NAVIGATING"
        else:
            self.state = "SEARCHING"

    # ---------------- ACT ----------------
    def _steer_to(self, heading_deg, max_turn):
        diff = (heading_deg - self.angle) % 360
        if diff > 180:
            diff -= 360
        self.angle += max(-max_turn, min(diff, max_turn))
        return diff

    def act(self, obstacles):
        self.frames += 1
        if self.state == "GOAL_REACHED":
            self.target_speed = 0
            self.speed *= 0.92
            return

        self.update_seen()
        goal = self.goal_point()
        exploring = goal is None
        if exploring and self.scan_left > 0:
            # 360-degree look-around in place
            self.target_speed = 0.0
            self.angle += 3.0
            self.scan_left -= 3.0
        else:
            if exploring:
                self.wp_frames += 1
                wp = self.waypoint
                if wp is not None:
                    i, j = int(wp[0] // CELL), int(wp[1] // CELL)
                    arrived = math.hypot(wp[0] - self.x, wp[1] - self.y) < 40
                    if arrived and self.unseen_near(4) >= 10:
                        self.scan_left = 360.0
                    if arrived or self.seen[i][j] or self.wp_frames > 420:
                        if self.wp_frames > 420:
                            self.blocked.add((i, j))
                        wp = self.waypoint = None
                if wp is None:
                    self.pick_waypoint()
                goal = self.waypoint
            # free-space (VFH-style) planner: pick the open direction closest to the goal
            head = math.radians(self.angle)
            if goal is not None:
                goal_ang = math.atan2(goal[1] - self.y, goal[0] - self.x)
                dist_goal = math.hypot(goal[0] - self.x, goal[1] - self.y)
            else:
                goal_ang, dist_goal = head, PROX_RANGE
            look = max(30.0, min(dist_goal, 100.0))
            step = 2 * math.pi / N_PROX
            best_k, best_cost = None, 1e9
            for k in range(N_PROX):
                c = self.clear[k]
                if c < 6:
                    continue
                a = k * step
                cost = abs(_angdiff(a, goal_ang)) + 0.3 * abs(_angdiff(a, head)) + 2.0 * max(0.0, 1.0 - c / look)
                if cost < best_cost:
                    best_k, best_cost = k, cost
            if best_k is None:                       # boxed in: turn on the spot
                self.plan_dir = None
                self.avoiding = True
                self.target_speed = 0.0
                self.angle += 4.0
            else:
                a = best_k * step
                self.plan_dir = a
                diff = self._steer_to(math.degrees(a), self.turn_speed * 2.5)
                k_goal = int(round(goal_ang / step)) % N_PROX
                self.avoiding = abs(_angdiff(a, goal_ang)) > 0.35 and self.clear[k_goal] < min(dist_goal, 80)
                c = self.clear[best_k]
                self.target_speed = self.max_speed * max(0.2, min(1.0, c / 80.0)) * max(0.15, math.cos(math.radians(diff))) ** 2
                if not exploring and self.state == "NAVIGATING":
                    self.target_speed *= max(0.35, min(1.0, dist_goal / 120))

        self.speed += (self.target_speed - self.speed) * 0.1
        # safety layer (proximity readings only): never creep into something right in front
        head = math.radians(self.angle)
        for i, d in enumerate(self.prox):
            if d - self.radius < 3 and abs(_angdiff(i * 2 * math.pi / N_PROX, head)) < math.radians(75):
                self.speed = min(self.speed, 0.0)
                break
        old_x, old_y = self.x, self.y
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed

        # physical contact resolution (counted honestly as collisions)
        touching = set()
        for o in obstacles:
            hit = o.contact_normal(self.x, self.y, self.radius)
            if hit:
                nx, ny, depth = hit
                self.x += nx * depth
                self.y += ny * depth
                touching.add(id(o))
                if id(o) not in self.in_contact:
                    self.collisions += 1
        self.in_contact = touching
        self.x = max(self.radius, min(ARENA_W - self.radius, self.x))
        self.y = max(self.radius, min(H - self.radius, self.y))
        self.distance_traveled += math.hypot(self.x - old_x, self.y - old_y)

        # stuck detection: < 20 px of progress in 2 s while trying to move -> re-plan
        if self.scan_left > 0:
            self.pos_hist.clear()          # spinning in place on purpose is not "stuck"
        elif self.frames % 20 == 0:
            self.pos_hist.append((self.x, self.y))
            if len(self.pos_hist) > 6:
                self.pos_hist.pop(0)
            if len(self.pos_hist) == 6 and self.scan_left <= 0 and \
                    math.hypot(self.x - self.pos_hist[0][0], self.y - self.pos_hist[0][1]) < 20:
                self.recoveries += 1
                self.pos_hist.clear()
                self.angle += 150
                if self.waypoint is not None:
                    self.blocked.add((int(self.waypoint[0] // CELL), int(self.waypoint[1] // CELL)))
                    self.waypoint = None

        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 6:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 160:
                self.trail.pop(0)
        self.radar_sweep = (self.radar_sweep + 2) % 360

    # ---------------- exploration memory ----------------
    def update_seen(self):
        r = self.vision_range
        i0, i1 = max(0, int((self.x - r) // CELL)), min(GX - 1, int((self.x + r) // CELL))
        j0, j1 = max(0, int((self.y - r) // CELL)), min(GY - 1, int((self.y + r) // CELL))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if self.seen[i][j]:
                    continue
                cx, cy = (i + 0.5) * CELL, (j + 0.5) * CELL
                inside, d = self._in_cone(cx, cy)
                if inside or d < 45:
                    self.seen[i][j] = True
                    self.seen_count += 1

    def unseen_near(self, rad_cells):
        ci, cj = int(self.x // CELL), int(self.y // CELL)
        return sum(1 for i in range(max(0, ci - rad_cells), min(GX, ci + rad_cells + 1))
                   for j in range(max(0, cj - rad_cells), min(GY, cj + rad_cells + 1)) if not self.seen[i][j])

    def pick_waypoint(self):
        """Frontier exploration: head for the unexplored area with the best gain per distance."""
        best, best_score = None, -1.0
        for i in range(GX):
            for j in range(GY):
                if self.seen[i][j] or (i, j) in self.blocked:
                    continue
                cx, cy = (i + 0.5) * CELL, (j + 0.5) * CELL
                d = math.hypot(cx - self.x, cy - self.y)
                if d < 60:
                    continue
                if any(math.hypot(cx - o.x, cy - o.y) < o.r_eff + 30 for o in self.known_obstacles.values()):
                    continue
                gain = sum(1 for a in range(max(0, i - 2), min(GX, i + 3))
                           for b in range(max(0, j - 2), min(GY, j + 3)) if not self.seen[a][b])
                score = gain / (d + 150)
                if score > best_score:
                    best, best_score = (cx, cy), score
        if best is None:                    # everything explored: forget and look again
            self.seen = [[False] * GY for _ in range(GX)]
            self.seen_count = 0
            self.blocked.clear()
            self.scan_left = 360.0
        self.waypoint = best
        self.wp_frames = 0

    def _front_indices(self):
        heading = math.radians(self.angle) % (2 * math.pi)
        k = int(round(heading / (2 * math.pi / N_PROX))) % N_PROX
        return [(k - 1) % N_PROX, k, (k + 1) % N_PROX]

    # ---------------- DRAW ----------------
    def draw(self, surf, overlay, t):
        for i, (tx, ty) in enumerate(self.trail):
            progress = i / max(len(self.trail), 1)
            pygame.draw.circle(overlay, (*TEAL, int(120 * progress)), (int(tx), int(ty)), int(2 + progress * 2))

        # proximity ring rays
        for i, d in enumerate(self.prox):
            a = 2 * math.pi * i / N_PROX
            close = d - self.radius < SAFE_DIST
            col = (220, 55, 55, 90) if close else (0, 145, 190, 30)
            pygame.draw.line(overlay, col, (self.x, self.y), (self.x + math.cos(a) * d, self.y + math.sin(a) * d), 1)

        cone_color = (220, 55, 55, 25) if self.state == "AVOIDING" else (0, 145, 190, 18)
        pts = self._cone_points()
        pygame.draw.polygon(overlay, cone_color, pts)
        bc = (220, 55, 55, 50) if self.state == "AVOIDING" else (0, 145, 190, 40)
        pygame.draw.lines(overlay, bc, False, pts[1:], 1)

        sweep_a = math.radians(self.angle - self.vision_angle / 2 + (self.radar_sweep / 360) * self.vision_angle)
        pygame.draw.line(overlay, (*TEAL, 60), (self.x, self.y),
                         (self.x + math.cos(sweep_a) * self.vision_range, self.y + math.sin(sweep_a) * self.vision_range), 1)

        for d in self.detected_objects:
            obs = d["obj"]
            color = GREEN if obs is self.target_obj else TEAL
            dx, dy = obs.x - self.x, obs.y - self.y
            dist = math.hypot(dx, dy)
            steps = int(dist / 8)
            for s in range(0, steps, 2):
                p1, p2 = s / steps, min((s + 1) / steps, 1)
                pygame.draw.line(overlay, (*color, 70), (self.x + dx * p1, self.y + dy * p1), (self.x + dx * p2, self.y + dy * p2), 1)

        # exploration waypoint (where it is heading to look next)
        if self.waypoint is not None and self.memory is None and self.state != "GOAL_REACHED":
            wx, wy = int(self.waypoint[0]), int(self.waypoint[1])
            pygame.draw.line(overlay, (*ORANGE, 170), (wx - 7, wy - 7), (wx + 7, wy + 7), 2)
            pygame.draw.line(overlay, (*ORANGE, 170), (wx + 7, wy - 7), (wx - 7, wy + 7), 2)
            pygame.draw.circle(overlay, (*ORANGE, 110), (wx, wy), 13, 1)

        # remembered target position
        if self.target_obj is None and self.memory is not None and self.state != "GOAL_REACHED":
            mx, my = int(self.memory[0]), int(self.memory[1])
            pygame.draw.circle(overlay, (*GREEN, 120), (mx, my), 10, 2)
            pygame.draw.line(overlay, (*GREEN, 90), (self.x, self.y), (mx, my), 1)

        surf.blit(overlay, (0, 0))

        pygame.draw.circle(surf, TEAL_LIGHT, (int(self.x), int(self.y)), self.radius + 4)
        pygame.draw.circle(surf, TEAL, (int(self.x), int(self.y)), self.radius + 4, 2)
        pygame.draw.circle(surf, (255, 255, 255), (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surf, TEAL, (int(self.x), int(self.y)), self.radius, 2)
        rad = math.radians(self.angle)
        tip = (self.x + math.cos(rad) * (self.radius + 10), self.y + math.sin(rad) * (self.radius + 10))
        left = (self.x + math.cos(rad + 2.6) * (self.radius - 2), self.y + math.sin(rad + 2.6) * (self.radius - 2))
        right = (self.x + math.cos(rad - 2.6) * (self.radius - 2), self.y + math.sin(rad - 2.6) * (self.radius - 2))
        pygame.draw.polygon(surf, TEAL, [tip, left, right])
        pygame.draw.circle(surf, TEAL, (int(self.x), int(self.y)), 3)

        sc = STATE_COLORS.get(self.state, TEXT)
        state_lbl = TINY_FNT.render(self.state, True, sc)
        lw, lh = state_lbl.get_size()
        lbg = pygame.Surface((lw + 8, lh + 4), pygame.SRCALPHA)
        pygame.draw.rect(lbg, (255, 255, 255, 200), (0, 0, lw + 8, lh + 4), border_radius=3)
        pygame.draw.rect(lbg, (*sc, 150), (0, 0, lw + 8, lh + 4), 1, border_radius=3)
        surf.blit(lbg, (self.x - lw // 2 - 4, self.y - self.radius - 24))
        surf.blit(state_lbl, (self.x - lw // 2, self.y - self.radius - 22))


# ======================== HUD ========================
def draw_hud(surf, robot, target_name, prompt_text, events_log, elapsed, seen_total):
    x0 = ARENA_W + 1
    pygame.draw.rect(surf, PANEL_BG, (x0, 0, HUD_W, H))
    pygame.draw.line(surf, BORDER, (x0, 0), (x0, H), 2)
    x, y = x0 + 16, 16

    surf.blit(TITLE_FNT.render("SightSeeker", True, TEAL), (x, y))
    y += 36
    surf.blit(TINY_FNT.render("Autonomous Vision Robot - camera + proximity ring", True, TEXT_DIM), (x, y))
    y += 22
    pygame.draw.line(surf, BORDER, (x, y), (x0 + HUD_W - 16, y), 1)
    y += 12

    sc = STATE_COLORS.get(robot.state, TEXT)
    surf.blit(HEAD_FNT.render("STATE", True, TEXT_DIM), (x, y))
    y += 20
    state_text = BIG_FNT.render(robot.state, True, sc)
    sw = state_text.get_width()
    badge_bg = pygame.Surface((sw + 16, 28), pygame.SRCALPHA)
    pygame.draw.rect(badge_bg, (*sc, 30), (0, 0, sw + 16, 28), border_radius=6)
    pygame.draw.rect(badge_bg, (*sc, 100), (0, 0, sw + 16, 28), 1, border_radius=6)
    surf.blit(badge_bg, (x, y))
    surf.blit(state_text, (x + 8, y + 2))
    y += 38

    surf.blit(HEAD_FNT.render("GOAL", True, TEXT_DIM), (x, y))
    y += 20
    surf.blit(ICON_FNT.render("->", True, TEAL), (x, y))
    surf.blit(MAIN_FNT.render(target_name, True, TEXT), (x + 24, y + 2))
    y += 20
    if prompt_text and prompt_text.lower() != target_name.lower():
        surf.blit(TINY_FNT.render(f'from: "{prompt_text[:40]}"', True, TEXT_DIM), (x, y))
    y += 16
    goal = robot.goal_point()
    if goal is not None:
        dist = math.hypot(goal[0] - robot.x, goal[1] - robot.y)
        how = "in view" if robot.target_detected else "from memory"
        surf.blit(SMALL_FNT.render(f"Distance: {dist:.0f}px ({how})", True, GREEN), (x, y))
    else:
        surf.blit(SMALL_FNT.render("Target not seen yet", True, TEXT_DIM), (x, y))
    y += 20
    pygame.draw.line(surf, BORDER, (x, y), (x0 + HUD_W - 16, y), 1)
    y += 10

    surf.blit(HEAD_FNT.render("STATS", True, TEXT_DIM), (x, y))
    y += 20
    fps = clock.get_fps()
    stats = [
        (f"Position:   ({int(robot.x)}, {int(robot.y)})", TEXT),
        (f"Heading:    {int(robot.angle % 360)} deg", TEXT),
        (f"Speed:      {robot.speed:.1f} px/frame", TEXT),
        (f"Distance:   {robot.distance_traveled:.0f} px", TEXT),
        (f"Collisions: {robot.collisions}", GREEN if robot.collisions == 0 else RED),
        (f"Explored:   {100 * robot.seen_count // (GX * GY)}%", TEXT),
        (f"Seen types: {seen_total}", TEXT),
        (f"FPS:        {int(fps)}", GREEN if fps > 30 else RED),
        (f"Elapsed:    {elapsed:.0f}s", TEXT),
    ]
    for text, color in stats:
        surf.blit(SMALL_FNT.render(text, True, color), (x, y))
        y += 17
    y += 6
    pygame.draw.line(surf, BORDER, (x, y), (x0 + HUD_W - 16, y), 1)
    y += 10

    surf.blit(HEAD_FNT.render("DETECTED (camera)", True, TEXT_DIM), (x, y))
    y += 20
    if not robot.detected_objects:
        surf.blit(SMALL_FNT.render("Scanning...", True, TEXT_DIM), (x, y))
        y += 17
    else:
        for obj in robot.detected_objects[:5]:
            conf_color = GREEN if obj["conf"] > 70 else YELLOW
            pygame.draw.circle(surf, conf_color, (x + 5, y + 6), 4)
            name = ("* " if obj["obj"] is robot.target_obj else "") + obj["name"]
            surf.blit(SMALL_FNT.render(name, True, TEXT), (x + 14, y))
            surf.blit(TINY_FNT.render(f"{obj['conf']}%", True, conf_color), (x + 150, y + 1))
            y += 17
    y += 6
    pygame.draw.line(surf, BORDER, (x, y), (x0 + HUD_W - 16, y), 1)
    y += 10

    surf.blit(HEAD_FNT.render("EVENT LOG", True, TEXT_DIM), (x, y))
    y += 20
    mm_size = 130
    mm_y = H - mm_size - 40
    for ev_time, ev_text, ev_color in events_log[-6:]:
        if y > mm_y - 34:
            break
        alpha = max(90, 255 - int((time.time() - ev_time) * 20))
        ev_surf = TINY_FNT.render(ev_text, True, ev_color)
        ev_surf.set_alpha(alpha)
        surf.blit(ev_surf, (x, y))
        y += 14

    mm_x = x0 + (HUD_W - mm_size) // 2
    surf.blit(HEAD_FNT.render("MINIMAP", True, TEXT_DIM), (mm_x, mm_y - 20))
    pygame.draw.rect(surf, (255, 255, 255), (mm_x, mm_y, mm_size, mm_size), border_radius=4)
    pygame.draw.rect(surf, BORDER, (mm_x, mm_y, mm_size, mm_size), 1, border_radius=4)
    sx_ratio, sy_ratio = mm_size / ARENA_W, mm_size / H
    cw, ch = max(1, int(CELL * sx_ratio) + 1), max(1, int(CELL * sy_ratio) + 1)
    for i in range(GX):
        for j in range(GY):
            if robot.seen[i][j]:
                pygame.draw.rect(surf, (214, 238, 246), (mm_x + int(i * CELL * sx_ratio), mm_y + int(j * CELL * sy_ratio), cw, ch))
    if robot.waypoint is not None and robot.state != "GOAL_REACHED":
        wx, wy = mm_x + int(robot.waypoint[0] * sx_ratio), mm_y + int(robot.waypoint[1] * sy_ratio)
        pygame.draw.line(surf, ORANGE, (wx - 3, wy - 3), (wx + 3, wy + 3), 2)
        pygame.draw.line(surf, ORANGE, (wx + 3, wy - 3), (wx - 3, wy + 3), 2)
    for obj in robot.detected_objects:
        obs = obj["obj"]
        pygame.draw.circle(surf, obs.color, (mm_x + int(obs.x * sx_ratio), mm_y + int(obs.y * sy_ratio)), 3)
    rx, ry = mm_x + int(robot.x * sx_ratio), mm_y + int(robot.y * sy_ratio)
    pygame.draw.circle(surf, TEAL, (rx, ry), 4)
    rad = math.radians(robot.angle)
    pygame.draw.line(surf, TEAL, (rx, ry), (rx + int(math.cos(rad) * 15), ry + int(math.sin(rad) * 15)), 1)

    hint = TINY_FNT.render("R: new goal   SPACE: pause   Q: quit", True, TEXT_DIM)
    surf.blit(hint, (x0 + HUD_W // 2 - hint.get_width() // 2, H - 18))


# ======================== WORLD + GOAL INPUT ========================
def get_goal():
    """Natural-language goal entry. Returns (canonical target name, raw prompt)."""
    user_text = ""
    error = ""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    raw = user_text.strip()
                    if not raw:
                        return "Red Barrel", ""
                    name, _ = parse_goal(raw)
                    if name:
                        return name, raw
                    error = f'I can\'t find "{raw[:30]}" in this room - try one of the objects below.'
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.unicode and event.unicode.isprintable() and len(user_text) < 48:
                    user_text += event.unicode

        screen.fill(BG)
        card_w, card_h = 640, 250
        cx, cy = W // 2 - card_w // 2, H // 2 - card_h // 2
        pygame.draw.rect(screen, (255, 255, 255), (cx, cy, card_w, card_h), border_radius=12)
        pygame.draw.rect(screen, TEAL, (cx, cy, card_w, card_h), 2, border_radius=12)
        title = TITLE_FNT.render("SightSeeker", True, TEAL)
        screen.blit(title, (W // 2 - title.get_width() // 2, cy + 20))
        prompt = MAIN_FNT.render('Tell the robot what to find (e.g. "find the red barrel"):', True, TEXT)
        screen.blit(prompt, (W // 2 - prompt.get_width() // 2, cy + 70))
        ib = pygame.Rect(W // 2 - 220, cy + 100, 440, 36)
        pygame.draw.rect(screen, (245, 246, 250), ib, border_radius=6)
        pygame.draw.rect(screen, TEAL, ib, 2, border_radius=6)
        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        screen.blit(MAIN_FNT.render(user_text + cursor, True, TEXT), (ib.x + 10, ib.y + 9))
        if user_text.strip():
            name, hits = parse_goal(user_text)
            preview = f"-> understood as: {name}" if name else "-> (no known object yet)"
            screen.blit(SMALL_FNT.render(preview, True, GREEN if name else TEXT_DIM), (ib.x + 4, ib.y + 42))
        if error:
            err = SMALL_FNT.render(error, True, RED)
            screen.blit(err, (W // 2 - err.get_width() // 2, cy + 165))
        hint = SMALL_FNT.render("Red Barrel / Blue Box / Table / Chair / Plant / Crate / Server Rack / Desk / Pillar / Cone", True, TEXT_DIM)
        screen.blit(hint, (W // 2 - hint.get_width() // 2, cy + 190))
        enter_hint = SMALL_FNT.render("Press ENTER to start (empty = Red Barrel)", True, TEXT_DIM)
        screen.blit(enter_hint, (W // 2 - enter_hint.get_width() // 2, cy + 215))
        pygame.display.flip()
        clock.tick(30)


def setup_world(target_name, rng=random):
    """One instance of the target (far from the start) + 9 other objects that never overlap."""
    obstacles = [Obstacle(name=target_name, rng=rng,
                          forced_pos=(rng.randint(ARENA_W // 2, ARENA_W - 100), rng.randint(H // 2, H - 100)))]
    attempts = 0
    while len(obstacles) < 10 and attempts < 400:
        attempts += 1
        obs = Obstacle(exclude=target_name, rng=rng)
        if any(math.hypot(obs.x - e.x, obs.y - e.y) < obs.r_eff + e.r_eff + 45 for e in obstacles):
            continue
        if math.hypot(obs.x - 120, obs.y - 120) < 110:
            continue
        obstacles.append(obs)
    return obstacles


# ======================== HEADLESS BENCHMARK ========================
def run_benchmark(n, max_seconds=90):
    names = [t[0] for t in Obstacle.TYPES]
    results = []
    for i in range(n):
        rng = random.Random(1000 + i)
        target = names[i % len(names)]
        obstacles = setup_world(target, rng)
        tgt = obstacles[0]
        robot = Robot(rng)
        ok = False
        for _ in range(max_seconds * FPS):
            near = robot.sense(obstacles, target)
            robot.think(near, tgt.r_eff)
            robot.act(obstacles)
            if robot.state == "GOAL_REACHED":
                ok = True
                break
        results.append((ok, robot.frames / FPS, robot.collisions, robot.recoveries))
    times = sorted(r[1] for r in results if r[0])
    succ = sum(1 for r in results if r[0])
    coll = sum(r[2] for r in results)
    med = times[len(times) // 2] if times else float("nan")
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))] if times else float("nan")
    print(f"Trials: {n} | success: {succ}/{n} ({100 * succ / n:.0f}%) | median time: {med:.1f}s | "
          f"90th pct: {p90:.1f}s | collisions: {coll} | stuck recoveries: {sum(r[3] for r in results)}")
    return results


# ======================== MAIN ========================
def main():
    if BENCHMARK:
        idx = sys.argv.index("--benchmark")
        n = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 and sys.argv[idx + 1].isdigit() else 100
        run_benchmark(n)
        pygame.quit()
        return

    target_name, prompt_text = get_goal()
    obstacles = setup_world(target_name)
    robot = Robot()
    particles = []
    start_time = time.time()
    paused = False
    events_log = [(time.time(), f"Goal set: {target_name}", TEAL), (time.time(), "Robot initialized", TEXT_DIM)]
    seen_types = set()
    overlay = pygame.Surface((ARENA_W, H), pygame.SRCALPHA)
    last_coll = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    target_name, prompt_text = get_goal()
                    obstacles = setup_world(target_name)
                    robot = Robot()
                    particles = []
                    events_log = [(time.time(), f"Reset. Goal: {target_name}", TEAL)]
                    seen_types = set()
                    start_time = time.time()
                    last_coll = 0

        t = time.time() - start_time
        target_obj = obstacles[0]

        if not paused:
            obstacle_near = robot.sense(obstacles, target_name)
            robot.think(obstacle_near, target_obj.r_eff)
            robot.act(obstacles)
            for d in robot.detected_objects:
                if d["name"] not in seen_types:
                    seen_types.add(d["name"])
                    events_log.append((time.time(), f"Detected {d['name']} ({d['conf']}%)", BLUE))
            if robot.state != robot.prev_state:
                events_log.append((time.time(), f"State -> {robot.state}", STATE_COLORS.get(robot.state, TEXT)))
                if robot.state == "GOAL_REACHED":
                    events_log.append((time.time(), f"TARGET REACHED in {t:.1f}s", GREEN))
            if robot.collisions != last_coll:
                last_coll = robot.collisions
                events_log.append((time.time(), "Contact! (collision counted)", RED))
            if robot.state == "GOAL_REACHED" and random.random() < 0.4:
                for _ in range(3):
                    particles.append(Particle(robot.x, robot.y, random.choice([TEAL, GREEN, BLUE]), speed=3, life=60))
            for p in particles:
                p.update()
            particles = [p for p in particles if p.life > 0]

        # ---- DRAW ----
        screen.fill(ARENA_BG)
        for gx in range(0, ARENA_W, 40):
            pygame.draw.line(screen, GRID_COLOR, (gx, 0), (gx, H))
        for gy in range(0, H, 40):
            pygame.draw.line(screen, GRID_COLOR, (0, gy), (ARENA_W, gy))
        pygame.draw.rect(screen, BORDER, (0, 0, ARENA_W, H), 2)
        for obs in obstacles:
            obs.draw(screen, obs is target_obj, t)
        overlay.fill((0, 0, 0, 0))
        robot.draw(screen, overlay, t)
        for p in particles:
            p.draw(screen)
        if paused:
            ps = pygame.Surface((ARENA_W, H), pygame.SRCALPHA)
            ps.fill((255, 255, 255, 100))
            screen.blit(ps, (0, 0))
            ptxt = BIG_FNT.render("PAUSED", True, TEXT)
            screen.blit(ptxt, (ARENA_W // 2 - ptxt.get_width() // 2, H // 2))
        draw_hud(screen, robot, target_name, prompt_text, events_log, t, len(seen_types))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
