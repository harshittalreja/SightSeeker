"""
SightSeeker - Mission Control Dashboard (Pygame)
No web server needed - runs directly as a desktop app.
Run: python dashboard.py
"""

import math
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pygame

pygame.init()

WIDTH, HEIGHT = 1000, 700
FPS = 30

# Colors - light theme
BG = (245, 245, 250)
CARD = (232, 233, 241)
CYAN = (0, 120, 165)
TEXT = (30, 30, 40)
GRAY = (110, 110, 128)
DARK_GRAY = (150, 150, 165)
GREEN = (25, 140, 55)
YELLOW = (180, 135, 0)
RED = (210, 50, 50)
BLUE = (40, 90, 200)
ORANGE = (210, 120, 25)
PURPLE = (120, 70, 190)
WHITE = (255, 255, 255)

TITLE_FONT = pygame.font.SysFont("impact", 48)
HEADING_FONT = pygame.font.SysFont("consolas", 21, bold=True)
MAIN_FONT = pygame.font.SysFont("consolas", 16)
SMALL_FONT = pygame.font.SysFont("consolas", 13)
TINY_FONT = pygame.font.SysFont("consolas", 11)
BIG_NUM_FONT = pygame.font.SysFont("impact", 38)

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable            # the same Python that is running this dashboard

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SightSeeker - Mission Control")
clock = pygame.time.Clock()

# Measured results (randomized trials). Re-run them yourself: 3D sim -> "Trials" button,
# or the "Run 2D benchmark" button on this dashboard.
MEASURED = {
    "3d_trials": 1000, "3d_success": "100%", "3d_collisions": "0", "3d_median": "51 s",
    "2d_trials": 300, "2d_success": "100%", "2d_median": "7.5 s",
}

status_msg = ""
status_until = 0.0
bench_state = {"running": False, "result": ""}


def flash(msg, seconds=4.0):
    global status_msg, status_until
    status_msg, status_until = msg, time.time() + seconds


def launch_python(script, *args):
    path = PROJECT_DIR / script
    if not path.exists():
        flash(f"Missing file: {script}")
        return
    try:
        subprocess.Popen([PYTHON, str(path), *args], cwd=str(PROJECT_DIR))
        flash(f"Launched {script}")
    except OSError as e:
        flash(f"Could not launch {script}: {e}")


def launch_html(name):
    path = PROJECT_DIR / name
    if not path.exists():
        flash(f"Missing file: {name}")
        return
    webbrowser.open(path.as_uri())
    flash(f"Opened {name} in your browser")


def run_benchmark_2d():
    if bench_state["running"]:
        return
    bench_state["running"] = True
    bench_state["result"] = "Running 100 randomized 2D trials..."

    def worker():
        try:
            out = subprocess.run([PYTHON, str(PROJECT_DIR / "robot_simulation.py"), "--benchmark", "100"],
                                 cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=600)
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith("Trials:")]
            bench_state["result"] = lines[-1] if lines else (out.stderr.strip().splitlines() or ["Benchmark failed"])[-1]
        except Exception as e:  # noqa: BLE001
            bench_state["result"] = f"Benchmark failed: {e}"
        bench_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()


# ----------------------------------------------------------------------------- drawing helpers
def draw_rounded_rect(surface, color, rect, radius=10, alpha=255):
    if alpha < 255:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
        surface.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_outlined_rect(surface, color, rect, radius=10, width=2):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def draw_metric_card(surface, x, y, w, h, value, label, sub, color=CYAN):
    draw_rounded_rect(surface, CARD, (x, y, w, h), 12)
    draw_outlined_rect(surface, color, (x, y, w, h), 12, 1)
    val = BIG_NUM_FONT.render(str(value), True, color)
    surface.blit(val, (x + w // 2 - val.get_width() // 2, y + 6))
    lbl = SMALL_FONT.render(label, True, TEXT)
    surface.blit(lbl, (x + w // 2 - lbl.get_width() // 2, y + h - 38))
    sb = TINY_FONT.render(sub, True, GRAY)
    surface.blit(sb, (x + w // 2 - sb.get_width() // 2, y + h - 20))


def draw_button(surface, rect, text, color, hover):
    bg = tuple(min(c + 25, 255) for c in color) if hover else color
    draw_rounded_rect(surface, bg, rect, 8)
    txt = MAIN_FONT.render(text, True, WHITE)
    surface.blit(txt, (rect.x + rect.w // 2 - txt.get_width() // 2, rect.y + rect.h // 2 - txt.get_height() // 2))


def draw_state_indicator(surface, x, y, state_name, state_color, description):
    pygame.draw.circle(surface, state_color, (x + 8, y + 8), 6)
    surface.blit(MAIN_FONT.render(state_name, True, state_color), (x + 22, y))
    surface.blit(SMALL_FONT.render(description, True, GRAY), (x + 22, y + 19))


def draw_tech_badge(surface, x, y, text, color=CYAN):
    txt = SMALL_FONT.render(text, True, color)
    tw = txt.get_width() + 16
    draw_outlined_rect(surface, color, (x, y, tw, 24), 12, 1)
    surface.blit(txt, (x + 8, y + 4))
    return tw + 6


def draw_flow_arrow(surface, x1, y1, x2, y2, color=CYAN):
    pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)
    angle = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - 8 * math.cos(angle - 0.4), y2 - 8 * math.sin(angle - 0.4))
    p2 = (x2 - 8 * math.cos(angle + 0.4), y2 - 8 * math.sin(angle + 0.4))
    pygame.draw.polygon(surface, color, [(x2, y2), p1, p2])


def draw_flow_box(surface, x, y, w, h, text, color):
    draw_rounded_rect(surface, CARD, (x, y, w, h), 6)
    draw_outlined_rect(surface, color, (x, y, w, h), 6, 2)
    txt = SMALL_FONT.render(text, True, TEXT)
    surface.blit(txt, (x + w // 2 - txt.get_width() // 2, y + h // 2 - txt.get_height() // 2))
    return x + w // 2, y + h


# ----------------------------------------------------------------------------- main loop
current_tab = 0
tab_names = ["Overview", "Architecture", "Demo Plan"]
start_time = time.time()

running = True
while running:
    mx, my = pygame.mouse.get_pos()
    clicked = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.key == pygame.K_1:
                current_tab = 0
            elif event.key == pygame.K_2:
                current_tab = 1
            elif event.key == pygame.K_3:
                current_tab = 2
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked = True

    t = time.time() - start_time
    screen.fill(BG)

    # ---------------- header ----------------
    pulse = int(abs(math.sin(t * 1.5)) * 45)
    title = TITLE_FONT.render("SightSeeker", True, (0, 105 + pulse, 150 + pulse))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))
    subtitle = SMALL_FONT.render("Vision-Guided Autonomous Navigator  |  Hack in Hills '26  |  HARSHIT", True, GRAY)
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 62))
    pygame.draw.line(screen, (210, 215, 225), (50, 84), (WIDTH - 50, 84), 1)

    # ---------------- tabs ----------------
    tab_y, tab_w = 94, 150
    for i, name in enumerate(tab_names):
        tx = WIDTH // 2 - (len(tab_names) * tab_w) // 2 + i * tab_w + 5
        rect = pygame.Rect(tx, tab_y, tab_w - 10, 30)
        if i == current_tab:
            draw_rounded_rect(screen, (200, 230, 245), rect, 6)
        elif rect.collidepoint(mx, my):
            draw_rounded_rect(screen, (222, 222, 232), rect, 6)
        tab_txt = MAIN_FONT.render(f"{i + 1} {name}", True, CYAN if i == current_tab else GRAY)
        screen.blit(tab_txt, (rect.x + rect.w // 2 - tab_txt.get_width() // 2, tab_y + 6))
        if clicked and rect.collidepoint(mx, my):
            current_tab = i

    content_y = 138

    # ======================= OVERVIEW =======================
    if current_tab == 0:
        card_w, card_h, gap = 210, 96, 20
        sx = WIDTH // 2 - (4 * card_w + 3 * gap) // 2
        draw_metric_card(screen, sx, content_y, card_w, card_h, MEASURED["3d_success"], "Missions completed",
                         f"{MEASURED['3d_trials']} random 3D warehouses", GREEN)
        draw_metric_card(screen, sx + (card_w + gap), content_y, card_w, card_h, MEASURED["3d_collisions"], "Collisions",
                         f"across {MEASURED['3d_trials']} 3D trials", CYAN)
        draw_metric_card(screen, sx + 2 * (card_w + gap), content_y, card_w, card_h, MEASURED["3d_median"], "Median time to goal",
                         "robot-time, 120 m arena", ORANGE)
        draw_metric_card(screen, sx + 3 * (card_w + gap), content_y, card_w, card_h, "55.0", "YOLO26l mAP50-95",
                         "COCO, on the NVIDIA GPU", PURPLE)

        sm_y = content_y + card_h + 22
        screen.blit(HEADING_FONT.render("Robot State Machine", True, TEXT), (50, sm_y))
        states = [
            ("SEARCHING", YELLOW, "360 deg scan + frontier exploration"),
            ("NAVIGATING", GREEN, "Target seen/remembered -> drive to it"),
            ("AVOIDING", RED, "LiDAR free-space planner + safety stop"),
            ("GOAL_REACHED", CYAN, "Stop at the target, report the mission"),
        ]
        for i, (name, color, desc) in enumerate(states):
            draw_state_indicator(screen, 66, sm_y + 34 + i * 44, name, color, desc)

        feat_x = WIDTH // 2 + 40
        screen.blit(HEADING_FONT.render("What actually runs", True, TEXT), (feat_x, sm_y))
        features = [
            ("YOLO26 on webcam", "real detections -> steering"),
            ("Language goals", "'find my phone' -> class"),
            ("Sensor fusion", "camera + 360 deg LiDAR"),
            ("Semantic map", "remembers what it has seen"),
            ("Safety layer", "stops before any contact"),
            ("Arduino bridge", "L/R wheel PWM over serial"),
        ]
        for i, (feat, desc) in enumerate(features):
            fy = sm_y + 34 + i * 27
            pygame.draw.circle(screen, CYAN, (feat_x + 8, fy + 8), 3)
            ft = MAIN_FONT.render(feat, True, TEXT)
            screen.blit(ft, (feat_x + 20, fy))
            screen.blit(SMALL_FONT.render(f"  {desc}", True, GRAY), (feat_x + 20 + ft.get_width(), fy + 2))

        # Launch buttons
        btn_top = 488
        pygame.draw.line(screen, (210, 215, 225), (50, btn_top - 10), (WIDTH - 50, btn_top - 10), 1)
        screen.blit(HEADING_FONT.render("Launch Demo", True, TEXT), (50, btn_top))
        bw, bh, bg_ = 218, 42, 14
        bx0 = WIDTH // 2 - (4 * bw + 3 * bg_) // 2
        by = btn_top + 32
        buttons = [
            ("3D Simulator", (0, 120, 170), lambda: launch_html("robot_simulation_3d.html")),
            ("Vision Brain (webcam)", (205, 110, 20), lambda: launch_python("yolo_detection.py")),
            ("2D Simulator", (40, 120, 70), lambda: launch_python("robot_simulation.py")),
            ("Slides", (110, 70, 180), lambda: launch_html("presentation_mascot.html")),
        ]
        for i, (label, color, action) in enumerate(buttons):
            rect = pygame.Rect(bx0 + i * (bw + bg_), by, bw, bh)
            draw_button(screen, rect, label, color, rect.collidepoint(mx, my))
            if clicked and rect.collidepoint(mx, my):
                action()

        brect = pygame.Rect(bx0, by + bh + 12, 260, 32)
        draw_button(screen, brect, "Run 2D benchmark (100)" if not bench_state["running"] else "Running...",
                    (90, 95, 110), brect.collidepoint(mx, my))
        if clicked and brect.collidepoint(mx, my):
            run_benchmark_2d()
        res = bench_state["result"] or f"Last measured: {MEASURED['2d_trials']} trials, {MEASURED['2d_success']} success, median {MEASURED['2d_median']}"
        screen.blit(SMALL_FONT.render(res[:72], True, GRAY), (brect.right + 12, brect.y + 8))

        tx = 50
        for tech in ["Python 3.12", "Pygame", "Ultralytics YOLO26", "OpenCV", "Three.js", "NumPy", "Arduino"]:
            tx += draw_tech_badge(screen, tx, HEIGHT - 62, tech)

    # ======================= ARCHITECTURE =======================
    elif current_tab == 1:
        screen.blit(HEADING_FONT.render("System Architecture: Sense -> Think -> Act", True, TEXT), (50, content_y))
        box_w, box_h = 190, 36
        cx = WIDTH // 2
        cy = content_y + 46
        b1 = draw_flow_box(screen, cx - box_w - 20, cy, box_w, box_h, "Camera frame", BLUE)
        b1b = draw_flow_box(screen, cx + 20, cy, box_w, box_h, "Text goal: 'find ...'", BLUE)
        cy2 = cy + box_h + 34
        b2 = draw_flow_box(screen, cx - box_w - 20, cy2, box_w, box_h, "YOLO26 detection", GREEN)
        b3 = draw_flow_box(screen, cx + 20, cy2, box_w, box_h, "Language grounding", ORANGE)
        draw_flow_arrow(screen, b1[0], b1[1], b2[0], cy2, BLUE)
        draw_flow_arrow(screen, b1b[0], b1b[1], b3[0], cy2, BLUE)
        cy3 = cy2 + box_h + 34
        b4 = draw_flow_box(screen, cx - box_w // 2, cy3, box_w, box_h, "State machine (FSM)", YELLOW)
        draw_flow_arrow(screen, b2[0], b2[1], b4[0] - 30, cy3, GREEN)
        draw_flow_arrow(screen, b3[0], b3[1], b4[0] + 30, cy3, ORANGE)
        draw_flow_box(screen, 50, cy3, 170, box_h, "LiDAR / proximity", RED)
        draw_flow_arrow(screen, 220, cy3 + box_h // 2, cx - box_w // 2, cy3 + box_h // 2, RED)
        cy4 = cy3 + box_h + 34
        b5 = draw_flow_box(screen, cx - box_w // 2, cy4, box_w, box_h, "Planner + safety stop", CYAN)
        draw_flow_arrow(screen, b4[0], b4[1], b5[0], cy4, YELLOW)
        draw_flow_box(screen, WIDTH - 230, cy3, 180, box_h, "Semantic map + log", PURPLE)
        draw_flow_arrow(screen, cx + box_w // 2, cy3 + box_h // 2, WIDTH - 230, cy3 + box_h // 2, PURPLE)
        cy5 = cy4 + box_h + 34
        draw_flow_box(screen, cx - box_w // 2, cy5, box_w, box_h, "Wheel commands (L/R)", BLUE)
        draw_flow_arrow(screen, b5[0], b5[1], cx, cy5, CYAN)
        loop_txt = SMALL_FONT.render("repeats every frame (sim: 20 Hz planning, 60 Hz physics)", True, GRAY)
        screen.blit(loop_txt, (cx - loop_txt.get_width() // 2, cy5 + box_h + 10))

        table_y = cy5 + box_h + 44
        screen.blit(HEADING_FONT.render("Tech Stack", True, TEXT), (50, table_y))
        stack = [
            ("YOLO26", "real-time detection"),
            ("OpenCV", "camera + HUD"),
            ("Three.js", "3D simulator"),
            ("Pygame", "2D sim + dashboard"),
            ("NumPy", "control maths"),
            ("Arduino", "motor bridge (L298N)"),
        ]
        for i, (tech, purpose) in enumerate(stack):
            tx = 66 + (i % 3) * 310
            ty = table_y + 32 + (i // 3) * 26
            ts = MAIN_FONT.render(tech, True, CYAN)
            screen.blit(ts, (tx, ty))
            screen.blit(SMALL_FONT.render(f" - {purpose}", True, GRAY), (tx + ts.get_width(), ty + 2))

    # ======================= DEMO PLAN =======================
    else:
        screen.blit(HEADING_FONT.render("Live Demo Scenario (about 2 minutes)", True, TEXT), (50, content_y))
        steps = [
            ("1", "Natural-language goal", 'Type "find the red barrel" in the 3D simulator', BLUE),
            ("2", "Explore", "360 deg scan, then smart search: likely spots first (press D in the sim)", YELLOW),
            ("3", "Perceive", "Detector boxes in the robot camera; objects added to the robot's map", GREEN),
            ("4", "Avoid", "LiDAR rays turn red, planner steers around, safety layer prevents contact", RED),
            ("5", "Reach + remember", 'GOAL REACHED, then "go to the yellow pillar" -> uses its memory', ORANGE),
            ("6", "Real vision", "Vision Brain: YOLO26 on the webcam -> TURN LEFT / FORWARD / STOP", CYAN),
        ]
        step_y = content_y + 40
        for num, title_, desc, color in steps:
            pygame.draw.circle(screen, color, (80, step_y + 18), 16)
            n = MAIN_FONT.render(num, True, WHITE)
            screen.blit(n, (80 - n.get_width() // 2, step_y + 18 - n.get_height() // 2))
            screen.blit(HEADING_FONT.render(title_, True, color), (110, step_y + 2))
            screen.blit(SMALL_FONT.render(desc, True, GRAY), (110, step_y + 27))
            if num != "6":
                pygame.draw.line(screen, DARK_GRAY, (80, step_y + 34), (80, step_y + 54), 1)
            step_y += 56

        perf_y = step_y + 14
        screen.blit(HEADING_FONT.render("Measured Performance", True, TEXT), (50, perf_y))
        metrics = [
            (MEASURED["3d_success"], "3D success", f"{MEASURED['3d_trials']} trials", GREEN),
            (MEASURED["3d_collisions"], "3D collisions", f"{MEASURED['3d_trials']} trials", CYAN),
            (MEASURED["2d_success"], "2D success", f"{MEASURED['2d_trials']} trials", YELLOW),
            ("55.0", "YOLO26l mAP", "COCO, published", PURPLE),
        ]
        mw = 200
        msx = WIDTH // 2 - (len(metrics) * mw) // 2
        for i, (val, lbl, sub, clr) in enumerate(metrics):
            draw_metric_card(screen, msx + i * mw, perf_y + 32, mw - 16, 90, val, lbl, sub, clr)

    # ---------------- footer ----------------
    if time.time() < status_until:
        msg = MAIN_FONT.render(status_msg, True, CYAN)
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT - 32))
    footer_txt = TINY_FONT.render("1/2/3: tabs  |  Q: quit  |  click a button to launch a demo", True, DARK_GRAY)
    screen.blit(footer_txt, (WIDTH // 2 - footer_txt.get_width() // 2, HEIGHT - 15))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
