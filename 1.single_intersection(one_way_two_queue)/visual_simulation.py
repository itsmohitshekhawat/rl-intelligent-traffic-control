"""
Reinforcement Learning-Based Intelligent Traffic Control System
Final Year Project Demonstration — Chandigarh University
=============================================================
Run: python traffic_simulation.py
Requirements: pip install pygame
"""

import pygame
import random
import math
import time

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
W, H = 1280, 760
DASH_X = 820          # Dashboard starts here
SIM_W   = DASH_X      # Simulation area width

# Intersection centre
CX, CY = SIM_W // 2, H // 2
ROAD_W  = 80          # half-road width (each direction)

# Colors — dark-theme, neon-accent palette
BG          = (12,  15,  23)
ROAD_COL    = (30,  34,  46)
LANE_MARK   = (60,  65,  80)
GRASS       = (18,  28,  22)

RED_COL     = (255,  60,  60)
GREEN_COL   = ( 50, 220, 100)
YELLOW_COL  = (255, 200,  40)
AMBER       = (255, 150,  30)

DASH_BG     = (16,  20,  32)
PANEL_BG    = (22,  27,  42)
ACCENT      = ( 70, 140, 255)
ACCENT2     = (120, 220, 180)
TEXT_MAIN   = (220, 225, 240)
TEXT_DIM    = (100, 110, 140)
TEXT_BRIGHT = (255, 255, 255)
BORDER      = ( 40,  48,  70)

CAR_COLORS  = [
    (255,  90,  90), (90, 180, 255), (255, 210,  60),
    (120, 220, 120), (200, 120, 255), (255, 160,  80),
    (80,  230, 200), (255, 255, 120),
]

FPS = 60

# ─── PYGAME INIT ──────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("RL-Based Intelligent Traffic Control System")
clock  = pygame.time.Clock()

# Fonts
try:
    FONT_XL  = pygame.font.SysFont("Consolas", 26, bold=True)
    FONT_LG  = pygame.font.SysFont("Consolas", 20, bold=True)
    FONT_MD  = pygame.font.SysFont("Consolas", 15)
    FONT_SM  = pygame.font.SysFont("Consolas", 13)
    FONT_HED = pygame.font.SysFont("Consolas", 11)
except:
    FONT_XL  = pygame.font.SysFont(None, 28, bold=True)
    FONT_LG  = pygame.font.SysFont(None, 22, bold=True)
    FONT_MD  = pygame.font.SysFont(None, 16)
    FONT_SM  = pygame.font.SysFont(None, 14)
    FONT_HED = pygame.font.SysFont(None, 12)

# ─── CAR CLASS ────────────────────────────────────────────────────────────────
CAR_L, CAR_H_H = 28, 13   # car length, half-height

class Car:
    _id = 0

    def __init__(self, road, lane):
        Car._id += 1
        self.id   = Car._id
        self.road = road      # 1 = horizontal, 2 = vertical
        self.lane = lane      # +1 or -1 (direction)
        self.color  = random.choice(CAR_COLORS)
        self.waiting = False
        self.wait_time = 0.0
        self.speed_base = random.uniform(1.8, 2.8)
        self.speed = self.speed_base

        # Spawn off-screen
        if road == 1:
            self.y = CY + lane * (ROAD_W // 2)
            self.x = -60 if lane == 1 else SIM_W + 60
        else:
            self.x = CX + lane * (ROAD_W // 2)
            self.y = -60 if lane == 1 else H + 60

    def stop_line(self):
        """Position just before intersection."""
        if self.road == 1:
            return (CX - ROAD_W - CAR_L // 2) if self.lane == 1 else (CX + ROAD_W + CAR_L // 2)
        else:
            return (CY - ROAD_W - CAR_L // 2) if self.lane == 1 else (CY + ROAD_W + CAR_L // 2)

    def past_intersection(self):
        if self.road == 1:
            return self.x > CX + ROAD_W + 10 if self.lane == 1 else self.x < CX - ROAD_W - 10
        else:
            return self.y > CY + ROAD_W + 10 if self.lane == 1 else self.y < CY - ROAD_W - 10

    def at_stop(self):
        sl = self.stop_line()
        if self.road == 1:
            if self.lane == 1:  return self.x >= sl - 4 and not self.past_intersection()
            else:               return self.x <= sl + 4 and not self.past_intersection()
        else:
            if self.lane == 1:  return self.y >= sl - 4 and not self.past_intersection()
            else:               return self.y <= sl + 4 and not self.past_intersection()

    def update(self, dt, green_road, cars_ahead):
        self.waiting = False
        can_go = (self.road == green_road) or self.past_intersection()

        sl = self.stop_line()

        # Check car ahead spacing
        blocked_by_car = False
        for other in cars_ahead:
            if other is self: continue
            gap = 50
            if self.road == 1:
                if self.lane == 1 and 0 < (other.x - self.x) < gap:
                    blocked_by_car = True; break
                if self.lane == -1 and 0 < (self.x - other.x) < gap:
                    blocked_by_car = True; break
            else:
                if self.lane == 1 and 0 < (other.y - self.y) < gap:
                    blocked_by_car = True; break
                if self.lane == -1 and 0 < (self.y - other.y) < gap:
                    blocked_by_car = True; break

        if blocked_by_car:
            self.speed = 0
            self.waiting = True
        elif not can_go and self.at_stop():
            self.speed = 0
            self.waiting = True
        else:
            self.speed = self.speed_base

        if self.waiting:
            self.wait_time += dt
        
        # Move
        if self.road == 1:
            self.x += self.speed * self.lane
        else:
            self.y += self.speed * self.lane

    def off_screen(self):
        return self.x < -100 or self.x > SIM_W + 100 or self.y < -100 or self.y > H + 100

    def draw(self, surface):
        if self.road == 1:
            rect = pygame.Rect(self.x - CAR_L//2, self.y - CAR_H_H, CAR_L, CAR_H_H*2)
        else:
            rect = pygame.Rect(self.x - CAR_H_H, self.y - CAR_L//2, CAR_H_H*2, CAR_L)

        # Body
        pygame.draw.rect(surface, self.color, rect, border_radius=4)
        # Windshield shimmer
        if self.road == 1:
            ws = pygame.Rect(self.x - CAR_L//2 + 4, self.y - CAR_H_H + 3, 10, CAR_H_H*2 - 6)
            pygame.draw.rect(surface, (200, 230, 255, 120), ws, border_radius=2)
        else:
            ws = pygame.Rect(self.x - CAR_H_H + 3, self.y - CAR_L//2 + 4, CAR_H_H*2 - 6, 10)
            pygame.draw.rect(surface, (200, 230, 255, 120), ws, border_radius=2)
        # Outline
        pygame.draw.rect(surface, (0,0,0,80), rect, 1, border_radius=4)

        # Brake light glow if waiting
        if self.waiting:
            glow = pygame.Surface((CAR_L + 10, CAR_H_H*2 + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow, (255, 50, 50, 60), glow.get_rect(), border_radius=6)
            surface.blit(glow, (rect.x - 5, rect.y - 5))


# ─── TRAFFIC LIGHT ────────────────────────────────────────────────────────────
class TrafficLight:
    def __init__(self):
        self.green_road = 1      # which road has green
        self.timer = 0.0
        self.phase_duration = 6.0   # seconds per phase
        self.transition = False
        self.trans_timer = 0.0
        self.trans_duration = 0.8

    def update(self, dt, road1_count, road2_count):
        self.timer += dt
        if self.transition:
            self.trans_timer += dt
            if self.trans_timer >= self.trans_duration:
                self.transition = False
                self.trans_timer = 0.0
            return

        if self.timer >= self.phase_duration:
            self.timer = 0.0
            # RL Decision: give green to road with more cars
            if road1_count > road2_count:
                new_green = 1
            elif road2_count > road1_count:
                new_green = 2
            else:
                new_green = 2 if self.green_road == 1 else 1  # alternate if equal

            if new_green != self.green_road:
                self.green_road = new_green
                self.transition = True

    def get_color(self, road):
        if self.transition:
            return YELLOW_COL
        return GREEN_COL if self.green_road == road else RED_COL

    def is_green(self, road):
        if self.transition: return False
        return self.green_road == road


# ─── DRAW UTILITIES ───────────────────────────────────────────────────────────
def draw_text(surface, text, font, color, x, y, align="left"):
    surf = font.render(text, True, color)
    if align == "center":
        x -= surf.get_width() // 2
    elif align == "right":
        x -= surf.get_width()
    surface.blit(surf, (x, y))
    return surf.get_height()


def draw_rounded_box(surface, rect, color, radius=8, border=None, border_color=BORDER):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border_color, rect, border, border_radius=radius)


def draw_traffic_light_fixture(surface, x, y, road, tl):
    """Draw a realistic traffic light pole + housing."""
    # Pole
    pygame.draw.rect(surface, (50, 55, 70), (x - 3, y, 6, 40))
    # Housing
    housing = pygame.Rect(x - 14, y - 50, 28, 55)
    pygame.draw.rect(surface, (35, 38, 52), housing, border_radius=5)
    pygame.draw.rect(surface, BORDER, housing, 1, border_radius=5)

    col = tl.get_color(road)
    # dim bulbs
    pygame.draw.circle(surface, (60, 20, 20), (x, y - 36), 9)
    pygame.draw.circle(surface, (20, 60, 20), (x, y - 18), 9)
    # active bulb
    if tl.get_color(road) == RED_COL:
        pygame.draw.circle(surface, RED_COL, (x, y - 36), 9)
        # glow
        glow = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 50, 50, 50), (20, 20), 18)
        surface.blit(glow, (x - 20, y - 56))
    elif tl.get_color(road) == GREEN_COL:
        pygame.draw.circle(surface, GREEN_COL, (x, y - 18), 9)
        glow = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow, (50, 220, 100, 50), (20, 20), 18)
        surface.blit(glow, (x - 20, y - 38))
    else:  # yellow
        pygame.draw.circle(surface, YELLOW_COL, (x, y - 27), 9)


def draw_intersection(surface):
    # Grass / background
    surface.fill(GRASS, (0, 0, SIM_W, H))

    # Road stripes - horizontal
    pygame.draw.rect(surface, ROAD_COL, (0, CY - ROAD_W, SIM_W, ROAD_W * 2))
    # Road stripes - vertical
    pygame.draw.rect(surface, ROAD_COL, (CX - ROAD_W, 0, ROAD_W * 2, H))

    # Intersection box
    pygame.draw.rect(surface, (35, 38, 50), (CX - ROAD_W, CY - ROAD_W, ROAD_W*2, ROAD_W*2))

    # Lane markings - horizontal dashes
    for xi in range(0, SIM_W, 24):
        if abs(xi - CX) > ROAD_W + 4:
            pygame.draw.rect(surface, LANE_MARK, (xi, CY - 2, 14, 4))

    # Lane markings - vertical dashes
    for yi in range(0, H, 24):
        if abs(yi - CY) > ROAD_W + 4:
            pygame.draw.rect(surface, LANE_MARK, (CX - 2, yi, 4, 14))

    # Stop lines
    pygame.draw.rect(surface, (200, 200, 200), (CX - ROAD_W - 3, CY - ROAD_W, 3, ROAD_W), border_radius=1)
    pygame.draw.rect(surface, (200, 200, 200), (CX + ROAD_W, CY - ROAD_W, 3, ROAD_W), border_radius=1)
    pygame.draw.rect(surface, (200, 200, 200), (CX - ROAD_W, CY - ROAD_W - 3, ROAD_W, 3), border_radius=1)
    pygame.draw.rect(surface, (200, 200, 200), (CX - ROAD_W, CY + ROAD_W, ROAD_W, 3), border_radius=1)

    # Road edge lines
    pygame.draw.rect(surface, (80, 85, 100), (0, CY - ROAD_W, SIM_W, 2))
    pygame.draw.rect(surface, (80, 85, 100), (0, CY + ROAD_W - 2, SIM_W, 2))
    pygame.draw.rect(surface, (80, 85, 100), (CX - ROAD_W, 0, 2, H))
    pygame.draw.rect(surface, (80, 85, 100), (CX + ROAD_W - 2, 0, 2, H))

    # Road labels
    draw_text(surface, "ROAD 1  ───────►", FONT_SM, TEXT_DIM, 20, CY - ROAD_W - 18)
    draw_text(surface, "◄───────  ROAD 1", FONT_SM, TEXT_DIM, SIM_W - 160, CY + ROAD_W + 4)
    # Vertical labels (rotated manually via surface)
    lbl2a = FONT_SM.render("ROAD 2  ▼", True, TEXT_DIM)
    lbl2a_r = pygame.transform.rotate(lbl2a, 90)
    surface.blit(lbl2a_r, (CX + ROAD_W + 4, 20))
    lbl2b = FONT_SM.render("▲  ROAD 2", True, TEXT_DIM)
    lbl2b_r = pygame.transform.rotate(lbl2b, 90)
    surface.blit(lbl2b_r, (CX - ROAD_W - 18, H - 120))


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
def draw_bar(surface, x, y, w, h, value, max_val, color, bg=(30, 35, 52)):
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
    fill_w = int(w * min(value / max(max_val, 1), 1.0))
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=4)
    pygame.draw.rect(surface, BORDER, (x, y, w, h), 1, border_radius=4)


def draw_pill(surface, x, y, label, color, text_color=BG):
    surf = FONT_SM.render(label, True, text_color)
    pw = surf.get_width() + 16
    ph = surf.get_height() + 8
    pygame.draw.rect(surface, color, (x, y, pw, ph), border_radius=10)
    surface.blit(surf, (x + 8, y + 4))
    return pw, ph


def draw_dashboard(surface, tl, cars, sim_time, total_wait, speed_mult, paused,
                   road1_count, road2_count, status_msg):
    # Background
    pygame.draw.rect(surface, DASH_BG, (DASH_X, 0, W - DASH_X, H))
    pygame.draw.rect(surface, BORDER, (DASH_X, 0, 1, H))

    DX = DASH_X + 16
    DW = W - DASH_X - 32
    y  = 12

    # ── TITLE ──
    draw_rounded_box(surface, (DASH_X, y - 4, W - DASH_X, 52), (20, 25, 40), radius=0)
    draw_text(surface, "INTELLIGENT TRAFFIC AI", FONT_LG, ACCENT, DX, y)
    y += 22
    draw_text(surface, "RL-Based Control System", FONT_SM, TEXT_DIM, DX, y)
    y += 22

    # Divider
    pygame.draw.rect(surface, BORDER, (DASH_X + 8, y, DW + 8, 1))
    y += 10

    # ── STATUS BADGE ──
    if paused:
        draw_pill(surface, DX, y, "  ⏸  PAUSED  ", (80, 80, 100), TEXT_BRIGHT)
    else:
        draw_pill(surface, DX, y, "  ●  RUNNING  ", (30, 150, 80), TEXT_BRIGHT)
    y += 30

    # ─ SECTION: WHAT IS HAPPENING ─
    draw_text(surface, "WHAT IS HAPPENING", FONT_HED, ACCENT, DX, y)
    y += 14
    draw_rounded_box(surface, (DX, y, DW, 52), PANEL_BG, radius=6, border=1)
    # Word-wrap status message
    words = status_msg.split()
    line, lines_out = "", []
    for w2 in words:
        test = (line + " " + w2).strip()
        if FONT_SM.size(test)[0] < DW - 14:
            line = test
        else:
            lines_out.append(line); line = w2
    lines_out.append(line)
    ty = y + 6
    for ln in lines_out[:3]:
        draw_text(surface, ln, FONT_SM, TEXT_MAIN, DX + 8, ty)
        ty += 15
    y += 58

    # ─ SECTION: CURRENT TRAFFIC ─
    draw_text(surface, "CURRENT TRAFFIC", FONT_HED, ACCENT, DX, y)
    y += 14

    for road_n, count in [(1, road1_count), (2, road2_count)]:
        is_green = tl.is_green(road_n)
        box_col  = (25, 45, 30) if is_green else (45, 22, 22)
        draw_rounded_box(surface, (DX, y, DW, 44), box_col, radius=6, border=1,
                         border_color=GREEN_COL if is_green else RED_COL)
        # signal dot
        sig_col = GREEN_COL if is_green else RED_COL
        pygame.draw.circle(surface, sig_col, (DX + 14, y + 22), 7)
        # glow
        glow_s = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*sig_col, 50), (15, 15), 13)
        surface.blit(glow_s, (DX + 1, y + 9))

        draw_text(surface, f"Road {road_n}", FONT_MD, TEXT_BRIGHT, DX + 28, y + 4)
        sig_label = "GREEN — GO" if is_green else "RED — STOP"
        draw_text(surface, sig_label, FONT_SM, sig_col, DX + 28, y + 22)
        # Car count badge
        cnt_str = str(count)
        draw_text(surface, cnt_str, FONT_XL, TEXT_BRIGHT, DX + DW - 40, y + 8, align="right")
        draw_text(surface, "cars", FONT_HED, TEXT_DIM, DX + DW - 36, y + 30)

        # Bar chart
        bar_y = y + 36
        draw_bar(surface, DX + 28, bar_y, DW - 36, 5, count, 15,
                 GREEN_COL if is_green else RED_COL)
        y += 52

    # ─ SECTION: AI DECISION ─
    draw_text(surface, "AI DECISION", FONT_HED, ACCENT, DX, y)
    y += 14
    draw_rounded_box(surface, (DX, y, DW, 68), PANEL_BG, radius=6, border=1)

    if road1_count > road2_count:
        reason = f"Road 1 has {road1_count} cars vs Road 2's {road2_count}."
        decision = "GREEN → Road 1"
        d_col = GREEN_COL
    elif road2_count > road1_count:
        reason = f"Road 2 has {road2_count} cars vs Road 1's {road1_count}."
        decision = "GREEN → Road 2"
        d_col = GREEN_COL
    else:
        reason = f"Both roads equal ({road1_count} cars each)."
        decision = "Alternating signals"
        d_col = YELLOW_COL

    draw_text(surface, reason, FONT_SM, TEXT_DIM, DX + 8, y + 8)
    draw_text(surface, "Decision:", FONT_SM, TEXT_DIM, DX + 8, y + 26)
    draw_text(surface, decision, FONT_MD, d_col, DX + 72, y + 24)
    draw_text(surface, "AI gives priority to the busier road.", FONT_SM, TEXT_DIM, DX + 8, y + 46)
    y += 76

    # ─ SECTION: SYSTEM GOAL ─
    draw_text(surface, "SYSTEM GOAL", FONT_HED, ACCENT, DX, y)
    y += 14
    draw_rounded_box(surface, (DX, y, DW, 36), (20, 28, 45), radius=6, border=1, border_color=(50, 80, 130))
    draw_text(surface, "Minimize waiting • Maximize flow", FONT_SM, ACCENT2, DX + 8, y + 10)
    y += 44

    # ─ SECTION: PERFORMANCE METRICS ─
    draw_text(surface, "PERFORMANCE METRICS", FONT_HED, ACCENT, DX, y)
    y += 14
    draw_rounded_box(surface, (DX, y, DW, 108), PANEL_BG, radius=6, border=1)

    metrics = [
        ("Sim Time",    f"{sim_time:.0f}s",        ACCENT),
        ("Total Wait",  f"{total_wait:.0f}s",       AMBER),
        ("Active Cars", f"{len(cars)}",             ACCENT2),
    ]
    my = y + 8
    for label, val, col in metrics:
        draw_text(surface, label, FONT_SM, TEXT_DIM, DX + 8, my)
        draw_text(surface, val,   FONT_LG, col,       DX + DW - 8, my, align="right")
        my += 28

    # Traffic load
    total_cars = road1_count + road2_count
    if total_cars <= 4:   load_label, load_col = "LOW",    GREEN_COL
    elif total_cars <= 9: load_label, load_col = "MEDIUM", YELLOW_COL
    else:                 load_label, load_col = "HIGH",   RED_COL

    draw_text(surface, "Traffic Load", FONT_SM, TEXT_DIM, DX + 8, my)
    draw_pill(surface, DX + DW - 70, my - 3, f"  {load_label}  ", load_col,
              BG if load_label != "MEDIUM" else BG)
    y += 116

    # ─ SECTION: HOW IT WORKS ─
    draw_text(surface, "HOW IT WORKS", FONT_HED, ACCENT, DX, y)
    y += 14
    steps = [
        "1. Count cars on each road",
        "2. Compare traffic density",
        "3. AI picks the busier road",
        "4. Green signal → cars move",
        "5. Repeat every few seconds",
    ]
    draw_rounded_box(surface, (DX, y, DW, len(steps)*18 + 12), PANEL_BG, radius=6, border=1)
    sy = y + 6
    for step in steps:
        draw_text(surface, step, FONT_SM, TEXT_MAIN, DX + 8, sy)
        sy += 18
    y += len(steps)*18 + 20

    # ─ CONTROLS ─
    pygame.draw.rect(surface, BORDER, (DASH_X + 8, y, DW + 8, 1))
    y += 8
    draw_text(surface, "CONTROLS", FONT_HED, TEXT_DIM, DX, y)
    y += 14
    draw_text(surface, "SPACE = Pause/Resume", FONT_SM, TEXT_DIM, DX, y); y += 16
    draw_text(surface, "↑/↓  =  Speed control", FONT_SM, TEXT_DIM, DX, y); y += 16
    draw_text(surface, f"Speed: {'▶▶' if speed_mult > 1 else ('▶' if speed_mult == 1 else '▶▷')}", FONT_SM, ACCENT2, DX, y)


# ─── MESSAGE GENERATOR ────────────────────────────────────────────────────────
def make_status(tl, road1_count, road2_count):
    if tl.transition:
        return "Switching signal to reduce road congestion..."
    if road1_count > road2_count:
        return f"More vehicles on Road 1 ({road1_count}). System gives green to Road 1."
    elif road2_count > road1_count:
        return f"More vehicles on Road 2 ({road2_count}). System gives green to Road 2."
    elif road1_count == 0 and road2_count == 0:
        return "Intersection clear. System is monitoring for incoming traffic."
    else:
        return f"Equal traffic on both roads ({road1_count} each). Alternating signals."


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    tl = TrafficLight()
    cars = []
    spawn_timer = 0.0
    spawn_interval = 1.8   # seconds between spawns

    sim_time   = 0.0
    total_wait = 0.0
    paused     = False
    speed_mult = 1

    # Particle / effect list: [(x,y, dx,dy, life, color)]
    particles = []

    running = True
    while running:
        dt_raw = clock.tick(FPS) / 1000.0
        dt = min(dt_raw, 0.05) * speed_mult

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    speed_mult = min(speed_mult + 1, 3)
                if event.key == pygame.K_DOWN:
                    speed_mult = max(speed_mult - 1, 1)
                if event.key == pygame.K_ESCAPE:
                    running = False

        if paused:
            # Draw frozen frame with PAUSED overlay
            pygame.display.flip()
            # Redraw dashboard with paused=True
            draw_intersection(screen)
            for car in cars:
                car.draw(screen)
            # Traffic lights
            draw_traffic_light_fixture(screen, CX - ROAD_W - 10, CY + 8, 1, tl)
            draw_traffic_light_fixture(screen, CX + ROAD_W - 4,  CY - ROAD_W - 50, 2, tl)

            road1_count = sum(1 for c in cars if c.road == 1 and not c.past_intersection())
            road2_count = sum(1 for c in cars if c.road == 2 and not c.past_intersection())
            draw_dashboard(screen, tl, cars, sim_time, total_wait, speed_mult, paused,
                           road1_count, road2_count, make_status(tl, road1_count, road2_count))
            pygame.display.flip()
            continue

        sim_time   += dt

        # ── Spawn cars ──
        spawn_timer += dt
        if spawn_timer >= spawn_interval / speed_mult:
            spawn_timer = 0.0
            # Random road + lane
            road = random.choice([1, 1, 2])
            lane = random.choice([1, -1])
            cars.append(Car(road, lane))

        # ── Update cars ──
        # Sort for ahead-checking
        h_cars = sorted([c for c in cars if c.road == 1 and c.lane == 1],  key=lambda c: c.x)
        h_cars_r = sorted([c for c in cars if c.road == 1 and c.lane == -1], key=lambda c: -c.x)
        v_cars = sorted([c for c in cars if c.road == 2 and c.lane == 1],  key=lambda c: c.y)
        v_cars_r = sorted([c for c in cars if c.road == 2 and c.lane == -1], key=lambda c: -c.y)

        def get_ahead(car):
            if car.road == 1:
                group = h_cars if car.lane == 1 else h_cars_r
            else:
                group = v_cars if car.lane == 1 else v_cars_r
            return group

        for car in cars:
            car.update(dt, tl.green_road, get_ahead(car))
            if car.waiting:
                total_wait += dt

        # Remove off-screen
        cars = [c for c in cars if not c.off_screen()]

        # Count waiting cars per road
        road1_count = sum(1 for c in cars if c.road == 1 and not c.past_intersection())
        road2_count = sum(1 for c in cars if c.road == 2 and not c.past_intersection())

        # ── RL update ──
        tl.update(dt, road1_count, road2_count)

        # ── Draw ──
        draw_intersection(screen)

        for car in cars:
            car.draw(screen)

        # Traffic lights fixtures
        draw_traffic_light_fixture(screen, CX - ROAD_W - 10, CY + 8,         1, tl)
        draw_traffic_light_fixture(screen, CX + ROAD_W - 4,  CY - ROAD_W - 50, 2, tl)

        # Signal phase countdown bar
        prog = 1.0 - (tl.timer / tl.phase_duration)
        bar_x = CX - ROAD_W
        bar_y = CY + ROAD_W + 6
        bar_w = ROAD_W * 2
        pygame.draw.rect(screen, (40, 44, 60), (bar_x, bar_y, bar_w, 5), border_radius=2)
        fill = int(bar_w * prog)
        if fill > 0:
            col = tl.get_color(tl.green_road)
            pygame.draw.rect(screen, col, (bar_x, bar_y, fill, 5), border_radius=2)

        # Dashboard
        status_msg = make_status(tl, road1_count, road2_count)
        draw_dashboard(screen, tl, cars, sim_time, total_wait, speed_mult, paused,
                       road1_count, road2_count, status_msg)

        # Title watermark in sim area
        draw_text(screen, "RL TRAFFIC CONTROL", FONT_HED, (40, 48, 65), 8, H - 18)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()