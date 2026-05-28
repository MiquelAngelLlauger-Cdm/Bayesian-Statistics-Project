"""
visualize.py
============
Pygame visualization of the constrained submarine search.

A chess-like (black/white) 3x3 board pops up. The searcher (a surface ship with
a sonar ping) moves along the optimal path every SEARCHER_INTERVAL seconds; one
second after each searcher move the submarine performs its Markov transition.
The episode ends when the searcher finds the submarine or the optimal path is
exhausted.

Run:
    python visualize.py
    python visualize.py --start 1 --horizon 10 --p-stay 0.4 --seed 7

Icons are drawn vectorially, so there are no external asset dependencies.
"""

import argparse
import sys

import numpy as np
import pygame

from simulation import SearchSimulation

# ---- timing (milliseconds) ----
SEARCHER_INTERVAL = 2000   # searcher moves every 2 s
SUB_DELAY = 1000           # submarine moves 1 s after the searcher
END_HOLD = 3500            # how long to hold the final frame before idle

# ---- layout ----
BOARD = 540                # board pixel size
MARGIN = 40
SIDEBAR = 360
WIN_W = MARGIN * 2 + BOARD + SIDEBAR
WIN_H = MARGIN * 2 + BOARD + 60
CELL = BOARD // 3

# ---- palette ----
LIGHT = (238, 238, 226)
DARK = (118, 150, 86)        # classic chessboard green-white
BG = (24, 28, 36)
PANEL = (34, 40, 52)
GRID_LINE = (60, 70, 86)
TEXT = (228, 232, 240)
SUBTEXT = (150, 160, 176)
SEARCHER_COL = (235, 110, 70)
SEARCHER_DARK = (170, 60, 30)
SUB_COL = (245, 210, 70)
SUB_DARK = (150, 120, 20)
PING = (120, 200, 255)
FOUND_COL = (90, 220, 130)
ESCAPE_COL = (235, 90, 90)
PATH_COL = (235, 110, 70)


def cell_center(cell):
    """1-indexed cell -> pixel center on the board."""
    row, col = (cell - 1) // 3, (cell - 1) % 3
    x = MARGIN + col * CELL + CELL // 2
    y = MARGIN + row * CELL + CELL // 2
    return x, y


def draw_submarine(surf, cx, cy, scale=1.0, alpha=255):
    """Draw a little submarine centered at (cx, cy)."""
    s = scale
    body_w, body_h = int(74 * s), int(34 * s)
    layer = pygame.Surface((body_w + 60, body_h + 60), pygame.SRCALPHA)
    ox, oy = (body_w + 60) // 2, (body_h + 60) // 2
    # hull
    pygame.draw.ellipse(layer, (*SUB_COL, alpha),
                        (ox - body_w // 2, oy - body_h // 2, body_w, body_h))
    pygame.draw.ellipse(layer, (*SUB_DARK, alpha),
                        (ox - body_w // 2, oy - body_h // 2, body_w, body_h), 3)
    # conning tower
    tw, th = int(20 * s), int(20 * s)
    pygame.draw.rect(layer, (*SUB_COL, alpha),
                     (ox - tw // 2, oy - body_h // 2 - th + 4, tw, th),
                     border_radius=int(5 * s))
    pygame.draw.rect(layer, (*SUB_DARK, alpha),
                     (ox - tw // 2, oy - body_h // 2 - th + 4, tw, th),
                     2, border_radius=int(5 * s))
    # periscope
    pygame.draw.line(layer, (*SUB_DARK, alpha),
                     (ox, oy - body_h // 2 - th + 4),
                     (ox, oy - body_h // 2 - th - int(10 * s)), max(2, int(3 * s)))
    # portholes
    for k in (-1, 0, 1):
        pygame.draw.circle(layer, (*SUB_DARK, alpha),
                           (ox + k * int(16 * s), oy), max(2, int(4 * s)))
    # propeller
    pygame.draw.line(layer, (*SUB_DARK, alpha),
                     (ox + body_w // 2 - 2, oy),
                     (ox + body_w // 2 + int(10 * s), oy), max(2, int(3 * s)))
    surf.blit(layer, (cx - ox, cy - oy))


def draw_searcher(surf, cx, cy, scale=1.0):
    """Draw the search vessel (a ship with a sonar dome) centered at (cx, cy)."""
    s = scale
    w, h = int(80 * s), int(80 * s)
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    ox, oy = w // 2, h // 2
    # hull (trapezoid)
    hull = [
        (ox - int(30 * s), oy + int(2 * s)),
        (ox + int(30 * s), oy + int(2 * s)),
        (ox + int(20 * s), oy + int(18 * s)),
        (ox - int(20 * s), oy + int(18 * s)),
    ]
    pygame.draw.polygon(layer, SEARCHER_COL, hull)
    pygame.draw.polygon(layer, SEARCHER_DARK, hull, 2)
    # superstructure
    pygame.draw.rect(layer, SEARCHER_COL,
                     (ox - int(10 * s), oy - int(14 * s), int(20 * s), int(16 * s)),
                     border_radius=int(3 * s))
    pygame.draw.rect(layer, SEARCHER_DARK,
                     (ox - int(10 * s), oy - int(14 * s), int(20 * s), int(16 * s)),
                     2, border_radius=int(3 * s))
    # mast
    pygame.draw.line(layer, SEARCHER_DARK,
                     (ox, oy - int(14 * s)), (ox, oy - int(26 * s)), max(2, int(3 * s)))
    pygame.draw.circle(layer, PING, (ox, oy - int(26 * s)), max(2, int(4 * s)))
    surf.blit(layer, (cx - ox, cy - oy))


class Animator:
    """Smoothly interpolates an entity between two cells over a duration."""

    def __init__(self, cell):
        self.from_cell = cell
        self.to_cell = cell
        self.start_ms = 0
        self.dur = 1
        self.cur = cell_center(cell) if cell else (0, 0)

    def move_to(self, cell, now_ms, dur):
        self.from_cell = self.to_cell
        self.to_cell = cell
        self.start_ms = now_ms
        self.dur = max(dur, 1)

    def place(self, cell):
        self.from_cell = self.to_cell = cell
        self.cur = cell_center(cell)

    def position(self, now_ms):
        if self.to_cell is None:
            return None
        if self.from_cell is None:
            self.cur = cell_center(self.to_cell)
            return self.cur
        t = min(1.0, (now_ms - self.start_ms) / self.dur)
        # ease in-out
        t = t * t * (3 - 2 * t)
        x0, y0 = cell_center(self.from_cell)
        x1, y1 = cell_center(self.to_cell)
        self.cur = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        return self.cur


def run(start_cell=None, horizon=10, p_stay=0.4, seed=None,
        sub_start=None, first_search_is_start=False):
    rng = np.random.default_rng(seed)
    sim = SearchSimulation(p_stay=p_stay, start_cell=start_cell,
                           horizon=horizon, sub_start=sub_start, rng=rng,
                           first_search_is_start=first_search_is_start)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Constrained Submarine Search (Eagle 1984)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big = pygame.font.SysFont("consolas,menlo,monospace", 30, bold=True)
    small = pygame.font.SysFont("consolas,menlo,monospace", 14)

    searcher_anim = Animator(None)
    searcher_anim.to_cell = None
    sub_anim = Animator(sim.sub_cell)
    sub_anim.place(sim.sub_cell)

    # ---- timeline state machine ----
    # phase: 'searcher' (do a searcher step) then 'submarine' (after SUB_DELAY)
    now = pygame.time.get_ticks()
    next_searcher_ms = now + 600          # small initial pause
    pending_sub_ms = None
    ping_ms = None
    finished_at = None
    visited = []                          # cells searched so far (for trail)

    running = True
    while running:
        now = pygame.time.get_ticks()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # -------- simulation timeline --------
        if sim.is_running():
            if next_searcher_ms is not None and now >= next_searcher_ms:
                still = sim.searcher_step()
                ping_ms = now
                if sim.searcher_cell is not None:
                    if searcher_anim.to_cell is None:
                        searcher_anim.place(sim.searcher_cell)
                    else:
                        searcher_anim.move_to(sim.searcher_cell, now, 600)
                    if sim.searcher_cell not in visited:
                        visited.append(sim.searcher_cell)
                next_searcher_ms = None
                if still:
                    pending_sub_ms = now + SUB_DELAY
                else:
                    finished_at = now  # found, or escaped on this step

            if pending_sub_ms is not None and now >= pending_sub_ms:
                prev = sim.sub_cell
                sim.submarine_step()
                if sim.sub_cell != prev:
                    sub_anim.move_to(sim.sub_cell, now, 600)
                pending_sub_ms = None
                if sim.is_running():
                    next_searcher_ms = now + (SEARCHER_INTERVAL - SUB_DELAY)
        else:
            if finished_at is None:
                finished_at = now

        # -------- draw --------
        screen.fill(BG)
        _draw_board(screen)
        _draw_path_trail(screen, sim, visited)

        sub_pos = sub_anim.position(now)
        if sub_pos:
            draw_submarine(screen, int(sub_pos[0]), int(sub_pos[1]), scale=1.0)

        # sonar ping ring on the searched cell
        if ping_ms is not None and sim.searcher_cell is not None:
            _draw_ping(screen, sim.searcher_cell, now - ping_ms)

        sp = searcher_anim.position(now)
        if sp:
            draw_searcher(screen, int(sp[0]), int(sp[1]), scale=1.0)

        _draw_sidebar(screen, sim, font, big, small)
        _draw_outcome_banner(screen, sim, big)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _draw_board(screen):
    for cell in range(1, 10):
        row, col = (cell - 1) // 3, (cell - 1) % 3
        x = MARGIN + col * CELL
        y = MARGIN + row * CELL
        color = LIGHT if (row + col) % 2 == 0 else DARK
        pygame.draw.rect(screen, color, (x, y, CELL, CELL))
    # grid + frame
    for k in range(4):
        off = MARGIN + k * CELL
        pygame.draw.line(screen, GRID_LINE, (MARGIN, off), (MARGIN + BOARD, off), 2)
        pygame.draw.line(screen, GRID_LINE, (off, MARGIN), (off, MARGIN + BOARD), 2)
    pygame.draw.rect(screen, (200, 205, 215), (MARGIN, MARGIN, BOARD, BOARD), 3)
    # cell numbers
    f = pygame.font.SysFont("consolas,monospace", 16, bold=True)
    for cell in range(1, 10):
        row, col = (cell - 1) // 3, (cell - 1) % 3
        x = MARGIN + col * CELL + 8
        y = MARGIN + row * CELL + 6
        ink = (60, 60, 60) if (row + col) % 2 == 0 else (235, 235, 235)
        screen.blit(f.render(str(cell), True, ink), (x, y))


def _draw_path_trail(screen, sim, visited):
    if len(visited) < 1:
        return
    pts = [cell_center(c) for c in visited]
    if len(pts) >= 2:
        pygame.draw.lines(screen, (*SEARCHER_DARK, 255), False, pts, 4)
    for i, p in enumerate(pts):
        pygame.draw.circle(screen, SEARCHER_DARK, (int(p[0]), int(p[1])), 6)


def _draw_ping(screen, cell, age_ms):
    if age_ms > 900:
        return
    cx, cy = cell_center(cell)
    t = age_ms / 900.0
    radius = int(20 + t * (CELL * 0.45))
    alpha = max(0, int(180 * (1 - t)))
    ring = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(ring, (*PING, alpha), (radius + 2, radius + 2), radius, 4)
    screen.blit(ring, (cx - radius - 2, cy - radius - 2))



def _draw_sidebar(screen, sim, font, big, small):
    x = MARGIN + BOARD + 30
    y = MARGIN
    w = SIDEBAR - 50
    pygame.draw.rect(screen, PANEL, (x - 12, y - 8, w + 24, BOARD + 16),
                     border_radius=10)

    def line(txt, color=TEXT, fnt=font, dy=26):
        nonlocal y
        screen.blit(fnt.render(txt, True, color), (x, y))
        y += dy

    line("SUBMARINE SEARCH", PING, big, 40)
    line("Eagle (1984) POMDP", SUBTEXT, small, 26)
    y += 6
    line(f"Start cell : {sim.start_cell}", TEXT)
    line(f"Horizon T  : {sim.horizon}", TEXT)
    line(f"p(stay)    : {sim.p_stay:.2f}", TEXT)
    line(f"Optimal Pd : {sim.opt_pd:.4f}", FOUND_COL)
    y += 4
    path_str = "-".join(map(str, sim.optimal_path))
    line("Optimal path:", SUBTEXT, small, 20)
    # wrap path string
    for chunk in _wrap(path_str, 26):
        line(chunk, TEXT, small, 18)
    y += 6

    done = max(0, sim.path_index + (0 if sim.is_running() else 1))
    done = min(sim.path_index + 1, sim.horizon) if sim.searcher_cell else 0
    line(f"Step       : {min(max(sim.path_index+1,0),sim.horizon)}/{sim.horizon}",
         TEXT)
    cur = sim.searcher_cell if sim.searcher_cell else "-"
    line(f"Searcher at: {cur}", SEARCHER_COL)
    sub_txt = sim.sub_cell if not sim.is_running() else "hidden"
    line(f"Submarine  : {sub_txt}", SUB_COL)
    y += 10

    # event log (last few lines)
    line("Log:", SUBTEXT, small, 20)
    for entry in sim.log[-8:]:
        for chunk in _wrap(entry, 30):
            line(chunk, SUBTEXT, small, 16)


def _draw_outcome_banner(screen, sim, big):
    if sim.is_running():
        return
    if sim.status == sim.FOUND:
        msg, col = "TARGET FOUND", FOUND_COL
    else:
        msg, col = "SUBMARINE ESCAPED", ESCAPE_COL
    surf = big.render(msg, True, col)
    bx = MARGIN + BOARD // 2 - surf.get_width() // 2
    by = MARGIN + BOARD + 14
    pad = 12
    pygame.draw.rect(screen, PANEL,
                     (bx - pad, by - 6, surf.get_width() + 2 * pad,
                      surf.get_height() + 12), border_radius=8)
    screen.blit(surf, (bx, by))


def _wrap(text, width):
    out = []
    while len(text) > width:
        out.append(text[:width])
        text = text[width:]
    out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser(description="Submarine search visualization")
    ap.add_argument("--start", type=int, default=None,
                    help="searcher start cell 1-9 (default: best)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--p-stay", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sub-start", type=int, default=None,
                    help="force submarine start cell 1-9")
    ap.add_argument("--first-search-is-start", action="store_true",
                    help="first searched cell IS the start cell "
                         "(default: a neighbour, per Eagle)")
    args = ap.parse_args()
    run(start_cell=args.start, horizon=args.horizon, p_stay=args.p_stay,
        seed=args.seed, sub_start=args.sub_start,
        first_search_is_start=args.first_search_is_start)


if __name__ == "__main__":
    main()
