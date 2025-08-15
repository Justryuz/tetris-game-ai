from __future__ import annotations
import time
from typing import Optional
import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

CELL = 24
MARGIN = 2
BG = (20, 20, 24)
FG = (230, 230, 230)
BLOCK = (90, 180, 255)


def ascii_render(grid: np.ndarray) -> str:
    # grid: (H, W) or (1,H,W)
    if grid.ndim == 3:
        grid = grid[0]
    lines = []
    for r in grid:
        line = ''.join('#' if c > 0 else '.' for c in r)
        lines.append(line)
    return '\n'.join(lines)


class PygameViewer:
    def __init__(self, board_h: int = 20, board_w: int = 10, title: str = 'Tetris DQN'):
        if not _HAS_PYGAME:
            raise RuntimeError('pygame not available')
        pygame.init()
        self.h, self.w = board_h, board_w
        W = self.w * (CELL + MARGIN) + MARGIN
        H = self.h * (CELL + MARGIN) + MARGIN
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def draw(self, grid: np.ndarray, fps: int = 60):
        if grid.ndim == 3:
            grid = grid[0]
        import pygame
        self.screen.fill(BG)
        for y in range(self.h):
            for x in range(self.w):
                rect = (MARGIN + x*(CELL+MARGIN), MARGIN + y*(CELL+MARGIN), CELL, CELL)
                pygame.draw.rect(self.screen, FG, rect, width=1, border_radius=4)
                if grid[y, x] > 0:
                    pygame.draw.rect(self.screen, BLOCK, rect, width=0, border_radius=4)
        pygame.display.flip()
        self.clock.tick(fps)

    def pump(self):
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True