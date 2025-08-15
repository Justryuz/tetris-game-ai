from __future__ import annotations
import numpy as np
from typing import Tuple

BOARD_H = 20
BOARD_W = 10

TETROMINOES = {
    'I': np.array([[1, 1, 1, 1]], dtype=np.uint8),
    'O': np.array([[1, 1],
                   [1, 1]], dtype=np.uint8),
    'T': np.array([[1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8),
    'S': np.array([[0, 1, 1],
                   [1, 1, 0]], dtype=np.uint8),
    'Z': np.array([[1, 1, 0],
                   [0, 1, 1]], dtype=np.uint8),
    'J': np.array([[1, 0, 0],
                   [1, 1, 1]], dtype=np.uint8),
    'L': np.array([[0, 0, 1],
                   [1, 1, 1]], dtype=np.uint8),
}
PIECE_TYPES = list(TETROMINOES.keys())


def collide(board: np.ndarray, piece: np.ndarray, y: int, x: int) -> bool:
    H, W = board.shape
    ph, pw = piece.shape
    if x < 0 or x + pw > W or y < 0 or y + ph > H:
        return True
    region = board[y:y+ph, x:x+pw]
    return np.any(region & piece)


class TetrisEnv:
    """Minimal Tetris env for DQN.

    Actions:
        0: noop, 1: left, 2: right, 3: rotate CW, 4: soft drop, 5: hard drop

    Observation: (1, 20, 10) float32 grid — board + current piece overlayed.
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
        self.score = 0
        self.lines_cleared_total = 0
        self.piece_type = None
        self.piece = None
        self.px = 0
        self.py = 0
        self.game_over = False
        self.bag = []  # 7-bag randomizer
        self.spawn_new_piece()

    def reset(self):
        self.board[:] = 0
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self.bag.clear()
        self.spawn_new_piece()
        return self._get_obs()

    def sample_piece(self):
        if not self.bag:
            self.bag = PIECE_TYPES.copy()
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def spawn_new_piece(self):
        self.piece_type = self.sample_piece()
        self.piece = TETROMINOES[self.piece_type].copy()
        self.py = 0
        self.px = BOARD_W // 2 - self.piece.shape[1] // 2
        if collide(self.board, self.piece, self.py, self.px):
            self.game_over = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.game_over:
            return self._get_obs(), 0.0, True, {}

        reward = -0.01

        if action == 1:  # left
            if not collide(self.board, self.piece, self.py, self.px - 1):
                self.px -= 1
        elif action == 2:  # right
            if not collide(self.board, self.piece, self.py, self.px + 1):
                self.px += 1
        elif action == 3:  # rotate CW + simple wall kicks
            rotated = np.rot90(self.piece, -1)
            if not collide(self.board, rotated, self.py, self.px):
                self.piece = rotated
            else:
                for dx in (-1, 1, -2, 2):
                    if not collide(self.board, rotated, self.py, self.px + dx):
                        self.px += dx
                        self.piece = rotated
                        break
        elif action == 4:  # soft drop
            if not collide(self.board, self.piece, self.py + 1, self.px):
                self.py += 1
                reward += 0.01
        elif action == 5:  # hard drop
            while not collide(self.board, self.piece, self.py + 1, self.px):
                self.py += 1
                reward += 0.01
            locked_reward, _ = self._lock_and_clear()
            reward += locked_reward
            return self._post_lock_step(reward)

        # gravity
        if not collide(self.board, self.piece, self.py + 1, self.px):
            self.py += 1
        else:
            locked_reward, _ = self._lock_and_clear()
            reward += locked_reward
            return self._post_lock_step(reward)

        return self._get_obs(), float(reward), False, {}

    def _post_lock_step(self, reward: float):
        if self.game_over:
            return self._get_obs(), float(reward), True, {}
        self.spawn_new_piece()
        if self.game_over:
            reward -= 5.0
            return self._get_obs(), float(reward), True, {}
        return self._get_obs(), float(reward), False, {}

    def _lock_and_clear(self):
        ph, pw = self.piece.shape
        self.board[self.py:self.py+ph, self.px:self.px+pw] |= self.piece
        full_rows = np.where(self.board.sum(axis=1) == BOARD_W)[0]
        lines = int(len(full_rows))
        if lines > 0:
            self.board = np.delete(self.board, full_rows, axis=0)
            self.board = np.vstack([np.zeros((lines, BOARD_W), dtype=np.uint8), self.board])
            self.lines_cleared_total += lines
        reward = -0.05 + [0.0, 1.0, 3.0, 5.0, 8.0][lines]
        return reward, lines

    def _get_obs(self):
        grid = self.board.copy()
        ph, pw = self.piece.shape
        y0, x0 = max(self.py, 0), max(self.px, 0)
        y1, x1 = min(self.py+ph, BOARD_H), min(self.px+pw, BOARD_W)
        py0, px0 = y0 - self.py, x0 - self.px
        py1, px1 = py0 + (y1 - y0), px0 + (x1 - x0)
        if y0 < y1 and x0 < x1:
            grid[y0:y1, x0:x1] |= self.piece[py0:py1, px0:px1]
        return grid.astype(np.float32)[None, :, :]