from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from env import TetrisEnv
from visualize import ascii_render, PygameViewer


class Tetris:
    """Wrapper for TetrisEnv to match the expected interface."""
    
    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self.env = TetrisEnv(seed=seed)
        self.render_mode = render_mode
        self.action_space = 6  # 0: noop, 1: left, 2: right, 3: rotate, 4: soft drop, 5: hard drop
        self.viewer = None
        
        if render_mode == "pygame":
            try:
                self.viewer = PygameViewer()
            except RuntimeError:
                print("Warning: pygame not available, falling back to ascii")
                self.render_mode = "ascii"
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        state = self.env.reset()
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take a step in the environment."""
        state, reward, done, _ = self.env.step(action)
        return state, reward, done
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.env._get_obs()
    
    def get_frame(self) -> np.ndarray:
        """Get current frame for rendering."""
        return self.env._get_obs()
    
    def render(self):
        """Render the current state."""
        if self.render_mode == "ascii":
            frame = self.get_frame()
            print(ascii_render(frame))
        elif self.render_mode == "pygame" and self.viewer:
            frame = self.get_frame()
            self.viewer.draw(frame)
            return self.viewer.pump()
        return True