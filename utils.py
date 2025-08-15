from __future__ import annotations
import math
import os
import random
import numpy as np
import torch

class LinearEpsilon:
    def __init__(self, start=1.0, end=0.05, decay_steps=150_000):
        self.start, self.end, self.decay = start, end, decay_steps
        self.eps = start
        self.step_i = 0
    def step(self):
        if self.eps > self.end:
            self.eps = max(self.end, self.eps - (self.start - self.end)/max(1, self.decay))
        self.step_i += 1
        return self.eps


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)