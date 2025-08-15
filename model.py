from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Deque
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from env import BOARD_H, BOARD_W


class QNet(nn.Module):
    def __init__(self, in_channels=1, num_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * BOARD_H * BOARD_W, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )
    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.from_numpy(np.stack([t.s for t in batch])).float()
        a = torch.tensor([t.a for t in batch], dtype=torch.int64)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32)
        s2 = torch.from_numpy(np.stack([t.s2 for t in batch])).float()
        d = torch.tensor([t.d for t in batch], dtype=torch.bool)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)


def dqn_loss(policy: QNet, target: QNet, batch, gamma: float, device: str):
    s, a, r, s2, d = batch
    s = s.to(device)
    s2 = s2.to(device)
    a = a.to(device)
    r = r.to(device)
    d = d.to(device)

    q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target(s2).max(dim=1).values
        y = r + (~d).float() * gamma * q_next
    return nn.SmoothL1Loss()(q, y)