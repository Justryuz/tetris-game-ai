#!/usr/bin/env python3
"""
Setup script untuk Tetris DQN Project
Jalankan: python setup.py
"""

import os

def create_file(filename, content):
    """Create a file with given content."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created {filename}")

def main():
    print("Setting up Tetris DQN Project...")
    print("=" * 50)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    print("✓ Created models directory")
    
    # dqn_agent.py
    dqn_agent_content = '''from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNet, ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        device,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=50000,
        lr=1e-4,
        memory_size=50000,
        batch_size=64,
        target_update=1000
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        
        # Networks
        self.policy_net = QNet(state_shape[0], n_actions).to(device)
        self.target_net = QNet(state_shape[0], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
            
        batch = self.memory.sample(self.batch_size)
        loss = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())
        
        return nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
    
    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
'''
    
    # tetris_env.py
    tetris_env_content = '''from __future__ import annotations
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
'''
    
    # requirements.txt
    requirements_content = '''torch>=1.9.0
numpy>=1.21.0
tensorboard>=2.8.0
pygame>=2.1.0
'''
    
    # Create files
    create_file("dqn_agent.py", dqn_agent_content)
    create_file("tetris_env.py", tetris_env_content)
    create_file("requirements.txt", requirements_content)
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. python train_fixed.py --episodes 100 --render")
    print("=" * 50)

if __name__ == "__main__":
    main()