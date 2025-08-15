from __future__ import annotations
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