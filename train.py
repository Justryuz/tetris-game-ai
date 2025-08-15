import argparse
import os
import random
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tetris_env import Tetris
from dqn_agent import DQNAgent
from visualize import ascii_render  # Fixed import

# -------------------
# Argument Parser
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--epsilon_start", type=float, default=1.0)
parser.add_argument("--epsilon_end", type=float, default=0.1)
parser.add_argument("--epsilon_decay", type=int, default=50000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--target_update", type=int, default=1000)
parser.add_argument("--memory_size", type=int, default=50000)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
parser.add_argument("--render", action="store_true", help="Enable live rendering")
parser.add_argument("--render_backend", choices=["ascii", "pygame"], default="ascii")
parser.add_argument("--render_every", type=int, default=1000, help="Render every N steps")
args = parser.parse_args()

# -------------------
# Device & Seed
# -------------------
device = torch.device(args.device)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print(f"Using device: {device}")

# -------------------
# TensorBoard
# -------------------
if args.tb:
    writer = SummaryWriter()
else:
    writer = None

# -------------------
# Init Environment & Agent
# -------------------
env = Tetris(render_mode=args.render_backend if args.render else None)
state_shape = env.get_state().shape
n_actions = env.action_space

print(f"State shape: {state_shape}")
print(f"Number of actions: {n_actions}")

agent = DQNAgent(
    state_shape=state_shape,
    n_actions=n_actions,
    device=device,
    gamma=args.gamma,
    epsilon_start=args.epsilon_start,
    epsilon_end=args.epsilon_end,
    epsilon_decay=args.epsilon_decay,
    lr=args.lr,
    memory_size=args.memory_size,
    batch_size=args.batch_size,
    target_update=args.target_update
)

# -------------------
# Training Loop
# -------------------
global_step = 0

print("Starting training...")

for episode in range(args.episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done and global_step < args.max_steps:
        global_step += 1

        # Select action (epsilon-greedy)
        action = agent.select_action(state)

        # Take action in environment
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train from replay buffer
        if len(agent.memory) >= args.batch_size:
            loss = agent.train_step()
            if args.tb and loss is not None:
                writer.add_scalar("Loss/train", loss, global_step)

        # Update target network - FIXED: Added this missing part
        if global_step % args.target_update == 0:
            agent.update_target_network()

        # Epsilon decay step
        agent.update_epsilon()

        # Live rendering every N steps
        if args.render and global_step % args.render_every == 0:
            if args.render_backend == "ascii":
                print(f"\nStep {global_step}, Episode {episode + 1}")
                print(ascii_render(env.get_frame()))
            elif args.render_backend == "pygame":
                env.render()

    # Episode finished
    if args.tb:
        writer.add_scalar("Reward/episode", episode_reward, episode)

    print(f"[Episode {episode+1}/{args.episodes}] Reward: {episode_reward:.2f} Epsilon: {agent.epsilon:.4f}")

if args.tb:
    writer.close()

print("Training completed!")
