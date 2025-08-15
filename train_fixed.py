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
from visualize import ascii_render

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
parser.add_argument("--render_every", type=int, default=100, help="Render every N episodes")
parser.add_argument("--save_every", type=int, default=100, help="Save model every N episodes")
parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
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
# Create save directory
# -------------------
os.makedirs(args.save_dir, exist_ok=True)

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
episode_rewards = deque(maxlen=100)
best_reward = float('-inf')

print("Starting training...")

for episode in range(args.episodes):
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False

    while not done and episode_steps < args.max_steps:
        global_step += 1
        episode_steps += 1

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

        # Update target network
        if global_step % args.target_update == 0:
            agent.update_target_network()

        # Epsilon decay step
        agent.update_epsilon()

    # Episode finished
    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    
    if episode_reward > best_reward:
        best_reward = episode_reward
        # Save best model
        torch.save(agent.policy_net.state_dict(), 
                  os.path.join(args.save_dir, "best_model.pth"))

    if args.tb:
        writer.add_scalar("Reward/episode", episode_reward, episode)
        writer.add_scalar("Reward/average", avg_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)
        writer.add_scalar("Episode_Steps", episode_steps, episode)

    # Rendering every N episodes
    if args.render and episode % args.render_every == 0:
        print(f"\nRendering episode {episode + 1}:")
        if args.render_backend == "ascii":
            print(ascii_render(env.get_frame()))
        else:
            env.render()

    # Save model every N episodes
    if (episode + 1) % args.save_every == 0:
        torch.save(agent.policy_net.state_dict(), 
                  os.path.join(args.save_dir, f"model_episode_{episode+1}.pth"))

    print(f"[Episode {episode+1}/{args.episodes}] "
          f"Reward: {episode_reward:.2f} "
          f"Avg: {avg_reward:.2f} "
          f"Best: {best_reward:.2f} "
          f"Steps: {episode_steps} "
          f"Epsilon: {agent.epsilon:.4f}")

# Save final model
torch.save(agent.policy_net.state_dict(), 
          os.path.join(args.save_dir, "final_model.pth"))

if args.tb:
    writer.close()

print("Training completed!")
print(f"Best reward achieved: {best_reward:.2f}")
print(f"Models saved in: {args.save_dir}")