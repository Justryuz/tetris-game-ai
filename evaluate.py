import argparse
import time
import torch
import numpy as np
from tetris_env import Tetris
from dqn_agent import DQNAgent
from visualize import ascii_render

def evaluate_model(model_path, episodes=10, render=True, render_backend="ascii", delay=0.1):
    """Evaluate a trained DQN model."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = Tetris(render_mode=render_backend if render else None)
    state_shape = env.get_state().shape
    n_actions = env.action_space
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=device,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0
    )
    
    # Load trained model
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    
    rewards = []
    lines_cleared_list = []
    
    print(f"Evaluating model: {model_path}")
    print(f"Device: {device}")
    print(f"Episodes: {episodes}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        initial_lines = env.env.lines_cleared_total
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render:
                if render_backend == "ascii":
                    print(f"\nEpisode {episode + 1}, Step {steps}")
                    print(f"Action: {action}, Reward: {reward:.3f}")
                    print(ascii_render(env.get_frame()))
                    if delay > 0:
                        time.sleep(delay)
                elif render_backend == "pygame":
                    if not env.render():
                        break
                    if delay > 0:
                        time.sleep(delay)
        
        lines_cleared = env.env.lines_cleared_total - initial_lines
        rewards.append(episode_reward)
        lines_cleared_list.append(lines_cleared)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}, Lines={lines_cleared}")
    
    # Print statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Best Reward: {np.max(rewards):.2f}")
    print(f"Worst Reward: {np.min(rewards):.2f}")
    print(f"Average Lines Cleared: {np.mean(lines_cleared_list):.2f} ± {np.std(lines_cleared_list):.2f}")
    print(f"Best Lines Cleared: {np.max(lines_cleared_list)}")
    print(f"Total Lines Cleared: {np.sum(lines_cleared_list)}")
    
    return rewards, lines_cleared_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Tetris DQN model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--render_backend", choices=["ascii", "pygame"], default="ascii")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        episodes=args.episodes,
        render=args.render,
        render_backend=args.render_backend,
        delay=args.delay
    )