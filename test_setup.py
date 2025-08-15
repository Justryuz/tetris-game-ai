#!/usr/bin/env python3
"""
Test script untuk memastikan semua komponen berfungsi dengan betul
Jalankan: python test_setup.py
"""

import sys
import traceback

def test_imports():
    """Test semua import yang diperlukan"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(" numpy")
    except ImportError as e:
        print(f" numpy: {e}")
        return False
        
    try:
        import torch
        print(" torch")
    except ImportError as e:
        print(f" torch: {e}")
        return False
    
    try:
        from env import TetrisEnv, TETROMINOES
        print(" env.py")
    except ImportError as e:
        print(f"✗ env.py: {e}")
        return False
    
    try:
        from model import QNet, ReplayBuffer
        print(" model.py")
    except ImportError as e:
        print(f" model.py: {e}")
        return False
    
    try:
        from visualize import ascii_render, PygameViewer
        print(" visualize.py")
    except ImportError as e:
        print(f" visualize.py: {e}")
        return False
    
    try:
        from tetris_env import Tetris
        print(" tetris_env.py")
    except ImportError as e:
        print(f" tetris_env.py: {e}")
        return False
    
    try:
        from dqn_agent import DQNAgent
        print(" dqn_agent.py")
    except ImportError as e:
        print(f" dqn_agent.py: {e}")
        return False
    
    return True

def test_environment():
    """Test Tetris environment"""
    print("\nTesting environment...")
    
    try:
        from tetris_env import Tetris
        from visualize import ascii_render
        
        # Create environment
        env = Tetris()
        print(" Environment created")
        
        # Reset environment
        state = env.reset()
        print(f" Environment reset, state shape: {state.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.env.rng.randint(0, 6)  # Random action
            next_state, reward, done = env.step(action)
            print(f" Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
            if done:
                break
        
        # Test rendering
        frame = env.get_frame()
        rendered = ascii_render(frame)
        print("ASCII rendering works")
        print("Sample render:")
        print(rendered[:200] + "..." if len(rendered) > 200 else rendered)
        
        return True
    except Exception as e:
        print(f" Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_agent():
    """Test DQN agent"""
    print("\nTesting DQN agent...")
    
    try:
        from tetris_env import Tetris
        from dqn_agent import DQNAgent
        import torch
        
        device = torch.device("cpu")  # Use CPU for testing
        
        # Create environment and agent
        env = Tetris()
        state_shape = env.get_state().shape
        n_actions = env.action_space
        
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            device=device,
            memory_size=1000,  # Small for testing
            batch_size=32
        )
        print(" Agent created")
        
        # Test action selection
        state = env.reset()
        action = agent.select_action(state)
        print(f" Action selection works: {action}")
        
        # Test storing transition
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        print(" Transition storage works")
        
        # Fill memory for training test
        for _ in range(35):  # Ensure we have enough for batch
            state = env.reset()
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
        
        # Test training
        loss = agent.train_step()
        if loss is not None:
            print(f" Training works, loss: {loss:.4f}")
        else:
            print(" Training step (not enough data yet)")
        
        # Test target network update
        agent.update_target_network()
        print(" Target network update works")
        
        return True
    except Exception as e:
        print(f" Agent test failed: {e}")
        traceback.print_exc()
        return False

def test_training_script():
    """Test if training script can be imported"""
    print("\nTesting training script...")
    
    try:
        import train_fixed
        print(" train_fixed.py can be imported")
        return True
    except Exception as e:
        print(f" train_fixed.py import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("TETRIS DQN SETUP TEST")
    print("=" * 60)
    
    all_tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Agent Test", test_agent),
        ("Training Script Test", test_training_script)
    ]
    
    passed = 0
    total = len(all_tests)
    
    for test_name, test_func in all_tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f" {test_name} PASSED")
                passed += 1
            else:
                print(f" {test_name} FAILED")
        except Exception as e:
            print(f" {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("ALL TESTS PASSED! Setup is ready for training.")
        print("\nNext steps:")
        print("1. python train_fixed.py --episodes 10 --render")
        print("2. python train_fixed.py --episodes 500 --tb")
    else:
        print("Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
