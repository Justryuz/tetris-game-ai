# Tetris DQN - Deep Q-Network untuk Bermain Tetris

Implementasikan Deep Q-Network (DQN) sambil main game Tetris.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Struktur File
```
project/
├── env.py              # Tetris environment core
├── model.py            # Neural network models  
├── visualize.py        # Visualization utilities
├── utils.py            # Utility functions
├── tetris_env.py       # Environment wrapper
├── dqn_agent.py        # DQN agent implementation
├── train_fixed.py      # Training script (gunakan ini)
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
└── README.md           # File ini
```

## Training

**Training basic:**
```bash
python train_fixed.py --episodes 500
```

**Training dengan TensorBoard:**
```bash
python train_fixed.py --episodes 500 --tb --render_every 50
```

**Training dengan visual rendering:**
```bash
python train_fixed.py --episodes 500 --render --render_backend ascii
```

**Training quick test:**
```bash
python train_fixed.py --episodes 10 --render --render_every 2
```

## Evaluation

**Evaluate model:**
```bash
python evaluate.py --model models/best_model.pth --episodes 5 --render --delay 0.2
```

**Evaluate tanpa rendering (cepat):**
```bash
python evaluate.py --model models/best_model.pth --episodes 20
```

## Parameters Penting

- `--episodes`: Berapa episode untuk training
- `--batch_size`: Batch size untuk training (default: 64)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon_start/end/decay`: Parameter untuk epsilon-greedy exploration
- `--lr`: Learning rate (default: 1e-4)
- `--target_update`: Frequency untuk update target network (default: 1000)
- `--render`: Enable visual rendering
- `--tb`: Enable TensorBoard logging
- `--save_every`: Save model setiap N episodes

## Models Yang Disimpan

Models akan disimpan di folder `models/`:
- `best_model.pth`: Model dengan reward terbaik
- `final_model.pth`: Model setelah training selesai
- `model_episode_X.pth`: Model setiap X episodes

## Actions dalam Game

- 0: No operation (tidak buat apa-apa)
- 1: Move left (gerak kiri)
- 2: Move right (gerak kanan)  
- 3: Rotate clockwise (putar)
- 4: Soft drop (jatuh perlahan)
- 5: Hard drop (jatuh terus)

## Tips Training

1. **Mula dengan episode kecil untuk test:**
   ```bash
   python train_fixed.py --episodes 10 --render
   ```

2. **Untuk training serius (tanpa render untuk kecepatan):**
   ```bash
   python train_fixed.py --episodes 1000 --tb
   ```

3. **Monitor progress dengan TensorBoard:**
   ```bash
   tensorboard --logdir=runs
   ```

4. **Evaluate progress secara berkala:**
   ```bash
   python evaluate.py --model models/model_episode_100.pth --episodes 5 --render
   ```
