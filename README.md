
### Tetris DQN - Deep Q-Network untuk Bermain Tetris
Implementasikan Deep Q-Network (DQN) sambil main game Tetris.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt

##Training
```bash
python train_fixed.py --episodes 500

##Training dengan TensorBoard:
```bash
python train_fixed.py --episodes 500 --tb --render_every 50

##Training dengan visual rendering::
```bash
python train_fixed.py --episodes 500 --render --render_backend ascii


##Evaluation
```bash
python evaluate.py --model models/best_model.pth --episodes 5 --render --delay 0.2
