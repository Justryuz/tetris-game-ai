markdown# Tetris DQN - Deep Q-Network untuk Bermain Tetris

Project ini mengimplementasikan Deep Q-Network (DQN) untuk bermain game Tetris.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Training
```bash
python train_fixed.py --episodes 500

3. Training dengan TensorBoard:
```bash
python train_fixed.py --episodes 500 --tb --render_every 50

4. Training dengan visual rendering::
```bash
python train_fixed.py --episodes 500 --render --render_backend ascii


##Evaluation
```bash
python evaluate.py --model models/best_model.pth --episodes 5 --render --delay 0.2



## Cara Menggunakan:

1. **Buat folder baru** untuk project ini
2. **Copy semua file** di atas ke dalam folder tersebut
3. **Pastikan file asli** (env.py, model.py, visualize.py, utils.py) juga ada
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Mulai training**: `python train_fixed.py --episodes 100 --render`
