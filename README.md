# Self-Navigating Warehouse Robot (Q-Learning vs SARSA vs DQN)

A side-by-side reinforcement-learning comparison: three agents control three
robots solving the same delivery task in parallel, with a live leaderboard
ranking them by success rate, score, and speed.

This repo implements:
- Tabular **Q-Learning** (off-policy)
- Tabular **SARSA** (on-policy)
- **Dueling Double-DQN** (CNN + MLP, replay buffer, target network)

The environment is a custom Gymnasium env with crates (walls/obstacles),
dynamic traffic congestion, a charger, and a green delivery drop-off pad.

## Quickstart

1) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the live demo

```bash
python3 compare_run.py
```

While it runs, you’ll see a top-down visualization of the warehouse for the
three algorithms and a live leaderboard strip. Press `Ctrl+C` to exit.

## Notes about checkpoints

`compare_run.py` will load a cached DQN model if it exists at:
- `checkpoints/compare_dqn_v6.pt`

Because `checkpoints/` is ignored by git (so your repo stays small), the
first run may train the DQN once. Subsequent runs are faster because the
checkpoint is reused locally.

## Project structure

```text
RL_project/
├── compare_run.py               # demo entry point
├── abstract.md                  # detailed project description
├── requirements.txt
│
├── agents/
│   ├── q_learning_agent.py
│   ├── sarsa_agent.py
│   ├── dqn_agent.py
│   └── replay_buffer.py
│
├── environment/
│   ├── city_env.py
│   └── traffic_manager.py
│
└── models/
    └── networks.py
```

## How it’s evaluated

The demo uses **disjoint seeds**:
- tabular agents and DQN are pre-trained on `TRAIN_SEEDS` (shown in
  `compare_run.py`)
- live demo episodes run on `DEMO_SEEDS` that were not seen during
  pre-training

This is intended to measure generalization rather than memorization.

## License

Add a license file if you want one. (None is set up yet.)

