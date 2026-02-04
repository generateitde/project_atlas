# Atlas, Adaptive Thought & Learning Autonomous System (A.T.L.A.S.)

Atlas is a 2D pixel grid RL laboratory featuring a human player and an RL agent (Atlas). The MVP focuses on a runnable Python-only game loop, a Gymnasium environment, RecurrentPPO training, and SQLite logging.

## Features (MVP)
- 2D grid with gravity, jump, breakable tiles, and goal tiles.
- Human control (WASD / arrows + Space to jump).
- Atlas runs as an RL agent (RecurrentPPO) and can be trained headless.
- Console commands for world switching, mode setting, and save/load.
- SQLite logging + JSONL replay export.
- Runs without sprite assets (fallback rectangles).

## Install

### Conda (recommended)
```bash
conda create -n atlas python=3.11
conda activate atlas
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Run Game
```bash
python -m src.main run
```

## Train Headless
```bash
python -m src.main train --steps 20000
```

## Resume Training
```bash
python -m src.main resume --steps 20000
```

## Export Replay
```bash
python -m src.main export --db atlas.db --out replay.jsonl
```

## Console Commands
- `help`
- `world list`
- `world switch <preset> <seed>`
- `mode set <mode> <json_params>`
- `teleport <human|ai_atlas> <x> <y>`
- `pause ai` / `resume ai`
- `save` / `load`
- `reset episode`
- `print state`
- `print map`
- `agent info`

## Notes
- API keys are loaded only from `.env` (see `.env.example`).
- Animation support is left as TODOs for future work.
