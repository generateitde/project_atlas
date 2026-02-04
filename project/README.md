# Atlas, Adaptive Thought & Learning Autonomous System (A.T.L.A.S.)

A minimal 2D pixel grid RL lab with a human player and Atlas (AI). The project is designed as a runnable MVP with a grid world, RL agent, logging, and console controls.

## Requirements
- Python 3.11+

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run game
```bash
python -m src.main
```

## Train headless
```bash
python -m src.main train --steps 5000
```

## Resume training
```bash
python -m src.main resume --steps 5000
```

## Export replay
```bash
python -m src.main export --db atlas.db --out replays/atlas.jsonl
```

## Controls
- WASD / Arrow keys: Move
- Space: Jump
- E: Pickup
- F: Attack
- B: Break
- I: Inspect
- Tab: Pause/Resume AI
- F5: Save checkpoint
- F9: Load checkpoint

## Notes
- Fallback sprites are colored rectangles; TODO for animation swaps.
- Use `configs/default.yaml` to configure training/rendering.
