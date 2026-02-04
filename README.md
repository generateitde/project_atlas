# Atlas, Adaptive Thought & Learning Autonomous System (A.T.L.A.S.)

Atlas is a 2D pixel grid RL laboratory featuring a human player and an RL agent (Atlas). The MVP focuses on a runnable Python-only game loop, a Gymnasium environment, RecurrentPPO training, and SQLite logging.

## Features (MVP)
- 2D grid with gravity, jump, breakable tiles, and goal tiles.
- Human control (WASD / arrows + Space to jump).
- Atlas runs as an RL agent (RecurrentPPO) and can be trained headless.
- Console commands for world switching, mode setting, AI mode, control target, and save/load.
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

## Controls & Interaction
- **Movement (keyboard):** WASD or arrow keys move the currently controlled character. Space jumps. `B` breaks the tile in front, `I` inspects the tile in front.
- **Toggle console:** press <kbd>`</kbd> (backquote) to open/close the console input.
- **Chat input:** press <kbd>Enter</kbd> to open/close chat input. In *query mode* this is how you answer Atlas; pressing Enter again submits the response.
- **Pause AI:** press <kbd>Tab</kbd> to pause/resume the AI.
- **Fullscreen:** press <kbd>F11</kbd> to toggle fullscreen.
- **Click-to-inspect:** while the console is open, left-click a tile to show its info (tile type + actors) in the console output line.

## AI Modes & Chat Flow
- **Explore mode:** the AI continuously acts. You can still move the controlled character and use the console.
- **Query mode:** after a short interval the AI asks “Was soll ich als Nächstes tun?” and then *pauses* until you reply in chat. Once you close chat (press Enter again), the response is logged and the AI continues.
- **Goal text:** set a goal with `goal set <text>` and it appears in the HUD for reference.

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
- `goal set <text>`
- `ai mode <explore|query>`
- `control <human|ai_atlas>`
- `teleport <human|ai_atlas> <x> <y>`
- `pause ai` / `resume ai`
- `save` / `load`
- `reset episode`
- `print state`
- `print map`
- `agent info`

## Gameplay Notes
- If you want to control Atlas instead of the human character, run `control ai_atlas` in the console.
- If you want to control the human character again, run `control human`.

## Notes
- API keys are loaded only from `.env` (see `.env.example`).
- Animation support is left as TODOs for future work.
