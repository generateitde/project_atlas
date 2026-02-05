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

=====================================================================
TODO / Roadmap
=====================================================================

NEXT TASK: T3.1 Mode Interface Refactor + Mode HUD

### Next Items
- [x] T1.1 Reward Breakdown + Debug Overlay
- [x] T1.2 Seed Repro & World Snapshot
- [x] T2.1 Tool Contracts + Validation Layer
- [x] T2.2 Jump/Gravity State Machine + One-Way Platforms
- [ ] T3.1 Mode Interface Refactor + Mode HUD
- [ ] T3.2 ExitGame Solvable Generator (Baseline)
- [ ] T3.3 CaptureTheFlag Vollständig
- [ ] T3.4 HideAndSeek Vollständig
- [ ] T4.1 Goal Stack + Subgoals
- [ ] T4.2 Stuck Detector + Uncertainty Gate für ask_human
- [ ] T5.1 EXP/Level Persistenz + Gates
- [ ] T5.2 Transformations
- [ ] T5.3 Fly Items + Floating Islands
- [ ] T6.1 Behavior Cloning Aux Loss richtig
- [ ] T6.2 DAgger Queries + UI

### Backlog
- [ ] T6.3 Preference Reward Model

## EPIC 1: Observability & Determinism
### T1.1 Reward Breakdown + Debug Overlay
- **Problem:** Atlas wirkt zufällig. Wir sehen nicht warum.
- **Scope:** src/render/ui_overlays.py, src/env/rewards.py, src/logging/schema.py, src/main.py
- **Deliverables:**
  - Reward Breakdown pro Step: mode, progress, explore, preference, shaping, step_cost
  - On-screen HUD Toggle (F3): mode, subgoal, last_action, reward breakdown
  - DB Logging: reward_terms_json in steps table
- **Akzeptanztests:**
  - Starte game, drücke F3, HUD zeigt reward terms.
  - DB hat pro step reward_terms_json, nicht null.
- **Done Definition:** HUD + Logging laufen ohne Crash.

### T1.2 Seed Repro & World Snapshot
- **Problem:** Bugs sind nicht reproduzierbar.
- **Scope:** src/core/rng.py, src/env/world_gen.py, src/logging/schema.py, src/console.py
- **Deliverables:**
  - Global seed setzbar per Konsole.
  - World snapshot hash pro Episode.
  - Gleicher seed => gleiche world layout.
- **Akzeptanztests:**
  - world switch dungeon_exit 123 zweimal -> identische tile map.
- **Done Definition:** Repro bestätigt.

## EPIC 2: Tools, Rules, Action Masking (Stabilität)
### T2.1 Tool Contracts + Validation Layer
- **Problem:** Invalid actions, inkonsistente Weltinteraktionen.
- **Scope:** src/env/tools.py, src/env/rules.py, src/env/encoding.py, src/agent/action_masking.py
- **Deliverables:**
  - Einheitliches ToolResult (ok, error_code, events, delta)
  - Preconditions je Tool (adjacent, has_item, grounded, etc)
  - Action masking nutzt Tool precheck ohne Side-Effects
- **Akzeptanztests:**
  - invalid action rate < 0.5% in 1000 steps (headless eval)
- **Done Definition:** Masking + precheck stabil.

### T2.2 Jump/Gravity State Machine + One-Way Platforms
- **Problem:** Sprung und Plattformen sind unzuverlässig.
- **Scope:** src/env/grid_env.py, src/env/rules.py
- **Deliverables:**
  - grounded check, jump impulse, gravity tick
  - platform tiles: passable von unten, solid von oben
  - jump cooldown + fly override hooks
- **Akzeptanztests:**
  - Human kann 10x reproduzierbar auf Plattform springen.
  - Atlas kann in scripted test sequence springen ohne stuck.
- **Done Definition:** Stabiler Jump.

## EPIC 3: Modes als echte Objectives + Solvable WorldGen
### T3.1 Mode Interface Refactor + Mode HUD
- **Problem:** Modes fühlen sich nicht geführt an.
- **Scope:** src/env/modes.py, src/env/rewards.py, src/render/ui_overlays.py
- **Deliverables:**
  - ModeState pydantic pro Mode
  - mode.reset/mode.step/done/info konsistent
  - HUD zeigt mode spezifische Ziele
- **Akzeptanztests:**
  - mode set ExitGame -> HUD zeigt goal.
  - mode set CTF -> HUD zeigt flag status.
- **Done Definition:** Modes konsistent.

### T3.2 ExitGame Solvable Generator (Baseline)
- **Problem:** Agent kann nicht lernen wenn Welt unlösbar ist.
- **Scope:** src/env/world_gen.py, src/env/modes.py, src/env/rewards.py
- **Deliverables:**
  - Generator garantiert solvable Pfad: spawn -> key -> door -> goal
  - Optional breakables als Shortcut, aber nicht nötig
  - 100 seeds: >=95 solvable ohne Exploit
- **Akzeptanztests:**
  - Headless check_solvable(preset, seed) für 100 seeds.
- **Done Definition:** Solvable.

### T3.3 CaptureTheFlag Vollständig
- **Problem:** CTF Rewards/States fehlen.
- **Scope:** src/env/modes.py, src/env/rewards.py, src/env/rules.py
- **Deliverables:**
  - Flag entity, carry state, drop on hit optional
  - Score zone, tick reward, score reward
- **Akzeptanztests:**
  - Atlas kann flag pickup -> reward tick -> score -> reset.
- **Done Definition:** CTF spielbar.

### T3.4 HideAndSeek Vollständig
- **Problem:** Hide spot, find reward, shaping fehlen.
- **Scope:** src/env/modes.py, src/env/rewards.py, src/console.py
- **Deliverables:**
  - Konsole: hide set marker
  - Reward: distance shaping + find bonus
  - Optional time limit
- **Akzeptanztests:**
  - Human setzt hide, Atlas findet nach Training schneller.
- **Done Definition:** HideSeek spielbar.

## EPIC 4: Agent Zielsystem statt "Was soll ich tun?"
### T4.1 Goal Stack + Subgoals
- **Problem:** Atlas hat kein internes Ziel.
- **Scope:** src/agent/trainer.py, src/agent/policy.py, src/agent/world_model.py, src/render/ui_overlays.py
- **Deliverables:**
  - GoalManager: mode goal -> subgoals (find_key, go_door, go_goal, explore)
  - Subgoal im HUD sichtbar
- **Akzeptanztests:**
  - ExitGame zeigt Subgoal Sequenz.
- **Done Definition:** Ziele sichtbar.

### T4.2 Stuck Detector + Uncertainty Gate für ask_human
- **Problem:** Atlas fragt ständig oder looped.
- **Scope:** src/agent/dagger.py, src/agent/policy.py, src/agent/trainer.py
- **Deliverables:**
  - stuck detection über last positions + no progress timer
  - uncertainty aus action entropy oder value error
  - ask_human nur wenn stuck UND uncertainty hoch
- **Akzeptanztests:**
  - Nach 30 min Training: ask_human rate < 1% steps.
- **Done Definition:** Fragen reduziert.

## EPIC 5: Progression (EXP/Level/Gates/Transform/Fly)
### T5.1 EXP/Level Persistenz + Gates
- **Problem:** Progression fehlt oder ist inkonsistent.
- **Scope:** src/env/rules.py, src/env/rewards.py, src/logging/db.py, src/agent/trainer.py
- **Deliverables:**
  - EXP from kills/objectives
  - Level curve
  - Gate tile requires level
  - Persistenz über world switch
- **Akzeptanztests:**
  - Level steigt, Gate blockt unter level, save/load hält.
- **Done Definition:** Gates funktionieren.

### T5.2 Transformations
- **Problem:** Transforms als Skill Unlock fehlen.
- **Scope:** src/env/rules.py, src/env/rewards.py, src/render/renderer.py
- **Deliverables:**
  - Transform state mit duration
  - Sprite swap
  - Stats multipliers
- **Akzeptanztests:**
  - Transform aktiviert, Stats steigen, timer revert.
- **Done Definition:** Transform ok.

### T5.3 Fly Items + Floating Islands
- **Problem:** Unerreichbare Inseln und Fly fehlen.
- **Scope:** src/env/world_gen.py, src/env/rules.py
- **Deliverables:**
  - Floating islands generator
  - Fly item temp/perm
  - Fly movement override
- **Akzeptanztests:**
  - Ohne fly unerreichbar, mit fly erreichbar.
- **Done Definition:** Fly ok.

## EPIC 6: Lernen aus dir (Imitation, DAgger, Preference)
### T6.1 Behavior Cloning Aux Loss richtig
- **Problem:** Atlas nutzt dein Verhalten nicht.
- **Scope:** src/agent/imitation.py, src/agent/trainer.py, src/logging/schema.py
- **Deliverables:**
  - Log human actions dataset
  - BC aux loss integriert
- **Akzeptanztests:**
  - 10 min demos -> Atlas baseline besser.
- **Done Definition:** BC wirkt.

### T6.2 DAgger Queries + UI
- **Problem:** Korrekturen sind nicht effizient.
- **Scope:** src/agent/dagger.py, src/human/chat_ui.py
- **Deliverables:**
  - Atlas fragt konkrete Action choice
  - Human antwortet mit action
  - Daten werden gespeichert und trainiert
- **Akzeptanztests:**
  - Fragen sinken, success steigt.
- **Done Definition:** DAgger ok.

### T6.3 Preference Reward Model
- **Problem:** Chat scoring beeinflusst Verhalten nicht.
- **Scope:** src/agent/preference_reward.py, src/logging/schema.py
- **Deliverables:**
  - Reward model train auf (state features + text + score)
  - preference reward term wird genutzt
- **Akzeptanztests:**
  - Hypothesenqualität steigt, Halluzination sinkt.
- **Done Definition:** Preference ok.
