from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pygame
from dotenv import load_dotenv
from gymnasium.vector import SyncVectorEnv

from src.agent.trainer import AtlasTrainer
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.console import Console
from src.env.grid_env import GridEnv
from src.env import encoding
from src.human.input_keyboard import KeyboardController
from src.logging.db import DBLogger
from src.logging.replay import export_steps
from src.render.renderer import Renderer


class AtlasGame:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config = load_config(config_path)
        self.env = GridEnv(self.config)
        self.console = Console()
        self.chat_active = False
        self.chat_buffer = ""
        self.ai_paused = False
        self.ai_mode = "explore"
        self.waiting_for_response = False
        self.steps_since_question = 0
        self.question_interval = 120
        self.goal_text = ""
        self.fullscreen = False
        self.show_debug_hud = False
        self.last_action_name = "None"
        self.last_reward_terms: dict[str, float] = {}
        self.subgoal_text = "explore"
        self.renderer = None
        self.keyboard = KeyboardController(self.env.world, self.config.controls)
        self.trainer = AtlasTrainer(self.config, Path("checkpoints"))
        self.trainer.load(self.env)
        self.db = DBLogger(Path("atlas.db"))
        self.db.start_episode(
            self.env.preset,
            self.env.seed_value,
            self.env.mode.name,
            datetime.utcnow().isoformat(),
            self.env.world_hash,
        )
        self.recurrent_state = None
        self.episode_start = True

    def switch_world(self, preset: str, seed: int) -> None:
        self.env.preset = preset
        self.env.seed_value = seed
        self.env.reset(seed=seed)
        self.keyboard.world = self.env.world

    def reset_episode(self) -> None:
        self.env.reset()
        self.keyboard.world = self.env.world
        self.episode_start = True

    def set_seed(self, seed: int) -> None:
        self.config.training.seed = seed
        self.env.seed_value = seed
        self.env.reset(seed=seed)
        self.keyboard.world = self.env.world
        self.episode_start = True

    def save(self) -> None:
        self.trainer.save()
        save_payload = {
            "preset": self.env.preset,
            "seed": self.env.seed_value,
            "atlas": {
                "level": self.env.world.atlas.level,
                "exp": self.env.world.atlas.exp,
            },
        }
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        Path("checkpoints/state.json").write_text(json.dumps(save_payload), encoding="utf-8")

    def load(self) -> None:
        state_path = Path("checkpoints/state.json")
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            self.switch_world(data.get("preset", self.env.preset), data.get("seed", self.env.seed_value))
            self.env.world.atlas.level = data.get("atlas", {}).get("level", 1)
            self.env.world.atlas.exp = data.get("atlas", {}).get("exp", 0)
        self.trainer.load(self.env)

    def run(self) -> None:
        pygame.init()
        tile_size = self.config.rendering.tile_size
        width = self.config.world.width
        height = self.config.world.height
        surface = self._create_display(width, height, tile_size)
        pygame.display.set_caption("Atlas RL Grid")
        clock = pygame.time.Clock()
        self.renderer = Renderer(tile_size, width, height)

        running = True
        obs, _ = self.env.reset()
        self.subgoal_text = self.trainer.update_goals(self.env.mode.name, self.env.mode.info())
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_BACKQUOTE, pygame.K_F1):
                    self.console.active = not self.console.active
                    if self.console.active:
                        self.chat_active = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    if self.console.active:
                        self.console.last_message = self.console.execute(self, self.console.buffer)
                        self.console.buffer = ""
                    elif self.chat_active:
                        message = self.chat_buffer.strip()
                        if message:
                            self.env.world.messages.append(("Human", message))
                            if self.env.world.pending_question:
                                self.env.world.pending_question = False
                        self.chat_buffer = ""
                        self.chat_active = False
                    else:
                        self.chat_active = not self.chat_active
                        if not self.chat_active and self.waiting_for_response:
                            response = self.chat_buffer.strip()
                            if response:
                                self.env.world.messages.append(("Human", response))
                            self.chat_buffer = ""
                            self.waiting_for_response = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    self.ai_paused = not self.ai_paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self.fullscreen = not self.fullscreen
                    surface = self._create_display(width, height, tile_size)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    self.show_debug_hud = not self.show_debug_hud
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
                    self.save()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F9:
                    self.load()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self.fullscreen = not self.fullscreen
                    surface = self._create_display(width, height, tile_size)
                elif event.type == pygame.KEYDOWN:
                    if self.console.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.console.buffer = self.console.buffer[:-1]
                        else:
                            self.console.buffer += event.unicode
                    elif self.chat_active:
                        if event.key == pygame.K_BACKSPACE:
                            self.chat_buffer = self.chat_buffer[:-1]
                        else:
                            self.chat_buffer += event.unicode
                    else:
                        self.keyboard.handle_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.console.active:
                        mouse_x, mouse_y = event.pos
                        grid_width = width * tile_size
                        grid_height = height * tile_size
                        if mouse_x < grid_width and mouse_y < grid_height:
                            grid_x = mouse_x // tile_size
                            grid_y = mouse_y // tile_size
                            self.console.last_message = self.env.world.describe_at((grid_x, grid_y))

            if self.ai_mode == "query" and not self.waiting_for_response:
                self.steps_since_question += 1
                if self.steps_since_question >= self.question_interval:
                    self.env.world.messages.append(("Atlas (Frage)", "Was soll ich als Nächstes tun?"))
                    self.waiting_for_response = True
                    self.steps_since_question = 0

            if not self.ai_paused and not self.waiting_for_response:
                action, self.recurrent_state = self.trainer.predict(obs, state=self.recurrent_state, mask=self.episode_start)
                obs, reward, done, _, info = self.env.step(int(action))
                self.last_action_name = encoding.ACTION_MEANINGS.get(int(action), f"Action {int(action)}")
                self.last_reward_terms = info.get("reward_terms", {})
                self.subgoal_text = self.trainer.update_goals(self.env.mode.name, self.env.mode.info())
                self.db.log_step(obs, int(action), float(reward), bool(done), info, self.last_reward_terms)
                self.episode_start = done
                if done:
                    obs, _ = self.env.reset()
                    self.keyboard.world = self.env.world

            if self.renderer:
                self.renderer.render(
                    surface,
                    self.env.world,
                    self.env.mode.name,
                    self.env.world.messages,
                    goal_text=self.goal_text,
                    ai_mode=self.ai_mode,
                    waiting_for_response=self.waiting_for_response,
                    debug_hud=self.show_debug_hud,
                    last_action=self.last_action_name,
                    reward_terms=self.last_reward_terms,
                    subgoal_text=self.subgoal_text,
                    mode_info=self.env.mode.info(),
                )
                font = self.renderer.font
                ui = self.renderer.ui
                max_text_width = surface.get_width() - 8
                chat_bottom = self.renderer.last_chat_bottom
                if self.console.active:
                    input_y = max(chat_bottom + 6, self.config.world.height * tile_size + 70)
                    ui.draw_wrapped_text(surface, "> " + self.console.buffer, (4, input_y), max_text_width, (200, 200, 200))
                    if self.console.last_message:
                        output_y = input_y + font.get_linesize() * 2
                        ui.draw_wrapped_text(surface, self.console.last_message, (4, output_y), max_text_width, (120, 200, 120))
                if self.chat_active:
                    input_y = max(chat_bottom + 6, self.config.world.height * tile_size + 50)
                    ui.draw_wrapped_text(surface, "Chat: " + self.chat_buffer, (4, input_y), max_text_width, (200, 200, 200))

            pygame.display.flip()
            clock.tick(self.config.rendering.fps)
        pygame.quit()

    def _create_display(self, width: int, height: int, tile_size: int) -> pygame.Surface:
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        return pygame.display.set_mode((width * tile_size, height * tile_size + 120), flags)


def train_headless(config_path: Path | None, steps: int) -> None:
    config = load_config(config_path)
    def make_env():
        return GridEnv(config)

    env = SyncVectorEnv([make_env for _ in range(config.training.n_envs)])
    trainer = AtlasTrainer(config, Path("checkpoints"))
    trainer.load(env)
    trainer.train_steps(steps)
    trainer.save()


def resume_training(config_path: Path | None, steps: int) -> None:
    train_headless(config_path, steps)


def export_replay(db_path: Path, out_path: Path) -> None:
    export_steps(db_path, out_path)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Atlas RL Grid")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("run")
    run_cmd.set_defaults(command="run")

    train_cmd = subparsers.add_parser("train")
    train_cmd.add_argument("--steps", type=int, default=10000)

    resume_cmd = subparsers.add_parser("resume")
    resume_cmd.add_argument("--steps", type=int, default=10000)

    export_cmd = subparsers.add_parser("export")
    export_cmd.add_argument("--db", type=Path, default=Path("atlas.db"))
    export_cmd.add_argument("--out", type=Path, default=Path("replay.jsonl"))

    args = parser.parse_args()
    if args.command == "train":
        train_headless(args.config, args.steps)
    elif args.command == "resume":
        resume_training(args.config, args.steps)
    elif args.command == "export":
        export_replay(args.db, args.out)
    else:
        AtlasGame(args.config).run()


if __name__ == "__main__":
    main()
