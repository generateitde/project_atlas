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
        self.renderer = None
        self.keyboard = KeyboardController(self.env.world)
        self.trainer = AtlasTrainer(self.config, Path("checkpoints"))
        self.trainer.load(self.env)
        self.db = DBLogger(Path("atlas.db"))
        self.db.start_episode(self.env.preset, self.env.seed_value, self.env.mode.name, datetime.utcnow().isoformat())
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
        surface = pygame.display.set_mode((width * tile_size, height * tile_size + 120))
        pygame.display.set_caption("Atlas RL Grid")
        clock = pygame.time.Clock()
        self.renderer = Renderer(tile_size, width, height)

        running = True
        obs, _ = self.env.reset()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKQUOTE:
                    self.console.active = not self.console.active
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    if self.console.active:
                        self.console.last_message = self.console.execute(self, self.console.buffer)
                        self.console.buffer = ""
                    else:
                        self.chat_active = not self.chat_active
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    self.ai_paused = not self.ai_paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
                    self.save()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F9:
                    self.load()
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

            if not self.ai_paused:
                action, self.recurrent_state = self.trainer.predict(obs, state=self.recurrent_state, mask=self.episode_start)
                obs, reward, done, _, info = self.env.step(int(action))
                self.db.log_step(obs, int(action), float(reward), bool(done), info)
                self.episode_start = done
                if done:
                    obs, _ = self.env.reset()

            if self.renderer:
                self.renderer.render(surface, self.env.world, self.env.mode.name, self.env.world.messages)
                if self.console.active:
                    font = self.renderer.font
                    line = font.render("> " + self.console.buffer, True, (200, 200, 200))
                    surface.blit(line, (4, self.config.world.height * tile_size + 70))
                    if self.console.last_message:
                        msg = font.render(self.console.last_message, True, (120, 200, 120))
                        surface.blit(msg, (4, self.config.world.height * tile_size + 90))
                if self.chat_active:
                    font = self.renderer.font
                    line = font.render("Chat: " + self.chat_buffer, True, (200, 200, 200))
                    surface.blit(line, (4, self.config.world.height * tile_size + 50))

            pygame.display.flip()
            clock.tick(self.config.rendering.fps)
        pygame.quit()


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
