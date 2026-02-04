from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pygame
from dotenv import load_dotenv

from src.agent.trainer import Trainer
from src.config import load_config
from src.console import Console
from src.env.grid_env import GridEnv
from src.human.input_keyboard import KeyboardController
from src.logging.db import Database
from src.logging.replay import export_replay
from src.render.renderer import Renderer


def run_game(config_path: Path) -> None:
    config = load_config(config_path)
    pygame.init()
    env = GridEnv(
        width=config.world.width,
        height=config.world.height,
        visibility_radius=config.world.visibility_radius,
        preset="dungeon_exit",
        seed=config.training.seed,
    )
    trainer = Trainer(env, config.training)
    renderer = Renderer(config.rendering.tile_size, (config.world.width, config.world.height))
    screen = pygame.display.set_mode(
        (config.world.width * config.rendering.tile_size, config.world.height * config.rendering.tile_size)
    )
    clock = pygame.time.Clock()
    keyboard = KeyboardController()
    console = Console(env)
    db = Database(Path("atlas.db"))

    obs, _ = env.reset()
    episode_id = 1
    tick = 0

    running = True
    while running:
        human_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    console.ai_paused = not console.ai_paused
                if event.key == pygame.K_F5:
                    trainer.save(Path("src/checkpoints/atlas_model"))
                if event.key == pygame.K_F9:
                    trainer.load(Path("src/checkpoints/atlas_model.zip"))
            action = keyboard.handle_event(event)
            if action is not None:
                human_action = action

        if human_action is not None:
            env.step(human_action)

        if not console.ai_paused:
            action = trainer.predict(obs)
            obs, reward, done, _, info = env.step(action)
            db.log_step(episode_id, tick, obs, action, reward, done, info)
            tick += 1
            if done:
                obs, _ = env.reset()
                tick = 0

        renderer.draw(screen, env.world)
        pygame.display.flip()
        clock.tick(config.rendering.fps)

    pygame.quit()


def train_headless(config_path: Path, steps: int) -> None:
    config = load_config(config_path)
    env = GridEnv(
        width=config.world.width,
        height=config.world.height,
        visibility_radius=config.world.visibility_radius,
        preset="dungeon_exit",
        seed=config.training.seed,
    )
    trainer = Trainer(env, config.training)
    trainer.learn_headless(steps)
    trainer.save(Path("src/checkpoints/atlas_model"))


def resume_training(config_path: Path, steps: int) -> None:
    config = load_config(config_path)
    env = GridEnv(
        width=config.world.width,
        height=config.world.height,
        visibility_radius=config.world.visibility_radius,
        preset="dungeon_exit",
        seed=config.training.seed,
    )
    trainer = Trainer(env, config.training)
    trainer.load(Path("src/checkpoints/atlas_model.zip"))
    trainer.learn_headless(steps)
    trainer.save(Path("src/checkpoints/atlas_model"))


def export_replay_cmd(db_path: Path, output_path: Path) -> None:
    export_replay(db_path, output_path)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("game")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=2000)
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--steps", type=int, default=2000)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--db", default="atlas.db")
    export_parser.add_argument("--out", default="replays/atlas.jsonl")

    args = parser.parse_args()
    config_path = Path(args.config)
    if args.command == "train":
        train_headless(config_path, args.steps)
    elif args.command == "resume":
        resume_training(config_path, args.steps)
    elif args.command == "export":
        export_replay_cmd(Path(args.db), Path(args.out))
    else:
        run_game(config_path)


if __name__ == "__main__":
    main()
