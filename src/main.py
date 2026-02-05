from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pygame
from dotenv import load_dotenv

from src.agent.trainer import AtlasTrainer
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.console import Console
from src.env.grid_env import GridEnv
from src.env import encoding
from src.human.input_keyboard import KeyboardController
from src.human.chat_ui import format_action_choices, parse_human_action_choice
from src.agent.preference_reward import extract_state_features, parse_scored_feedback
from src.logging.db import DBLogger, write_eval_trend_report
from src.logging.replay import export_steps
from src.render.renderer import Renderer
from src.eval.harness import DeterministicEvalHarness
from src.runtime.inference import export_policy_artifact, load_runtime_policy


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
        self.last_stuck_state = False
        self.last_uncertainty = 0.0

    @staticmethod
    def _key_to_action(key: int) -> int | None:
        mapping = {
            pygame.K_w: 1,
            pygame.K_UP: 1,
            pygame.K_d: 2,
            pygame.K_RIGHT: 2,
            pygame.K_s: 3,
            pygame.K_DOWN: 3,
            pygame.K_a: 4,
            pygame.K_LEFT: 4,
            pygame.K_SPACE: 5,
            pygame.K_b: 10,
            pygame.K_i: 11,
        }
        return mapping.get(key)

    def switch_world(self, preset: str, seed: int) -> None:
        progression = self.trainer.snapshot_progression(self.env.world.atlas)
        self.env.preset = preset
        self.env.seed_value = seed
        self.env.reset(seed=seed)
        self.trainer.restore_progression(self.env.world.atlas, progression)
        self.keyboard.world = self.env.world

    def reset_episode(self) -> None:
        self.env.reset()
        self.keyboard.world = self.env.world
        self.episode_start = True
        self.trainer.reset_dagger()

    def set_seed(self, seed: int) -> None:
        progression = self.trainer.snapshot_progression(self.env.world.atlas)
        self.config.training.seed = seed
        self.env.seed_value = seed
        self.env.reset(seed=seed)
        self.trainer.restore_progression(self.env.world.atlas, progression)
        self.keyboard.world = self.env.world
        self.episode_start = True
        self.trainer.reset_dagger()

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


    def _request_dagger_action(self, obs: dict) -> None:
        choices = format_action_choices(obs.get("action_mask"))
        self.env.world.messages.append(("Atlas (Frage)", f"Welche Aktion? {choices}"))
        self.waiting_for_response = True

    def _handle_human_query_response(self, obs: dict) -> None:
        response = self.chat_buffer.strip()
        self.chat_buffer = ""
        self.waiting_for_response = False
        if not response:
            self.env.world.messages.append(("Atlas", "Keine Antwort erhalten, ich entscheide selbst."))
            return
        self.env.world.messages.append(("Human", response))
        parsed_action = parse_human_action_choice(response, obs.get("action_mask"))
        if parsed_action is None:
            self.env.world.messages.append(("Atlas", "Antwort nicht als gültige Aktion erkannt."))
            return
        self.db.log_human_action(obs, parsed_action)
        self.trainer.record_human_action(obs, parsed_action)
        self.env.world.messages.append(("Atlas", f"Danke, ich nutze Aktion {parsed_action}."))

    def _handle_preference_feedback(self, obs: dict, message: str) -> bool:
        score, feedback_text = parse_scored_feedback(message)
        if score is None:
            return False
        text_for_model = feedback_text or self.last_action_name
        self.trainer.record_preference_feedback(obs, text_for_model, score)
        state_features = extract_state_features(obs).tolist()
        self.db.log_human_feedback(
            target="atlas",
            msg_type="preference",
            score=score,
            correction_text=text_for_model,
            state_features=state_features,
        )
        self.env.world.messages.append(("Atlas", f"Preference gespeichert: {score:+d} für '{text_for_model}'."))
        return True

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
        self.trainer.reset_dagger()
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
                        if self.waiting_for_response:
                            self.chat_active = False
                            self._handle_human_query_response(obs)
                        else:
                            message = self.chat_buffer.strip()
                            if message:
                                self.env.world.messages.append(("Human", message))
                                if self.env.world.pending_question:
                                    self.env.world.pending_question = False
                                self._handle_preference_feedback(obs, message)
                            self.chat_buffer = ""
                            self.chat_active = False
                    else:
                        self.chat_active = not self.chat_active
                        if not self.chat_active and self.waiting_for_response:
                            self._handle_human_query_response(obs)
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
                        if self.keyboard.target_id == "ai_atlas":
                            action = self._key_to_action(event.key)
                            if action is not None:
                                demo_obs = self.env._obs()
                                self.db.log_human_action(demo_obs, action)
                                self.trainer.record_human_action(demo_obs, action)
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

            if not self.ai_paused and not self.waiting_for_response:
                action, self.recurrent_state = self.trainer.predict(obs, state=self.recurrent_state, mask=self.episode_start)
                current_obs = obs
                action_name = encoding.ACTION_MEANINGS.get(int(action), f"Action {int(action)}")
                predicted_preference = self.trainer.preference_reward(current_obs, action_name)
                obs, reward, done, _, info = self.env.step(int(action), preference_reward=predicted_preference)
                self.last_action_name = action_name
                self.last_reward_terms = info.get("reward_terms", {})
                progress_signal = float(self.last_reward_terms.get("progress", 0.0))
                atlas_pos = (int(self.env.world.atlas.pos.x), int(self.env.world.atlas.pos.y))
                self.last_stuck_state = self.trainer.update_stuck_state(atlas_pos, progress_signal, done=bool(done))
                action_mask = obs.get("action_mask")
                if action_mask is not None:
                    valid_actions = int(action_mask.sum())
                    if valid_actions > 0:
                        uniform_probs = [1.0 / valid_actions if allowed else 0.0 for allowed in action_mask]
                        self.last_uncertainty = self.trainer.dagger.entropy_from_probs(uniform_probs)
                    else:
                        self.last_uncertainty = 0.0
                if self.ai_mode == "query" and self.trainer.should_query_human(stuck=self.last_stuck_state, uncertainty=self.last_uncertainty):
                    self._request_dagger_action(obs)
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


def _apply_curriculum_stage(env: GridEnv, trainer: AtlasTrainer) -> None:
    stage = trainer.current_curriculum_stage()
    env.preset = stage.preset
    env.reset(seed=env.seed_value)
    env.set_mode(stage.mode, stage.mode_params)



def train_headless(config_path: Path | None, steps: int) -> None:
    config = load_config(config_path)
    env = GridEnv(config)
    trainer = AtlasTrainer(config, Path("checkpoints"))
    trainer.load(env)
    db = DBLogger(Path("atlas.db"))

    chunk_steps = max(2000, min(5000, steps // 4 if steps > 0 else 2000))
    remaining = steps
    transition_reason = "initial_stage"

    while remaining > 0:
        _apply_curriculum_stage(env, trainer)
        stage = trainer.current_curriculum_stage()
        db.start_episode(
            env.preset,
            env.seed_value,
            env.mode.name,
            datetime.utcnow().isoformat(),
            env.world_hash,
            curriculum_stage=stage.name,
            stage_transition_reason=transition_reason,
        )
        trainer.train_steps(min(chunk_steps, remaining))
        remaining -= min(chunk_steps, remaining)

        # Evaluate current stage and potentially advance.
        obs, _ = env.reset(seed=env.seed_value)
        env.set_mode(stage.mode, stage.mode_params)
        done = False
        episode_return = 0.0
        recurrent_state = None
        episode_start = True
        while not done:
            action, recurrent_state = trainer.predict(obs, state=recurrent_state, mask=episode_start)
            obs, reward, done, _, _ = env.step(int(action))
            episode_start = bool(done)
            episode_return += float(reward)

        transitioned, reason = trainer.record_curriculum_episode(
            mode_name=env.mode.name,
            mode_info=env.mode.info(),
            episode_return=episode_return,
        )
        transition_reason = reason or "stage_retained"
        if transitioned:
            next_stage = trainer.current_curriculum_stage()
            db.log_event(
                "curriculum_stage_transition",
                {
                    "from_stage": stage.name,
                    "to_stage": next_stage.name,
                    "reason": reason,
                },
            )

    trainer.save()


def resume_training(config_path: Path | None, steps: int) -> None:
    train_headless(config_path, steps)


def export_replay(db_path: Path, out_path: Path) -> None:
    export_steps(db_path, out_path)





def run_policy_export(config_path: Path | None, checkpoint: Path, out_dir: Path) -> None:
    config = load_config(config_path)
    env = GridEnv(config)
    manifest_path = export_policy_artifact(checkpoint, env, out_dir)
    print(f"Policy artifact exported: {manifest_path}")


def run_inference(config_path: Path | None, artifact_dir: Path, steps: int = 100) -> None:
    config = load_config(config_path)
    env = GridEnv(config)
    runtime = load_runtime_policy(artifact_dir, env)
    obs, _ = env.reset(seed=config.training.seed)
    for _ in range(steps):
        action = runtime.predict(obs, deterministic=True)
        obs, _reward, done, _truncated, _info = env.step(action)
        if done:
            runtime.mark_episode_done()
            obs, _ = env.reset(seed=config.training.seed)
    print(f"Inference run completed for {steps} steps.")

def run_eval(config_path: Path | None, checkpoint_paths: list[Path] | None = None) -> None:
    config = load_config(config_path)
    checkpoints = checkpoint_paths or [Path("checkpoints/atlas_model.zip")]
    seeds = [11, 23, 37, 49, 61]
    mode_matrix = [
        ("ExitGame", {}),
        ("CaptureTheFlag", {}),
        ("HideAndSeek", {"hide_target": (2, 2), "time_limit_steps": 120}),
    ]
    harness = DeterministicEvalHarness(config, Path("checkpoints"))
    report = harness.evaluate(checkpoints=checkpoints, seeds=seeds, mode_matrix=mode_matrix)
    out_dir = Path("reports")
    json_path = out_dir / "eval_trends.json"
    csv_path = out_dir / "eval_trends.csv"
    write_eval_trend_report(report.get("rows", []), json_path, csv_path)
    print(f"Eval report written: {json_path} and {csv_path}")


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

    eval_cmd = subparsers.add_parser("eval")
    eval_cmd.add_argument("--checkpoints", nargs="*", type=Path, default=None)

    export_policy_cmd = subparsers.add_parser("export-policy")
    export_policy_cmd.add_argument("--checkpoint", type=Path, default=Path("checkpoints/atlas_model.zip"))
    export_policy_cmd.add_argument("--out-dir", type=Path, default=Path("runtime_artifacts/latest"))

    infer_cmd = subparsers.add_parser("infer")
    infer_cmd.add_argument("--artifact-dir", type=Path, default=Path("runtime_artifacts/latest"))
    infer_cmd.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()
    if args.command == "train":
        train_headless(args.config, args.steps)
    elif args.command == "resume":
        resume_training(args.config, args.steps)
    elif args.command == "export":
        export_replay(args.db, args.out)
    elif args.command == "eval":
        run_eval(args.config, args.checkpoints)
    elif args.command == "export-policy":
        run_policy_export(args.config, args.checkpoint, args.out_dir)
    elif args.command == "infer":
        run_inference(args.config, args.artifact_dir, args.steps)
    else:
        AtlasGame(args.config).run()


if __name__ == "__main__":
    main()
