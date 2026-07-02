"""Replay a trained Foos-v0 policy in the viewer / over WebRTC livestream / to MP4.

Run from the repo root, inside the Isaac Lab venv. Three modes:

1) Live WebRTC (interactive):
       OMNI_KIT_ACCEPT_EULA=YES python scripts/replay_foos.py \
           --livestream 2 --num_envs 1 --checkpoint <path>.pth

2) Headless MP4 recording (network-resilient):
       OMNI_KIT_ACCEPT_EULA=YES python scripts/replay_foos.py \
           --video --video_length 600 --headless \
           --num_envs 1 --checkpoint <path>.pth
   Output lands at logs/rl_games/foos_direct/<ts>/videos/play/*.mp4

3) Headless eval (just step counts, no rendering):
       OMNI_KIT_ACCEPT_EULA=YES python scripts/replay_foos.py \
           --headless --no_real_time --num_envs 1 --checkpoint <path>.pth

Slim Hydra-free clone of IsaacLab's stock rl_games player, hardcoded for
Foos-v0.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained foosball PPO checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to .pth checkpoint",
)
parser.add_argument("--seed", type=int, default=42, help="RNG seed")
parser.add_argument(
    "--no_real_time",
    action="store_true",
    help="Don't pace stepping to wall-clock; run as fast as possible. Implied by --video.",
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions from the policy distribution instead of using the "
         "deterministic mean. Useful when the deterministic policy degenerates "
         "but the training rollouts (which were stochastic) actually scored.",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Render to MP4 instead of running interactively. Forces --headless and --no_real_time.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=600,
    help="Number of env steps to record when --video is set (~10 s at 60 Hz).",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Override directory for the MP4. Defaults to logs/rl_games/foos_direct/<ts>/videos/play.",
)
parser.add_argument(
    "--video_warmup",
    type=int,
    default=60,
    help="Steps to skip before video recording starts (default 60 = 1 s @ 60 Hz). "
         "Avoids the dark/blurry initial frames.",
)
parser.add_argument(
    "--spawn_x_near", type=float, default=None,
    help="Override ball spawn x_near (for evaluating at harder distributions).",
)
parser.add_argument(
    "--spawn_x_far", type=float, default=None,
    help="Override ball spawn x_far.",
)
parser.add_argument(
    "--spawn_vx", type=float, default=None,
    help="Override ball spawn initial vx toward goal.",
)
parser.add_argument(
    "--no_curriculum", action="store_true",
    help="Force curriculum to its target distribution (progress=1) for evaluation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    # Video mode requires the renderer to actually produce frames.
    args_cli.enable_cameras = True
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports ------------------------------------------------
import gymnasium as gym
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import foos  # noqa: F401  — registers Foos-v0
from foos.envs.foos_env import FoosEnvCfg


def _load_agent_cfg() -> dict:
    yaml_path = files("foos.agents").joinpath("rl_games_ppo_cfg.yaml")
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    device = args_cli.device or "cuda:0"
    env_cfg = FoosEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    # Always print termination events when replaying — useful for confirming
    # whether goals are entering the mouth or just back-wall collisions.
    env_cfg.log_terminations = True
    if args_cli.spawn_x_near is not None:
        env_cfg.ball_spawn_x_near = args_cli.spawn_x_near
    if args_cli.spawn_x_far is not None:
        env_cfg.ball_spawn_x_far = args_cli.spawn_x_far
    if args_cli.spawn_vx is not None:
        env_cfg.ball_spawn_vx_toward_goal = args_cli.spawn_vx
    if args_cli.no_curriculum:
        # Override bootstrap range to the curriculum target so the spawn
        # is at peak difficulty regardless of common_step_counter.
        env_cfg.ball_spawn_x_near = env_cfg.curriculum_target_x_near
        env_cfg.ball_spawn_x_far = env_cfg.curriculum_target_x_far
        env_cfg.ball_spawn_vx_toward_goal = env_cfg.curriculum_target_vx
        env_cfg.curriculum_decay_steps = 1  # progress=1 immediately


    agent_cfg = _load_agent_cfg()
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device
    agent_cfg["params"]["seed"] = args_cli.seed

    rl_device = device
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Foos-v0", cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        if args_cli.video_folder is not None:
            video_folder = args_cli.video_folder
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_folder = os.path.join(
                "logs", "rl_games", "foos_direct", ts, "videos", "play"
            )
        os.makedirs(video_folder, exist_ok=True)
        print(f"[INFO] recording video to {os.path.abspath(video_folder)}")
        warmup = args_cli.video_warmup
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == warmup,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(Path(args_cli.checkpoint).resolve())

    print(f"[INFO] loading checkpoint {agent_cfg['params']['load_path']}")

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(agent_cfg["params"]["load_path"])
    agent.reset()

    dt = env.unwrapped.step_dt

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    real_time = not (args_cli.no_real_time or args_cli.video)
    timestep = 0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(
                obs,
                is_deterministic=False if args_cli.stochastic else agent.is_deterministic,
            )
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_warmup + args_cli.video_length:
                print(f"[INFO] video_length reached ({timestep} steps); exiting.")
                break

        if real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
