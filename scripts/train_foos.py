"""Headless PPO training for the Foos-v0 task via rl_games.

Run from the repo root, inside the Isaac Lab venv:

    OMNI_KIT_ACCEPT_EULA=YES python scripts/train_foos.py \
        --num_envs 4096 --max_iterations 1500 --headless

This is a slimmed-down version of IsaacLab's stock rl_games trainer
(``scripts/reinforcement_learning/rl_games/train.py``) hardcoded for our
single task — no Hydra, no multi-GPU, no video recording. Logs land in
``logs/rl_games/foos_direct/<timestamp>/`` and stream to TensorBoard.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime
from importlib.resources import files

from isaaclab.app import AppLauncher

# AppLauncher must be constructed before any isaaclab.* / foos.* imports that
# pull in omni.* — those modules expect SimulationApp to already exist.
parser = argparse.ArgumentParser(description="Train a foosball PPO agent via rl_games")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel envs")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO epochs cap")
parser.add_argument("--seed", type=int, default=None, help="RNG seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--sigma", type=str, default=None, help="Initial action stddev override")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless training by default — videos/streaming would slow us down.
if not args_cli.headless and not args_cli.livestream:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports ------------------------------------------------
import time

import gymnasium as gym
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# Side-effect: registers Foos-v0 with gymnasium.
import foos  # noqa: F401
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

    agent_cfg = _load_agent_cfg()
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    if seed == -1:
        seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = seed
    env_cfg.seed = seed

    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device

    if args_cli.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.checkpoint

    config_name = agent_cfg["params"]["config"]["name"]
    log_root = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    print(f"[INFO] log dir: {os.path.join(log_root, log_dir)}")

    dump_yaml(os.path.join(log_root, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root, log_dir, "params", "agent.yaml"), agent_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make("Foos-v0", cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None
    run_args = {"train": True, "play": False, "sigma": train_sigma}
    if args_cli.checkpoint is not None:
        run_args["checkpoint"] = args_cli.checkpoint

    t0 = time.time()
    runner.run(run_args)
    print(f"[INFO] training time: {time.time() - t0:.1f}s")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
