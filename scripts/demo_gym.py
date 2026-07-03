"""Minimal Gymnasium demo for the Foos-v0 environment.

Verifies the env conforms to the Gymnasium API and shows exactly what
observation / action tensors look like. No RL, no viewer, no video —
just `env.reset()` and a random-action `env.step()` loop.

Run from the repo root, inside the Isaac Lab venv::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/demo_gym.py --num_envs 4 --steps 100

Expected output on a working setup::

    action space:      Box(-inf, inf, (16,), float32)
    observation space: {'policy': Box(-inf, inf, (38,), float32)}
    obs['policy']: shape=(4, 38), dtype=torch.float32, device=cuda:0
    step   0  reward mean=-0.100  terminated=0/4  truncated=0/4
    ...

For a live viewer instead, use ``scripts/play_foos.py``; for a trained
policy see ``scripts/train_foos.py`` and ``scripts/replay_foos.py``.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# AppLauncher must be constructed before any `isaaclab.*` / `foos.envs.*`
# import that transitively pulls in `pxr` (USD Python bindings). Those
# modules assume `SimulationApp` already exists.
parser = argparse.ArgumentParser(description="Minimal Gymnasium demo for Foos-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel envs")
parser.add_argument("--steps", type=int, default=100, help="How many env-steps to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless by default — this script isn't a viewer.
if not args_cli.headless and not args_cli.livestream:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports ------------------------------------------------
import gymnasium as gym
import torch

import foos  # noqa: F401 — side-effect: registers "Foos-v0" with gymnasium
from foos.envs.foos_env import FoosEnvCfg


def main() -> None:
    cfg = FoosEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Foos-v0", cfg=cfg)

    # Introspect the standard gym attributes so a reader sees exactly what
    # the interface looks like. flush=True on every print: Isaac Sim's
    # simulation_app.close() hard-exits the process, so buffered stdout
    # (when output is redirected to a file) would otherwise be lost.
    print(f"action space:      {env.action_space}", flush=True)
    print(f"observation space: {env.observation_space}", flush=True)

    obs_dict, info = env.reset()
    # Isaac Lab wraps observations in a dict keyed by policy name; that
    # leaves room for privileged critic obs / multi-agent obs later. Plain
    # single-agent callers just read obs_dict["policy"].
    obs = obs_dict["policy"]
    print(
        f"obs['policy']: shape={tuple(obs.shape)}, "
        f"dtype={obs.dtype}, device={obs.device}",
        flush=True,
    )

    device = obs.device
    n_actions = env.unwrapped.cfg.action_space
    for step in range(args_cli.steps):
        # Actions are torch tensors on GPU, values in [-1, 1] (env applies
        # per-joint scaling internally). For a numpy-first library like
        # SB3 you'd .cpu().numpy() on the way in.
        action = torch.empty(args_cli.num_envs, n_actions, device=device).uniform_(-1, 1)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        if step % 20 == 0 or step == args_cli.steps - 1:
            print(
                f"step {step:3d}  reward mean={reward.mean().item():+.3f}  "
                f"terminated={terminated.sum().item():d}/{args_cli.num_envs}  "
                f"truncated={truncated.sum().item():d}/{args_cli.num_envs}",
                flush=True,
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
