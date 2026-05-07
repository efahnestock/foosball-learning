"""Standalone viewer runner for the foosball Isaac Lab env.

Boots Isaac Sim with a viewer, instantiates the FoosEnv, and steps the sim
forever applying actions chosen by --mode. No RL training, no learning loop —
this is the verification path that proves the table loads and the ball
interacts with the rods.

Run from the repo root, inside the Isaac Lab venv:

    python scripts/play_foos.py --num_envs 1 --mode sine
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# AppLauncher must be constructed before any isaaclab.* / foos.* imports that
# pull in omni.* — those modules expect simulation_app to already exist.
parser = argparse.ArgumentParser(description="Foosball Isaac Lab viewer runner")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument(
    "--mode",
    choices=["random", "sine", "zero"],
    default="sine",
    help="Action source: random uniform, sinusoidal, or zero (rods at rest)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports ------------------------------------------------
import math

import torch

from foos.envs.foos_env import FoosEnv, FoosEnvCfg


def main() -> None:
    print(">>> [play_foos] building cfg", flush=True)
    cfg = FoosEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    print(">>> [play_foos] instantiating FoosEnv", flush=True)
    env = FoosEnv(cfg=cfg)
    print(">>> [play_foos] env.reset()", flush=True)
    env.reset()
    print(f">>> [play_foos] simulation_app.is_running()={simulation_app.is_running()}", flush=True)

    device = env.device
    n_actions = env.cfg.action_space
    num_envs = env.num_envs

    # Per-joint frequencies for sine mode: prismatic slower, revolute faster.
    joint_names = env.table.joint_names
    freqs = torch.tensor(
        [0.6 if name.endswith("_prismatic_joint") else 1.5 for name in joint_names],
        device=device,
    )
    phases = torch.linspace(0.0, math.pi, n_actions, device=device)

    step = 0
    sim_dt = env.cfg.sim.dt * env.cfg.decimation

    while simulation_app.is_running():
        if args_cli.mode == "zero":
            actions = torch.zeros(num_envs, n_actions, device=device)
        elif args_cli.mode == "random":
            actions = torch.empty(num_envs, n_actions, device=device).uniform_(-1.0, 1.0)
        else:  # sine
            t = step * sim_dt
            wave = torch.sin(2.0 * math.pi * freqs * t + phases)
            actions = wave.unsqueeze(0).expand(num_envs, -1).contiguous()

        env.step(actions)
        step += 1

    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        print(">>> [play_foos] closing simulation_app", flush=True)
        simulation_app.close()
