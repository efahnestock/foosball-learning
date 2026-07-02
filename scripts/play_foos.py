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
    choices=["random", "sine", "zero", "lift"],
    default="sine",
    help=(
        "Action source: random uniform, sinusoidal, zero (rods at rest), "
        "or lift (revolute joints held at +1.0 to swing player figures up "
        "and out of the playing-field plane — useful for verifying the "
        "ball can reach the goal mouth without rod obstruction)"
    ),
)
parser.add_argument(
    "--table_collision_approx",
    choices=["default", "convex_decomposition", "sdf"],
    default="default",
    help=(
        "Override the URDF importer's mesh collision approximation for the "
        "table. 'default' keeps the importer output (convex_decomposition "
        "with default params, which fails to capture the goal-mouth carveout). "
        "'convex_decomposition' re-applies aggressive decomposition params. "
        "'sdf' uses signed-distance-field collision (exact, ~2-3x slower)."
    ),
)
parser.add_argument(
    "--show_colliders",
    action="store_true",
    help="Enable PhysX collider wireframe overlay in the viewer / livestream.",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Render to MP4 instead of streaming. Forces --headless and enables cameras.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=600,
    help="Number of env steps to record when --video is set (~10 s at 60 Hz).",
)
parser.add_argument(
    "--video_warmup",
    type=int,
    default=60,
    help="Steps to skip before starting video recording (default 60 = 1 s @ 60 Hz). "
         "Avoids the black/blurry frames at the start of the rollout.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Override directory for the MP4. Defaults to logs/play/<ts>/.",
)
parser.add_argument(
    "--spawn_cycle",
    action="store_true",
    help=(
        "Demo mode: place the ball under each of 5 rod-line x-positions in "
        "rotation, every --cycle_seconds. Disables env auto-reset so the "
        "ball gets kicked around freely between cycle ticks."
    ),
)
parser.add_argument(
    "--cycle_seconds",
    type=float,
    default=5.0,
    help="Seconds between spawn-cycle resets when --spawn_cycle is set.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.show_colliders:
    import carb
    s = carb.settings.get_settings()
    s.set("/persistent/physics/visualizationDisplayColliders", True)
    s.set("/physics/visualizationDisplayColliders", True)
    print(">>> [play_foos] collider visualization enabled", flush=True)

# ---- post-AppLauncher imports ------------------------------------------------
import math
import os
from datetime import datetime

import torch
import gymnasium as gym

import foos  # noqa: F401  — registers Foos-v0
from foos.envs.foos_env import FoosEnvCfg


def main() -> None:
    print(">>> [play_foos] building cfg", flush=True)
    cfg = FoosEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.table_collision_approx = args_cli.table_collision_approx
    cfg.visualize_primitive_collision = args_cli.show_colliders
    if args_cli.spawn_cycle:
        # We manage resets manually — make the env's auto-reset bounds
        # large enough that they never fire on their own.
        cfg.episode_length_s = 1.0e6
        cfg.ball_x_limit = 1.0e6
        cfg.ball_y_limit = 1.0e6
        cfg.ball_z_min = -1.0e6
        # Disable the action low-pass filter so random actions actually
        # move the rods. With alpha=0.35 the LPF averages N(0,1) actions
        # toward zero and rods barely budge.
        cfg.action_smoothing_alpha = 1.0
    print(">>> [play_foos] instantiating FoosEnv", flush=True)
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Foos-v0", cfg=cfg, render_mode=render_mode)

    if args_cli.video:
        if args_cli.video_folder is not None:
            video_folder = args_cli.video_folder
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_folder = os.path.join("logs", "play", ts)
        os.makedirs(video_folder, exist_ok=True)
        print(f">>> [play_foos] recording video to {os.path.abspath(video_folder)}", flush=True)
        warmup = args_cli.video_warmup
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == warmup,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    print(">>> [play_foos] env.reset()", flush=True)
    env.reset()
    print(f">>> [play_foos] simulation_app.is_running()={simulation_app.is_running()}", flush=True)

    inner = env.unwrapped
    device = inner.device
    n_actions = inner.cfg.action_space
    num_envs = inner.num_envs

    # Per-joint frequencies for sine mode: prismatic slower, revolute faster.
    joint_names = inner.table.joint_names
    freqs = torch.tensor(
        [0.6 if name.endswith("_prismatic_joint") else 1.5 for name in joint_names],
        device=device,
    )
    phases = torch.linspace(0.0, math.pi, n_actions, device=device)
    # For lift mode: 0 on prismatic, +1 on revolute (player figures swung up
    # by ~86°, out of the playing-field plane).
    is_revolute = torch.tensor(
        [1.0 if name.endswith("_revolute_joint") else 0.0 for name in joint_names],
        device=device,
    )

    step = 0
    sim_dt = inner.cfg.sim.dt * inner.cfg.decimation

    # Track per-joint position range (full 16-joint articulation) so we can
    # verify rods actually move.
    n_joints = len(joint_names)
    prismatic_indices = [
        i for i, name in enumerate(joint_names) if name.endswith("_prismatic_joint")
    ]
    j_min = torch.full((n_joints,), float("inf"), device=device)
    j_max = torch.full((n_joints,), float("-inf"), device=device)
    t_min = torch.full((n_joints,), float("inf"), device=device)
    t_max = torch.full((n_joints,), float("-inf"), device=device)
    print(">>> [play_foos] joint order + scale tensor:", flush=True)
    for i, name in enumerate(joint_names):
        print(f"    [{i:2d}] {name:36s} scale={inner._joint_scale[i].item():+.4f}", flush=True)

    # Spawn-cycle setup: 5 evenly-spaced x positions across the playing
    # field, centered on rod x-coordinates. Cycled in order.
    spawn_x_list = [-0.4, -0.2, 0.0, 0.2, 0.4]
    cycle_steps = max(1, int(round(args_cli.cycle_seconds / sim_dt)))
    spawn_idx = 0

    def manual_spawn(x: float) -> None:
        n = inner.num_envs
        # Full articulation has 16 joints; trainee action space is 8.
        joint_pos = torch.zeros(n, len(joint_names), device=device)
        joint_vel = torch.zeros_like(joint_pos)
        inner.table.write_joint_state_to_sim(joint_pos, joint_vel)
        ball_state = inner.ball.data.default_root_state.clone()
        ball_state[:, :3] = inner.scene.env_origins
        ball_state[:, 0] += x
        ball_state[:, 1] = 0.0
        ball_state[:, 2] = 0.79
        ball_state[:, 7:13] = 0.0
        inner.ball.write_root_pose_to_sim(ball_state[:, :7])
        inner.ball.write_root_velocity_to_sim(ball_state[:, 7:])
        print(f">>> [play_foos] spawn cycle: x={x:+.2f}", flush=True)

    if args_cli.spawn_cycle:
        manual_spawn(spawn_x_list[spawn_idx])

    while simulation_app.is_running():
        if args_cli.mode == "zero":
            actions = torch.zeros(num_envs, n_actions, device=device)
        elif args_cli.mode == "random":
            actions = torch.empty(num_envs, n_actions, device=device).uniform_(-1.0, 1.0)
        elif args_cli.mode == "lift":
            actions = is_revolute.unsqueeze(0).expand(num_envs, -1).contiguous()
        else:  # sine
            t = step * sim_dt
            wave = torch.sin(2.0 * math.pi * freqs * t + phases)
            actions = wave.unsqueeze(0).expand(num_envs, -1).contiguous()

        env.step(actions)
        step += 1

        jp = inner.table.data.joint_pos[0]
        jt = inner._action_targets[0]
        j_min = torch.minimum(j_min, jp)
        j_max = torch.maximum(j_max, jp)
        t_min = torch.minimum(t_min, jt)
        t_max = torch.maximum(t_max, jt)

        if args_cli.spawn_cycle and step % cycle_steps == 0:
            spawn_idx = (spawn_idx + 1) % len(spawn_x_list)
            manual_spawn(spawn_x_list[spawn_idx])

        if args_cli.video and step >= args_cli.video_warmup + args_cli.video_length:
            print(f">>> [play_foos] video_length reached ({step} steps); exiting.", flush=True)
            break

    print(">>> [play_foos] per-joint observed range (env 0):", flush=True)
    for i, name in enumerate(joint_names):
        print(
            f"    {name:36s}  target=[{t_min[i].item():+.4f},{t_max[i].item():+.4f}]  "
            f"actual=[{j_min[i].item():+.4f},{j_max[i].item():+.4f}]",
            flush=True,
        )

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
