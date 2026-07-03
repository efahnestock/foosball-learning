"""Run a head-to-head match between two challenge agents.

Each agent is an importable subclass of :class:`foos.challenge.FoosAgent`.
Specify them as ``module.path:ClassName`` — e.g.::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/challenge_match.py \
        --team1 foos.challenge.example_agents:RandomAgent \
        --team2 foos.challenge.example_agents:ZeroAgent \
        --num_envs 16 --episodes 50

The framework:

1. Instantiates the env (16 joints, shared).
2. Calls each agent's :meth:`~foos.challenge.FoosAgent.act` per step
   with the full 38-d observation; each agent returns 8 actions for
   their team's rods.
3. Routes those into the env's 16-dim action vector at the correct
   joint indices, then steps physics.
4. Tallies goals scored by each team (using the env's built-in
   ``_goal_team1`` / ``_goal_team2`` flags — same classifier the
   single-agent training uses) and prints a summary.

You "win" by scoring more goals than your opponent across all envs and
episodes. Ball spawns are randomized each reset.
"""

from __future__ import annotations

import argparse
import importlib
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Head-to-head match between two foosball challenge agents"
)
parser.add_argument(
    "--team1",
    type=str,
    required=True,
    help="Agent class for team 1 as 'module.path:ClassName' "
         "(e.g. 'foos.challenge.example_agents:RandomAgent').",
)
parser.add_argument(
    "--team2",
    type=str,
    required=True,
    help="Agent class for team 2 (same format as --team1).",
)
parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs")
parser.add_argument(
    "--episodes",
    type=int,
    default=50,
    help="Total episodes to play per env before summarizing (an episode "
         "is a reset-to-terminate span). Match ends after any env reaches "
         "this count.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=60_000,
    help="Safety cap on step count in case episodes don't terminate.",
)
parser.add_argument("--seed", type=int, default=0, help="RNG seed")
parser.add_argument(
    "--serve_speed",
    type=float,
    default=0.4,
    help="Neutral serve: ball spawns at center (x=0) with a random planar "
         "velocity of up to this magnitude (m/s), so neither team gets a "
         "positional edge and the ball is 'in play' immediately. Set 0 for "
         "a dead-center stationary serve.",
)
parser.add_argument("--sim_device", type=str, default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless by default — this is a match runner, not a viewer.
if not args_cli.headless and not args_cli.livestream:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports ------------------------------------------------
import gymnasium as gym
import torch

import foos  # noqa: F401 — registers Foos-v0
from foos.challenge import FoosAgent
from foos.envs.foos_env import FoosEnvCfg


def _load_agent(spec: str, *, team: int, num_envs: int, device: torch.device,
                team_joint_slots: dict[str, int]) -> FoosAgent:
    """Parse 'module.path:ClassName' and instantiate the agent."""
    if ":" not in spec:
        raise SystemExit(
            f"--team{team} spec must be 'module.path:ClassName', got {spec!r}"
        )
    mod_path, cls_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    if not issubclass(cls, FoosAgent):
        raise SystemExit(
            f"{spec} must subclass foos.challenge.FoosAgent (got {cls.__mro__})."
        )
    return cls(
        team=team,
        num_envs=num_envs,
        device=device,
        team_joint_slots=team_joint_slots,
    )


def _team_joint_indices(joint_names: list[str], team: int) -> torch.Tensor:
    """Env joint indices for the given team's 8 joints, in the natural
    per-team ordering (goalie/bar2/bar5/bar3 × prismatic/revolute)."""
    marker = f"_team_{team}_"
    idx = [i for i, n in enumerate(joint_names) if marker in n]
    if len(idx) != 8:
        raise RuntimeError(
            f"Expected 8 joints for team {team}, found {len(idx)}: "
            f"{[joint_names[i] for i in idx]}"
        )
    return torch.tensor(idx, dtype=torch.long)


def main() -> None:
    device = torch.device(args_cli.sim_device)
    cfg = FoosEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = str(device)
    cfg.seed = args_cli.seed

    # Neutral serve for a fair match: ball at center, random-direction kick.
    # (Training uses an off-center curriculum spawn; a match wants symmetry.)
    cfg.ball_spawn_x_near = 0.0
    cfg.ball_spawn_x_far = 0.0
    cfg.ball_spawn_vx_toward_goal = 0.0
    cfg.ball_spawn_random_vmax = args_cli.serve_speed
    # Disable the training curriculum so the serve config above is used as-is.
    cfg.curriculum_decay_steps = 10**12

    env = gym.make("Foos-v0", cfg=cfg)
    inner = env.unwrapped

    # Find each team's joint indices + build the {joint_name: slot} map
    # that agents receive. Slot ordering here IS the agent-side action
    # index ordering — agents just fill 8 floats in this order.
    joint_names = inner.table.joint_names
    team1_idx = _team_joint_indices(joint_names, team=1).to(device)
    team2_idx = _team_joint_indices(joint_names, team=2).to(device)
    team1_slots = {joint_names[i.item()]: slot for slot, i in enumerate(team1_idx)}
    team2_slots = {joint_names[i.item()]: slot for slot, i in enumerate(team2_idx)}

    print(f"[INFO] team 1 joints (agent slot -> env joint):")
    for name, slot in sorted(team1_slots.items(), key=lambda kv: kv[1]):
        print(f"       [{slot}] {name}")
    print(f"[INFO] team 2 joints (agent slot -> env joint):")
    for name, slot in sorted(team2_slots.items(), key=lambda kv: kv[1]):
        print(f"       [{slot}] {name}")

    agent1 = _load_agent(
        args_cli.team1, team=1, num_envs=args_cli.num_envs, device=device,
        team_joint_slots=team1_slots,
    )
    agent2 = _load_agent(
        args_cli.team2, team=2, num_envs=args_cli.num_envs, device=device,
        team_joint_slots=team2_slots,
    )
    print(f"[INFO] team 1: {args_cli.team1} ({type(agent1).__name__})")
    print(f"[INFO] team 2: {args_cli.team2} ({type(agent2).__name__})")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    agent1.reset()
    agent2.reset()

    # Track episode termination counts and per-team goal counters.
    episodes_completed = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    team1_goals = 0
    team2_goals = 0
    oobs = 0
    stucks = 0

    full_action = torch.zeros(args_cli.num_envs, 16, device=device)
    t0 = time.time()
    step = 0
    while step < args_cli.max_steps and episodes_completed.min().item() < args_cli.episodes:
        with torch.inference_mode():
            act1 = agent1.act(obs)
            act2 = agent2.act(obs)
        if act1.shape != (args_cli.num_envs, 8):
            raise RuntimeError(
                f"team 1 agent returned shape {tuple(act1.shape)}, "
                f"expected ({args_cli.num_envs}, 8)"
            )
        if act2.shape != (args_cli.num_envs, 8):
            raise RuntimeError(
                f"team 2 agent returned shape {tuple(act2.shape)}, "
                f"expected ({args_cli.num_envs}, 8)"
            )
        full_action[:, team1_idx] = act1
        full_action[:, team2_idx] = act2

        obs_dict, _rew, terminated, truncated, _info = env.step(full_action)
        obs = obs_dict["policy"]

        # Read this-step per-team goal flags directly from the env. They
        # are set in _get_dones and cleared/overwritten each step.
        team1_goals += int(inner._goal_team1.sum().item())
        team2_goals += int(inner._goal_team2.sum().item())
        oobs += int(inner._oob.sum().item())
        stucks += int(inner._stuck.sum().item())

        # An episode ends on terminate OR truncate.
        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            episodes_completed[done_ids] += 1
            agent1.reset(done_ids)
            agent2.reset(done_ids)

        step += 1

    elapsed = time.time() - t0
    total_eps = int(episodes_completed.sum().item())
    print(flush=True)
    print("=" * 64, flush=True)
    print(
        f"MATCH RESULT after {step} steps ({elapsed:.1f} s, "
        f"{total_eps} episodes across {args_cli.num_envs} envs):",
        flush=True,
    )
    print(f"  team 1 ({type(agent1).__name__}): {team1_goals} goals", flush=True)
    print(f"  team 2 ({type(agent2).__name__}): {team2_goals} goals", flush=True)
    print(f"  out-of-bounds resets: {oobs}", flush=True)
    print(f"  dead-ball (stuck) resets: {stucks}", flush=True)
    if team1_goals > team2_goals:
        print(f"  -> team 1 wins by {team1_goals - team2_goals}", flush=True)
    elif team2_goals > team1_goals:
        print(f"  -> team 2 wins by {team2_goals - team1_goals}", flush=True)
    else:
        print(f"  -> tie ({team1_goals}-{team2_goals})", flush=True)
    print("=" * 64, flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
