# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- **Isaac Lab 2.x / Isaac Sim 5.x / Python 3.11** (NVIDIA's current robotics-sim stack — successor to the deprecated Isaac Gym Preview).
- **UV** for managing the foos package via `pyproject.toml` (hatchling build backend, `src/` layout).
- Isaac Sim and Isaac Lab themselves are installed out-of-band per their official installer; they bring their own Python 3.11 + torch 2.7+cu128 wheels. This repo declares no third-party runtime deps in `pyproject.toml` — anything else is environment-provided.

## Project status

WIP. The first runnable iteration loads the URDF as an articulation, spawns a primitive-sphere ball, and exposes a viewer runner (`scripts/play_foos.py`). Reward and observations are stubbed. No RL training loop yet.

## Setup and verification

The full bootstrap is documented in `README.md`. Quick reference for working inside the project venv (`./.venv`):

```bash
# `VIRTUAL_ENV` must point at this repo's .venv when invoking uv — if a global
# auto-activation sets a different VIRTUAL_ENV, prefix uv calls explicitly.
source .venv/bin/activate

OMNI_KIT_ACCEPT_EULA=YES python scripts/play_foos.py --num_envs 1 --mode sine
```

First `play_foos.py` launch spends ~30–60 s converting the URDF to a cached USD via Isaac Lab's `UrdfFileCfg`. Subsequent launches are fast.

`OMNI_KIT_ACCEPT_EULA=YES` (also accepts `Y` / `1`, case-insensitive) skips the Omniverse EULA prompt at first run. Set it whenever Isaac Sim is being initialized — install time AND every script invocation, unless you've otherwise persisted the acceptance.

The IsaacLab repo lives at `../IsaacLab` (a sibling of this repo). It's installed editable into our venv, so updates require `git pull` + re-running `./isaaclab.sh --install`.

## Repository layout

- `src/foos/envs/foos_env.py` — `FoosEnv(DirectRLEnv)` + `FoosEnvCfg(DirectRLEnvCfg)`. Action space 16, observation space 38 (joint pos/vel + ball pos/lin_vel). Reward is ball-speed shaping + scoring bonuses + action regularization.
- `src/foos/assets_cfg/foosball_table.py` — `FOOSBALL_TABLE_CFG` (`ArticulationCfg`) loading the URDF. Two `ImplicitActuatorCfg` groups keyed by joint-name regex (`.*_prismatic_joint`, `.*_revolute_joint`).
- `src/foos/assets_cfg/ball.py` — `BALL_CFG` (`RigidObjectCfg`) with `SphereCfg` spawn. The mesh `assets/foosball_table/meshes/ball.obj` is intentionally unused.
- `src/foos/agents/rl_games_ppo_cfg.yaml` — PPO hyperparameters consumed by `scripts/train_foos.py`. Network `[256, 128, 64]` with elu, `learning_rate=3e-4`, `horizon_length=32`, `minibatch_size=16384`. Tune `entropy_coef` if exploration is poor.
- `src/foos/__init__.py` — registers `Foos-v0` with gymnasium using *string* entry points so import doesn't pull in `isaaclab` / `pxr`.
- `scripts/play_foos.py` — standalone viewer runner. `AppLauncher` is constructed before any `foos.*`/`isaaclab.*` imports — that ordering is mandatory.
- `scripts/train_foos.py` — headless rl_games PPO trainer. Same AppLauncher ordering. Logs to `logs/rl_games/foos_direct/<timestamp>/`.
- `scripts/regenerate_meshes.py` — recolors per-team OBJ meshes. Run from repo root: `python scripts/regenerate_meshes.py`. Re-run after editing source meshes or team colors.
- `assets/foosball_table/foosball_table.urdf` — 16-DOF articulation: 4 rods/team × 2 teams × (1 prismatic + 1 revolute joint). Root link `table` is fixed. Mesh refs (`meshes/...`) resolve relative to the URDF.

## Reward + termination geometry

Current reward is **"score any goal, fast"** — pure-sparse goal bonus + time pressure. Lives in `FoosEnvCfg`:

- `rew_scale_ball_speed=0` (off — earlier dense |speed| shaping at 0.05 dominated the goal signal and produced rod-spinning policies that never scored)
- `rew_scale_ball_x_speed=0.05 * |ball.vx|` (**directional** shaping: only motion along the goal axis earns reward, so jittering the ball in place doesn't help; this exists because pure-sparse goal reward turned out to be too hard for PPO to discover within 500 epochs)
- `rew_scale_action=-5e-3 * |a|^2` (mild; magnitude cost above 1e-2 collapsed exploration to "rods idle" — the smoothness penalty below is the real anti-twitch lever, not magnitude)
- `rew_scale_action_delta=-1e-3 * |a_t - a_{t-1}|^2` (punish step-to-step thrashing — this is the actual lever against twitchy motion; magnitude alone isn't enough since +1/−1 oscillation has constant `|a|^2`)
- `rew_scale_step=-0.05` (per-step time penalty; rewards finishing the episode via goal vs. timing out)
- `rew_scale_goal=+10` (terminal, fired on either team scoring)
- `rew_scale_oob=-5` (terminal, off-side or fell-through; 2.5× the previous baseline of −2; pushing to −10 stalled training)

Reward arithmetic: a fast scorer (~30 steps, |vx|~1) ≈ `+10 − 1.5 (step) + 1.5 (x-speed) − action ≈ +9`. An idle policy times out at 1800 steps ≈ `−90`. Strong gradient toward fast scoring.

Per-team scoring counters (`score/team{1,2}_total` and `score/goals_total`) are still logged so you can verify the policy isn't always scoring in the same net. We'll reintroduce per-side asymmetric reward once we add self-play and each side has its own policy.

Goal/OOB classification in `_get_dones`:

- `bx < -ball_x_limit` ⇒ team 1 scored (ball past team 2's goalie at x=-0.525)
- `bx > +ball_x_limit` ⇒ team 2 scored (ball past team 1's goalie at x=+0.525)
- `|by| > ball_y_limit` or `bz < ball_z_min` ⇒ out-of-play

Per-step termination flags are stashed on `self._goal_team*` / `self._oob` in `_get_dones` and consumed by `_get_rewards` later in the same `step()` call (DirectRLEnv runs dones before rewards). Cumulative scores live in `self._score_team*` and surface via `extras["log"]` for tensorboard.

When swapping in self-play, the pieces to replace are: (1) action-space split + opponent-snapshot policy, (2) reward sign-flip from per-team perspective, (3) symmetric obs/action mirroring along the X axis. The reward weights and goal classifier should stay as-is.

## Key conventions

- **Joint naming**: `<rod>_team_{1,2}_{prismatic,revolute}_joint` (e.g. `goalie_team_1_prismatic_joint`, `bar5_team_2_revolute_joint`). The actuator regexes in `foosball_table.py` and the per-joint scaling in `FoosEnv.__init__` both depend on the `_prismatic_joint` / `_revolute_joint` suffixes — preserve them when adding joints.
- **No digit-prefixed link or joint names**: the URDF importer maps every digit-prefixed name (`2_bar`, `3_bar`, `5_bar`) to the same sanitized USD path, causing prim collisions and a hard `Used null prim` failure. Use `bar2`, `bar3`, `bar5` instead. Mesh OBJ filenames (`2-bar-centered_team_1.obj`, etc.) are fine — those are filesystem paths, not USD paths.
- **Team colors**: Team 1 is red (`Kd 0.8 0.2 0.2`), Team 2 is blue (`Kd 0.2 0.2 0.8`). Change in `scripts/regenerate_meshes.py` and re-run.
- **`merge_fixed_joints=False`** on the URDF spawn is **load-bearing**: the URDF chains the prismatic + revolute joints through `empty_*` shell links; merging would collapse the chain.

## URDF importer gotchas (Isaac Sim 5.1)

The Isaac Sim URDF→USD converter is fragile. Things we hit and how we fixed them:

- **Digit-prefixed names** → see "Key conventions" above.
- **Revolute joints need explicit `<limit>` and `<origin>`** even for unbounded rotation, otherwise the importer generates an empty USD path internally and dies with `Coding Error: Path must be an absolute path: <>` followed by `RuntimeError: Used null prim`.
- **`UrdfFileCfg.JointDriveCfg.gains.stiffness`** is a *required* configclass field even when runtime `ImplicitActuatorCfg` overrides it. We pass dict-keyed regex values matching the same joint groups as the actuators.
- **Inertia `iyy="0.0"` is invalid**. PhysX rejects degenerate inertia tensors. The original URDF rods had `iyy=0` (rod's spin axis); we patched to `iyy=0.0001` as a placeholder. Real per-rod inertia is a TODO.
- **Self-closing `<link name="X"/>` placeholders** generate phantom `/visuals/X` USD references that don't resolve. Empty intermediate links between joints need at least an `<inertial>` element with `<origin>`. We give them inertial + tiny invisible box visuals/collisions.
- **Rod-bar `.obj` meshes don't import in jointed contexts.** `1-bar-centered_team_*.obj` etc. trigger the same `Used null prim` C++ exception when used as a link's visual or collision *if that link is part of any joint chain*. The same OBJ in a single-link URDF, and `table.obj` in any context, both work. We've confirmed via standalone `convert_urdf.py` and full retry-from-scratch that this is reproducible. As a workaround the URDF currently uses `<cylinder length="1.08" radius="0.008"/>` primitives for rod visuals/collisions and keeps `table.obj` for the table itself. Restoring rod meshes is a follow-up — likely needs filing an Isaac Sim bug or pre-converting the rod OBJs to USD reference assets.
- **Mesh filename typos fail silently.** A bad `<mesh filename="..."/>` path doesn't error — the link just imports geometry-less. The original URDF had `1-bar-centered.obj_team_1` (extension malformed); fixed.

## Tunable placeholders

Most physical numbers are first-pass guesses marked `TUNABLE`:
- Actuator stiffness/damping/limits in `foosball_table.py`.
- Ball radius/mass/friction/restitution in `ball.py`.
- Action scales (`action_scale_prismatic`, `action_scale_revolute`) in `FoosEnvCfg`.
- The URDF inertias themselves (`mass=0.7638`, off-diagonal zeros on every rod) are also suspect — revisit once the viewer shows real rod behavior.

## Known follow-ups (not yet addressed)

- `meshes/table.obj` may not include side walls / playing-surface floor. If the ball clips through or rolls off in `--mode zero`, add primitive `CuboidCfg` collider walls.
- URDF importer's default convex-hull collision approximation may swallow the ball or stick on player figures. If so, switch `UrdfConverterCfg.collider_type` to convex decomposition.
- No goals or scoring yet.
- Self-collision is off (`enabled_self_collisions=False`); ball-to-rod still works because the ball is a separate rigid body.
