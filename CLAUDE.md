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

- `src/foos/envs/foos_env.py` — `FoosEnv(DirectRLEnv)` + `FoosEnvCfg(DirectRLEnvCfg)`. Action space 16, observation space 39 (stub: joint pos/vel + ball pos/lin_vel + phase placeholder). Stub reward and dones.
- `src/foos/assets_cfg/foosball_table.py` — `FOOSBALL_TABLE_CFG` (`ArticulationCfg`) loading the URDF. Two `ImplicitActuatorCfg` groups keyed by joint-name regex (`.*_prismatic_joint`, `.*_revolute_joint`).
- `src/foos/assets_cfg/ball.py` — `BALL_CFG` (`RigidObjectCfg`) with `SphereCfg` spawn. The mesh `assets/foosball_table/meshes/ball.obj` is intentionally unused.
- `scripts/play_foos.py` — standalone viewer runner. `AppLauncher` is constructed before any `foos.*`/`isaaclab.*` imports — that ordering is mandatory.
- `scripts/regenerate_meshes.py` — recolors per-team OBJ meshes. Run from repo root: `python scripts/regenerate_meshes.py`. Re-run after editing source meshes or team colors.
- `assets/foosball_table/foosball_table.urdf` — 16-DOF articulation: 4 rods/team × 2 teams × (1 prismatic + 1 revolute joint). Root link `table` is fixed. Mesh refs (`meshes/...`) resolve relative to the URDF.

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
