# foosball-learning

Train foosball agents in NVIDIA Isaac Lab.

## Status

Early WIP. The current iteration just stands up the simulation: a foosball table loads as an articulation, a primitive sphere ball drops onto the playing surface, and a runner script lets you apply random or sinusoidal rod actions to verify that kicking works. RL training plumbing comes next.

## Setup

This project targets **Isaac Lab 2.x / Isaac Sim 5.1 / Python 3.11**, managed by **UV**. Isaac Sim ships as a pip package; Isaac Lab is cloned from GitHub and pip-installed into the same venv.

```bash
# from the repo root
uv venv --python 3.11

# Isaac Sim 5.1 (~15 GB; first install is slow). OMNI_KIT_ACCEPT_EULA=YES skips
# the interactive EULA prompt at first run.
VIRTUAL_ENV="$PWD/.venv" OMNI_KIT_ACCEPT_EULA=YES \
  uv pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

# PyTorch matched to Isaac Sim 5.1
VIRTUAL_ENV="$PWD/.venv" \
  uv pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab — clone once, then install into our venv
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
source .venv/bin/activate
../IsaacLab/isaaclab.sh --install   # uses the active venv's pip

# Finally, the foos package itself
uv pip install -e .
```

> **Heads-up on `VIRTUAL_ENV`**: if your shell auto-activates a different venv (sets `VIRTUAL_ENV` globally), UV will use that one instead of `./.venv`. Either unset it or prefix every `uv pip` invocation with `VIRTUAL_ENV="$PWD/.venv"` as above. Activating the venv (`source .venv/bin/activate`) overrides it for the duration of the shell.

> **EULA**: Isaac Sim's first run shows an EULA prompt unless `OMNI_KIT_ACCEPT_EULA` is set to `YES`/`Y`/`1` (case-insensitive). Set it once during install, and again when running scripts.

## Verify

With the venv active:

```bash
source .venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES python scripts/play_foos.py --num_envs 1 --mode sine
```

First launch spends ~30–60 s converting the URDF to a cached USD; subsequent launches are fast. The viewer should show the foosball table on a ground plane, a yellow ball dropping onto the field, and rods sliding/spinning sinusoidally — kicks send the ball bouncing.

Other modes:
- `--mode zero` — rods at rest; ball settles and rolls. Confirms gravity and table collision.
- `--mode random` — uniform noise on all 16 joints; chaotic kicks.

## Layout

- `src/foos/envs/foos_env.py` — `FoosEnv` (`DirectRLEnv`) and `FoosEnvCfg`
- `src/foos/assets_cfg/foosball_table.py` — `ArticulationCfg` for the URDF
- `src/foos/assets_cfg/ball.py` — `RigidObjectCfg` for the primitive-sphere ball
- `scripts/play_foos.py` — standalone viewer runner
- `scripts/regenerate_meshes.py` — recolor per-team OBJ meshes (red = team 1, blue = team 2)
- `assets/foosball_table/` — URDF and OBJ meshes (consumed unchanged)
