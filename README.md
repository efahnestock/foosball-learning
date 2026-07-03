# foosball-learning

Foosball RL sandbox on NVIDIA Isaac Lab. The full 8-rod / 16-DOF articulation loads from a URDF, a primitive-sphere ball rolls on a physically accurate playing field, and there are three entry points: **run the sim**, **train a policy**, and **compete head-to-head**.

## Status

Runnable. Physics is stable (field surface at `z = 0.762`, walls behind + goal-mouth caps + goalie-line collision — no ball escapes, no rod-freeze from wall collisions). Single-agent PPO training scores goals reliably (see `scripts/train_foos.py` + `scripts/replay_foos.py`). Two-team head-to-head match runner shipped (see below).

## Setup

Targets **Isaac Lab 2.x / Isaac Sim 5.1 / Python 3.11**, managed by **UV**. Isaac Sim ships as a pip package; Isaac Lab is cloned from GitHub and pip-installed into the same venv.

```bash
# from the repo root
uv venv --python 3.11

# Isaac Sim 5.1 (~15 GB; first install is slow).
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

> **`VIRTUAL_ENV`**: if your shell auto-activates a different venv, UV will use *that* one instead of `./.venv`. Either unset it or prefix every `uv pip` invocation with `VIRTUAL_ENV="$PWD/.venv"` as above.

> **EULA**: Isaac Sim shows an EULA prompt unless `OMNI_KIT_ACCEPT_EULA=YES` (or `Y`/`1`, case-insensitive). Set it once during install, and again on every script invocation.

## Quickstart

Three progressively fancier commands:

```bash
source .venv/bin/activate

# 1. Minimal gym demo — env.reset(), env.step(random), print shapes
OMNI_KIT_ACCEPT_EULA=YES python scripts/demo_gym.py --num_envs 4 --steps 100

# 2. Live viewer with sine-wave rod actions
OMNI_KIT_ACCEPT_EULA=YES python scripts/play_foos.py --num_envs 1 --mode sine

# 3. Head-to-head match between two agents
OMNI_KIT_ACCEPT_EULA=YES python scripts/challenge_match.py \
    --team1 foos.challenge.example_agents:RandomAgent \
    --team2 foos.challenge.example_agents:ZeroAgent \
    --num_envs 16 --episodes 50
```

First launch of any script spends ~30–60 s converting the URDF to a cached USD; subsequent launches are fast.

## Interface

`FoosEnv` is a standard `gymnasium.Env` (subclass of Isaac Lab's `DirectRLEnv`). Registered as `"Foos-v0"` in `src/foos/__init__.py`, so:

```python
import foos                              # side-effect: registers Foos-v0
import gymnasium as gym
from foos.envs.foos_env import FoosEnvCfg

env = gym.make("Foos-v0", cfg=FoosEnvCfg())
obs_dict, info = env.reset()
obs_dict, reward, terminated, truncated, info = env.step(actions)
```

Three things worth knowing before you plug it into a trainer:

**1. AppLauncher must run before any `isaaclab.*` / `foos.envs.*` import.** Those modules pull in `pxr` (USD Python bindings) which only resolves after Isaac Sim's `SimulationApp` exists. The convention is to build `AppLauncher(args)` at the top of your script, then do the heavy imports.

**2. Observations are a dict, not a bare tensor.** `env.reset()` and `env.step()` return `obs_dict = {"policy": tensor}`, giving room for privileged critic obs / multi-agent later. Single-agent callers just take `obs_dict["policy"]`.

**3. Everything is torch tensors on the GPU.** Shapes:

| Field | Shape | Notes |
|---|---|---|
| `obs["policy"]` | `(num_envs, 38)` | float32, on `env.unwrapped.device` |
| `obs["policy"][..., 0:16]` | | joint positions (m for prismatic, rad for revolute) |
| `obs["policy"][..., 16:32]` | | joint velocities |
| `obs["policy"][..., 32:35]` | | ball position (env-local, m). Team-1 goal at `x=+0.6`, team-2 at `x=-0.6` |
| `obs["policy"][..., 35:38]` | | ball linear velocity (m/s, world frame) |
| `action` | `(num_envs, 16)` | float32, values in `[-1, 1]`. Scaled internally: `±0.15 m` prismatic, `±1.5 rad` revolute |
| `reward` | `(num_envs,)` | float32 |
| `terminated`, `truncated` | `(num_envs,)` | bool |

If you're wiring this into a numpy-first library (Stable-Baselines3 etc.), convert with `.cpu().numpy()` on the way in and `torch.as_tensor(..., device=env.unwrapped.device)` on the way back.

Joint order follows the URDF — introspect with `env.unwrapped.table.joint_names`. Team 1's joints contain `_team_1_`; team 2's contain `_team_2_`. Four rods per team (`goalie`, `bar2`, `bar5`, `bar3`), each with one prismatic + one revolute joint = 8 per team.

## Train

Single-agent PPO via rl_games. One policy controls all 16 joints; reward = `+100` for scoring in either net, `-10` for out-of-bounds, `+0.5 * v_toward_nearest_goal` per step, `-0.01` per step (time pressure).

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_foos.py \
    --num_envs 4096 --max_iterations 1500 --headless
```

Logs and checkpoints land in `logs/rl_games/foos_direct/<timestamp>/`. Point TensorBoard at that path to watch `rewards/iter`, `Episode/score/team1_total`, `Episode/score/team2_total`, etc.

Watch a trained checkpoint (optionally with `--video` for MP4 output):

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/replay_foos.py \
    --num_envs 1 --checkpoint logs/rl_games/foos_direct/<ts>/nn/foos_direct.pth \
    --video --video_length 1200 --video_warmup 60
```

`replay_foos.py` also supports `--spawn_x_near/_far/_vx` to override the spawn distribution at eval time.

### Start condition (ball spawn)

On every episode reset (`FoosEnv._reset_idx`) the ball is repositioned and the rods are zeroed. The spawn is a **curriculum interpolation** between an easy "bootstrap" distribution and a harder "target" distribution, blended by training progress (`common_step_counter / curriculum_decay_steps`):

- **x**: `sign * Uniform(near, far)` with a random `±` sign each reset, where `near`/`far` interpolate from `ball_spawn_x_near/_far` (bootstrap) toward `curriculum_target_x_near/_far`. Random sign means the trainee sees both attacking and defending starts.
- **y**: small jitter `Uniform(-0.05, 0.05)` so the ball lands on the playing surface, not on a side wall.
- **velocity**: `sign * vx` toward the nearer goal (interpolated `ball_spawn_vx_toward_goal → curriculum_target_vx`), plus an optional random kick of up to `ball_spawn_random_vmax`.

Defaults ship with `curriculum_decay_steps = 10**12` (effectively "always bootstrap") and a mild off-center spawn (`x ∈ ±[0, 0.3]`, at rest). Lower `curriculum_decay_steps` to ramp toward the harder target during a run. Special case: when `near == far == 0` the ball spawns dead-center — this is what `challenge_match.py` uses for a fair neutral serve (with a random kick via `ball_spawn_random_vmax`).

## Challenge — head-to-head competition

Each participant submits an agent that controls **one team**. Framework runs both agents against each other in a shared env, tallies goals, declares a winner.

Write your agent by subclassing `foos.challenge.FoosAgent`:

```python
# my_agent.py
import torch
from foos.challenge import FoosAgent

class MyAgent(FoosAgent):
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape (num_envs, 38) — same layout as the Interface section
        # return (num_envs, 8) actions for YOUR TEAM's 8 joints:
        #   [goalie_prism, goalie_rev, bar2_prism, bar2_rev,
        #    bar5_prism, bar5_rev, bar3_prism, bar3_rev]
        # values in [-1, 1]; framework routes to the correct env joint indices.
        return torch.zeros(self.num_envs, 8, device=self.device)
```

Run a match:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/challenge_match.py \
    --team1 my_module:MyAgent \
    --team2 foos.challenge.example_agents:RandomAgent \
    --num_envs 32 --episodes 100
```

Both teams see the **same absolute-frame observation** (team-1 goal at `+x`, team-2 at `-x`). If you want to write "attack-forward is `+x`" code that plays either side, apply your own mirror inside `.act()` when `self.team == 1` — negate `obs[..., 32]` (ball x) and `obs[..., 35]` (ball vx), and swap the team joint blocks.

Reference agents in `foos.challenge.example_agents`:
- `ZeroAgent` — rods held at default
- `RandomAgent` — uniform actions in `[-1, 1]`
- `SineAgent` — time-varying sine (ignores obs, useful as a moving opponent)
- `ObsPrintAgent` — prints the obs each step; smoke-test tool, not competitive

Trained PPO checkpoints from `train_foos.py` can be wrapped into a `FoosAgent` too — see `scripts/replay_foos.py` for the load/inference boilerplate.

### Match mechanics

- **Serve (start condition).** Each point (and each auto-reset) places the ball at **center** (`x=0`, small random `y`) with a **random-direction planar kick** (`--serve_speed`, default `0.4 m/s`) so neither team has a positional edge and the ball is in play immediately. Rods reset to centered/upright. (This differs from *training*, which uses an off-center curriculum spawn — see "Start condition" below.)
- **Point ends** on: a goal (past a goalie line **and** inside the goal mouth), out-of-bounds (off the side, through the floor, or past a goal line but outside the mouth), a **dead ball** (planar speed under `0.05 m/s` for 3 continuous seconds — handles the ball wedging in a corner or stalling mid-field), or the **30-second cap** (`episode_length_s`). Only goals score; OOB/dead-ball/timeout are neutral resets.
- **Winner** = most goals across all envs and episodes. The runner prints goals, OOB resets, and dead-ball resets.

## Layout

- `src/foos/envs/foos_env.py` — `FoosEnv` (`DirectRLEnv`) and `FoosEnvCfg`
- `src/foos/assets_cfg/foosball_table.py` — `ArticulationCfg` for the URDF
- `src/foos/assets_cfg/ball.py` — `RigidObjectCfg` for the sphere ball
- `src/foos/agents/rl_games_ppo_cfg.yaml` — PPO hyperparameters for rl_games
- `src/foos/challenge/base.py` — `FoosAgent` base class for submissions
- `src/foos/challenge/example_agents.py` — reference agents
- `src/foos/__init__.py` — registers `Foos-v0` with gymnasium
- `scripts/demo_gym.py` — minimal `env.reset()`/`env.step()` demo
- `scripts/play_foos.py` — viewer + video runner (random/sine/zero/lift modes, spawn cycle)
- `scripts/train_foos.py` — headless PPO training via rl_games
- `scripts/replay_foos.py` — checkpoint replay + video recording
- `scripts/challenge_match.py` — head-to-head match between two agents
- `scripts/regenerate_meshes.py` — recolor per-team OBJ meshes (red = team 1, blue = team 2)
- `assets/foosball_table/` — URDF, OBJ meshes, and baked collision USDs

## Common gotchas

- **URDF importer is fragile.** Rod-figure OBJs crash on import when they're inside a jointed link chain — we ship cylinder-primitive rods and attach the real figure meshes as USD refs at runtime. See `_ROD_TO_MESH` in `foos_env.py`.
- **Digit-prefixed link/joint names hard-crash the importer** (`2_bar`, `3_bar` → prim collisions). All rod names use `bar2`, `bar3`, `bar5`.
- **Isaac Sim writes ~15 GB into the venv.** Ensure `/home` has room before the initial install.
- **The action low-pass filter (`action_smoothing_alpha`) is disabled by default** (`alpha=1.0`). Set to `0.35` for ~50 ms smoothing if you want more human-like rod motion.
