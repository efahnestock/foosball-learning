"""Foos package — Isaac Lab environment for foosball.

Note: do not eagerly import :class:`FoosEnv` here. It transitively imports
``isaaclab`` modules (e.g. ``isaaclab.utils.math.mesh``) that pull in ``pxr``,
which only resolves after the Omniverse Kit ``SimulationApp`` is created.
Import inside your runner script *after* ``AppLauncher`` has been built::

    from foos.envs.foos_env import FoosEnv, FoosEnvCfg

The :func:`gym.register` call below uses string entry points, so it is safe
to run at import time without forcing those heavy imports.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Foos-v0",
    entry_point="foos.envs.foos_env:FoosEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "foos.envs.foos_env:FoosEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
