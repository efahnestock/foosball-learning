"""Reference agent implementations for the foosball challenge.

Use these as baselines / smoke-test opponents / starting templates for
your own submission. All three ignore the observation (except
`ObsPrintAgent`); they exist to demonstrate the interface, not to play
well.

To run a match::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/challenge_match.py \
        --team1 foos.challenge.example_agents:RandomAgent \
        --team2 foos.challenge.example_agents:ZeroAgent \
        --num_envs 16 --episodes 50
"""

from __future__ import annotations

import math

import torch

from foos.challenge.base import FoosAgent


class ZeroAgent(FoosAgent):
    """Baseline: hold rods at their joint defaults (centered, upright)."""

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.num_envs, 8, device=self.device)


class RandomAgent(FoosAgent):
    """Uniform random actions in [-1, 1]. Chaotic but occasionally scores."""

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.empty(self.num_envs, 8, device=self.device).uniform_(-1.0, 1.0)


class SineAgent(FoosAgent):
    """Time-varying sine — prismatic joints slide at 0.5 Hz, revolute at 1.5 Hz.

    Ignores the observation entirely. Useful as a "kick-happy" opponent
    that produces motion without any strategy.
    """

    def __init__(self, team, num_envs, device, team_joint_slots=None):
        super().__init__(team, num_envs, device, team_joint_slots)
        self._step = 0
        # Action layout is 8-d [goalie_prism, goalie_rev, bar2_prism, bar2_rev,
        #                       bar5_prism, bar5_rev, bar3_prism, bar3_rev]
        # Even indices → prismatic, odd → revolute.
        self._freqs = torch.tensor(
            [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5], device=self.device
        )
        self._phases = torch.linspace(0.0, math.pi, 8, device=self.device)

    def reset(self, env_ids=None):
        # `_step` is a scalar shared across envs so keep it monotonic.
        pass

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self._step += 1
        t = self._step / 60.0
        wave = torch.sin(2.0 * math.pi * self._freqs * t + self._phases)
        return wave.unsqueeze(0).expand(self.num_envs, -1).contiguous()


class ObsPrintAgent(FoosAgent):
    """Prints the observation summary once per call; returns zero action.

    Useful as a smoke test to verify the framework is feeding your agent
    what you expect. Not competitive.
    """

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        ball_pos = obs[0, 32:35].tolist()
        ball_vel = obs[0, 35:38].tolist()
        print(
            f"[ObsPrintAgent team={self.team}] "
            f"ball_pos=({ball_pos[0]:+.3f},{ball_pos[1]:+.3f},{ball_pos[2]:+.3f}) "
            f"ball_vel=({ball_vel[0]:+.3f},{ball_vel[1]:+.3f},{ball_vel[2]:+.3f})",
            flush=True,
        )
        return torch.zeros(self.num_envs, 8, device=self.device)
