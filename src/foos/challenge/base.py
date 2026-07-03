"""Base class for foosball challenge submissions.

Subclass :class:`FoosAgent` and implement :meth:`FoosAgent.act`. The
challenge match runner (:mod:`scripts.challenge_match`) instantiates two
of these, one for each team, and steps the shared foosball environment.

Observations, actions, and joint layout are documented in
:class:`FoosAgent` — please read that docstring before submitting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FoosAgent(ABC):
    """Base class for challenge submissions. One agent controls one team.

    Subclasses must implement :meth:`act`. Optionally override
    :meth:`reset` if the agent has episodic state (RNN hidden state,
    per-episode plans, etc.).

    **Observation** — ``obs``, shape ``(num_envs, 38)``, dtype float32,
    on ``self.device``. Layout:

    * ``obs[..., 0:16]``   joint positions (m for prismatic, rad for revolute)
    * ``obs[..., 16:32]``  joint velocities
    * ``obs[..., 32:35]``  ball position in env-local frame (m).
      Team 1's goal is at ``x = +0.6``; team 2's is at ``x = -0.6``.
    * ``obs[..., 35:38]``  ball linear velocity (m/s, world frame).

    The 16 joints follow the URDF order, which you can pull from the env
    at construction time (see :attr:`team_joint_slots`). Team-1 joint
    names contain the substring ``_team_1_``; team-2 names contain
    ``_team_2_``. Within each team the four rods are ``goalie``, ``bar2``
    (defense), ``bar5`` (mid), ``bar3`` (attack); each rod has one
    prismatic (slide along y) and one revolute (rotate about y) joint.

    **Action** — return a tensor of shape ``(num_envs, 8)``, dtype float32,
    on ``self.device``, values in ``[-1, 1]``. This is **your team's**
    eight joint targets, in the order given by :attr:`team_joint_slots`
    (which mirrors the env's per-team joint ordering). The match runner
    routes these to the correct env joint indices. Prismatic values are
    scaled by ``±0.15 m`` internally, revolute by ``±1.5 rad``.

    **Frames of reference** — the observation is in the env's absolute
    frame. Team 1 attacks toward ``-x``, team 2 attacks toward ``+x``.
    If you want to write "attack-forward is +x" code that works
    regardless of team, apply your own mirror inside :meth:`act` for
    ``team == 1`` (negate ``obs[..., 32]`` and ``obs[..., 35]``, swap
    team-block joint indices).

    Parameters
    ----------
    team : int
        1 or 2. Fixed for the lifetime of the agent instance.
    num_envs : int
        Number of parallel envs. Actions must have this as their first
        dim on every call to :meth:`act`.
    device : torch.device
        Where obs will arrive and where actions must live. Typically a
        CUDA device — Isaac Sim runs entirely on GPU.
    team_joint_slots : dict[str, int] | None
        Mapping from your team's joint names (e.g. ``"goalie_team_1_prismatic_joint"``)
        to the *index within your 8-element action vector*. Provided by the
        match runner so you don't have to hard-code ordering.
    """

    def __init__(
        self,
        team: int,
        num_envs: int,
        device: torch.device | str,
        team_joint_slots: dict[str, int] | None = None,
    ):
        if team not in (1, 2):
            raise ValueError(f"team must be 1 or 2, got {team}")
        self.team = team
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.team_joint_slots = team_joint_slots or {}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Called on episode reset. ``env_ids`` is a 1-D long tensor of
        the env indices that just reset, or ``None`` if all envs did.
        Override if your agent keeps per-episode state."""

    @abstractmethod
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Choose actions for your team's rods.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(num_envs, 38)``. See class docstring for layout.

        Returns
        -------
        torch.Tensor
            Shape ``(num_envs, 8)``, values in ``[-1, 1]``, on
            ``self.device``. Ordered per :attr:`team_joint_slots`.
        """
