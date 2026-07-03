"""Two-team foosball challenge: each submission controls one team.

Users subclass :class:`FoosAgent` (see :mod:`foos.challenge.base`) and
implement ``.act(obs) -> action``. Run head-to-head matches with
``scripts/challenge_match.py``.
"""

from foos.challenge.base import FoosAgent

__all__ = ["FoosAgent"]
