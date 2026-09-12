"""One call after keys, endpoints or pins changed in the running process.

Three things have to happen, in this order, and every front end used to be
one forgotten step away from a stale router:

  1. `overrides.apply()` -- re-register custom endpoints and rebuild the
     pinned chains in the live table;
  2. `llm_provider.reset()` -- drop cached provider instances and classes, so
     a new key or a new base URL is read on the next construction;
  3. `Router.reset_all()` -- every live router re-snapshots which providers
     are configured (agent/pipeline/nodes.py's module-level one included).

Precondition: call it between turns. A graph thread mid-resolve sees only
whole-tuple swaps, so nothing tears, but a turn already handed its model a
key it would be confusing to change under it.
"""
from __future__ import annotations

from agent.router import llm_provider, overrides
from agent.router.router import Router


def reload_everything() -> list[str]:
    """Returns the problems `overrides.apply()` reported, for display."""
    problems = overrides.apply()
    llm_provider.reset()
    Router.reset_all()
    return problems
