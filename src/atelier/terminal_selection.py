# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Gateway-lifetime active-model override for the Web Terminal Agent.

A single module-level slot tracks which catalog entry the terminal
session should route through.  The value survives for the gateway
process lifetime and resets on restart — same lifetime as
``config_overlay._overlay``.

Three consumers:

* ``terminal.py._query_sdk`` reads the active entry on every query
  so switching models is immediately effective (no session restart).
* ``gateway.py`` exposes GET / POST / DELETE ``/api/terminal/models/active``.
* ``terminal.py``'s welcome banner reads the active entry so the
  operator sees which model will handle their prompt.

When no override is set, :func:`get_active` falls back to the catalog
entry whose ``model_ref`` matches ``cfg.agent_model`` — the deploy
default stays faithfully represented.
"""

from __future__ import annotations

import logging
import threading

from atelier.terminal_catalog import (
    ModelEntry, available_models, derive_from_agent_model, find_by_id,
)

log = logging.getLogger(__name__)


_lock = threading.Lock()
_active_id: str | None = None


def get_active(cfg) -> ModelEntry | None:
    """Return the currently active ``ModelEntry``, or ``None``.

    Resolution order:

    1. Explicit override set via :func:`set_active` (checked against
       the current catalog so the id still exists).
    2. Catalog entry whose ``model_ref`` matches ``cfg.agent_model``.
    3. ``None`` when ``cfg.agent_model`` is not in the catalog — the
       caller should keep using ``cfg.agent_model`` directly.
    """
    with _lock:
        override = _active_id
    if override:
        entry = find_by_id(override, cfg)
        if entry is not None:
            return entry
        # Override id no longer in catalog — clear it so we don't
        # keep returning a stale pointer.
        clear_active()
    return derive_from_agent_model(cfg)


def set_active(entry_id: str, cfg) -> ModelEntry:
    """Set the override.  Validates the id against the catalog and
    rejects entries that aren't available under the current creds.

    Raises :class:`ValueError` on unknown or unavailable ids.
    """
    entry = find_by_id(entry_id, cfg)
    if entry is None:
        known = sorted(m.id for m in available_models(cfg))
        raise ValueError(
            f"unknown terminal model id {entry_id!r}; known: {known}"
        )
    if not entry.available:
        raise ValueError(
            f"terminal model {entry_id!r} is not available: "
            f"{entry.unavailable_reason}"
        )
    with _lock:
        global _active_id
        _active_id = entry.id
    log.info("terminal active model -> %s (%s)", entry.id, entry.model_ref)
    return entry


def clear_active() -> None:
    """Drop the override so :func:`get_active` falls back to cfg.agent_model."""
    global _active_id
    with _lock:
        _active_id = None


def has_override() -> bool:
    """True when an explicit override is in force (vs deploy default)."""
    with _lock:
        return _active_id is not None


def active_model_ref(cfg) -> str:
    """Convenience for ``terminal.py``: return the ``model_ref`` the SDK
    should receive, falling back to ``cfg.agent_model`` when no
    catalog match exists."""
    entry = get_active(cfg)
    if entry is not None and entry.model_ref:
        return entry.model_ref
    return getattr(cfg, "agent_model", "") or ""
