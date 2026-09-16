"""Atelier Metaflow flows — platform Metaflow discovered from Signals."""

from __future__ import annotations

FLOW_REGISTRY: dict[str, type] = {}


def register_flow(name: str):
    """Decorator to register a flow for CLI discovery and YK kind."""

    def decorator(cls):
        FLOW_REGISTRY[name] = cls
        cls.yk_kind = name.replace("_", "-")
        return cls

    return decorator
