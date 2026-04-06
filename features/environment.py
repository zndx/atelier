"""Behave environment hooks — tier filtering, stack health, cleanup."""

import logging
import os

log = logging.getLogger("atelier.bdd")

# ── Tier logic ──────────────────────────────────────────────────

TIER_TAGS = {"tier-0": 0, "tier-1": 1, "tier-cai": 99}


def _tier_from_scenario(scenario):
    """Return the highest tier required by scenario + feature tags."""
    all_tags = set(scenario.tags) | set(scenario.feature.tags)
    max_tier = 0
    for tag in all_tags:
        if tag in TIER_TAGS:
            max_tier = max(max_tier, TIER_TAGS[tag])
    return max_tier


def _max_tier():
    """Max tier allowed by ATELIER_BDD_TIER env var (default: 0)."""
    raw = os.environ.get("ATELIER_BDD_TIER", "0")
    if raw == "cai":
        return 99
    return int(raw)


# ── Stack health (cached, one-time) ────────────────────────────


def _ensure_stack_healthy(context):
    """Verify devenv services are reachable. Called once per session."""
    if getattr(context, "_stack_verified", False):
        return
    from atelier.config import load_config
    cfg = load_config()
    _wait_for("PostgreSQL", lambda: _check_pg(cfg.db_url))
    _wait_for("Qdrant",
              lambda: _check_qdrant(cfg.qdrant_host, cfg.qdrant_http_port))
    context._stack_verified = True
    log.info("Stack health verified")


def _wait_for(description, check_fn, timeout=60, interval=3):
    """Retry check_fn until truthy or timeout."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check_fn():
                return
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"{description} not healthy after {timeout}s")


def _check_pg(url):
    from sqlalchemy import create_engine, text
    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    engine.dispose()
    return True


def _check_qdrant(host, port):
    import urllib.request
    urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=5)
    return True


# ── Hooks ───────────────────────────────────────────────────────


def before_all(context):
    from pathlib import Path
    context.project_root = Path(__file__).resolve().parent.parent
    logging.basicConfig(level=logging.INFO)


def before_scenario(context, scenario):
    tier = _tier_from_scenario(scenario)
    allowed = _max_tier()
    if tier > allowed:
        scenario.skip(f"Requires tier-{tier}, max allowed is {allowed}")
        return
    if tier >= 1:
        _ensure_stack_healthy(context)


def after_scenario(context, scenario):
    for path in getattr(context, "_temp_files", []):
        try:
            os.unlink(path)
        except OSError:
            pass
