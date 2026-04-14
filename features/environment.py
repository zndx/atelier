"""Behave environment hooks — tier filtering, stack health, cleanup."""

import logging
import os
import subprocess

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


# ── Stack lifecycle ────────────────────────────────────────────


def _stack_is_reachable():
    """Quick probe: is the gateway responding?"""
    import urllib.request
    from atelier.config import load_config
    cfg = load_config()
    try:
        urllib.request.urlopen(
            f"http://localhost:{cfg.gateway_port}/api/health", timeout=3,
        )
        return True
    except Exception:
        return False


def _start_devenv_stack(project_root):
    """Bring down any existing devenv processes and start fresh.

    Runs ``devenv processes down`` then ``devenv up -d`` (detached mode).
    The caller should use ``_wait_for`` on health probes to know when
    the stack is ready.
    """
    log.info("Stack not reachable — starting devenv stack")
    subprocess.run(
        ["devenv", "processes", "down"],
        cwd=str(project_root),
        capture_output=True,
        timeout=30,
    )
    log.info("Previous devenv processes stopped")

    # Start in detached mode — devenv manages the process lifecycle.
    subprocess.run(
        ["devenv", "up", "-d"],
        cwd=str(project_root),
        capture_output=True,
        timeout=30,
    )
    log.info("devenv up -d launched")


# ── Stack health (cached, one-time) ────────────────────────────


def _ensure_stack_healthy(context):
    """Verify devenv services are reachable; start them if needed.

    Called once per session on the first tier-1 scenario. If the stack
    is not reachable, automatically runs ``devenv processes down`` then
    ``devenv up`` and waits for health probes.
    """
    if getattr(context, "_stack_verified", False):
        return

    if not _stack_is_reachable():
        _start_devenv_stack(context.project_root)

    from atelier.config import load_config
    cfg = load_config()
    _wait_for("PostgreSQL", lambda: _check_pg(cfg.db_url), timeout=90)
    _wait_for("Qdrant",
              lambda: _check_qdrant(cfg.qdrant_host, cfg.qdrant_http_port),
              timeout=90)
    _wait_for("Gateway",
              lambda: _check_gateway(cfg.gateway_port),
              timeout=90)
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


def _check_gateway(port):
    import json
    import urllib.request
    resp = urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=5)
    body = json.loads(resp.read().decode())
    return body.get("ok", False)


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
