# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Structured preflight validation for Atelier.

Validates that materialized config is complete and internally consistent
before services start. No network probes — this is config-only validation.

Each check returns a deny (blocking) or warn (advisory) status:
- deny: Service cannot start safely. Must be fixed.
- warn: Service can start but with degraded functionality.

Usage::

    from atelier.config import load_config
    from atelier.preflight import run_preflight

    result = run_preflight(load_config())
    if not result.ok:
        for d in result.denies:
            print(f"DENY: {d.name}: {d.message}")
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.config import AtelierConfig


@dataclass
class CheckResult:
    """Single preflight check outcome."""

    name: str
    status: str  # "pass", "deny", "warn"
    message: str
    remediation: str = ""


@dataclass
class PreflightResult:
    """Aggregated preflight outcome."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def denies(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "deny"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.denies


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ENV_PATH = _PROJECT_ROOT / "build" / "config" / "atelier.env"


def run_preflight(
    cfg: AtelierConfig,
    env_path: Path | None = None,
) -> PreflightResult:
    """Run all preflight checks against resolved config.

    Args:
        cfg: Resolved AtelierConfig instance.
        env_path: Path to materialized env file. Defaults to
            build/config/atelier.env.
    """
    env_path = env_path or _DEFAULT_ENV_PATH
    result = PreflightResult()

    _check_config_exists(result, env_path)
    _check_required_ports(result, env_path)
    _check_db_url_parseable(result, cfg)
    _check_qdrant_host_set(result, cfg)
    _check_credentials_configured(result, cfg)
    _check_parquet_dir_exists(result, cfg)
    _check_gpu(result)

    return result


# ── Individual checks ────────────────────────────────────────────


def _check_config_exists(result: PreflightResult, env_path: Path) -> None:
    if env_path.exists():
        result.checks.append(CheckResult(
            name="config_exists",
            status="pass",
            message=f"{env_path} exists",
        ))
    else:
        result.checks.append(CheckResult(
            name="config_exists",
            status="deny",
            message=f"{env_path} does not exist",
            remediation="Run 'just resolve-config' to materialize config.",
        ))


def _check_required_ports(result: PreflightResult, env_path: Path) -> None:
    required = ["ATELIER_GRPC_PORT", "ATELIER_GATEWAY_PORT"]
    if not env_path.exists():
        # Already reported by config_exists check
        return

    env = _parse_env_file(env_path)
    missing = [k for k in required if k not in env or not env[k]]

    if not missing:
        result.checks.append(CheckResult(
            name="required_ports",
            status="pass",
            message="gRPC and gateway ports configured",
        ))
    else:
        result.checks.append(CheckResult(
            name="required_ports",
            status="deny",
            message=f"Missing port config: {', '.join(missing)}",
            remediation="Check config/base.conf grpc.port and gateway.port settings.",
        ))


def _check_db_url_parseable(result: PreflightResult, cfg: AtelierConfig) -> None:
    url = cfg.db_url
    if "://" in url and len(url.split("://", 1)[1]) > 0:
        result.checks.append(CheckResult(
            name="db_url_parseable",
            status="pass",
            message="Database URL is parseable",
        ))
    else:
        result.checks.append(CheckResult(
            name="db_url_parseable",
            status="deny",
            message=f"Database URL is not parseable: {url}",
            remediation="Set ATELIER_DB_URL or check db.url in config/base.conf.",
        ))


def _check_qdrant_host_set(result: PreflightResult, cfg: AtelierConfig) -> None:
    if cfg.qdrant_host:
        result.checks.append(CheckResult(
            name="qdrant_host_set",
            status="pass",
            message=f"Qdrant host: {cfg.qdrant_host}:{cfg.qdrant_http_port}",
        ))
    else:
        result.checks.append(CheckResult(
            name="qdrant_host_set",
            status="deny",
            message="Qdrant host is empty",
            remediation="Set QDRANT_HOST or check qdrant.host in config/base.conf.",
        ))


def _check_credentials_configured(
    result: PreflightResult, cfg: AtelierConfig,
) -> None:
    if cfg.has_anthropic or cfg.has_bedrock:
        providers = []
        if cfg.has_anthropic:
            providers.append("Anthropic")
        if cfg.has_bedrock:
            providers.append("Bedrock")
        result.checks.append(CheckResult(
            name="credentials_configured",
            status="pass",
            message=f"LLM credentials: {', '.join(providers)}",
        ))
    else:
        result.checks.append(CheckResult(
            name="credentials_configured",
            status="warn",
            message="No LLM credentials configured",
            remediation="Set ANTHROPIC_API_KEY or AWS credentials in .env for agent features.",
        ))


def _check_parquet_dir_exists(result: PreflightResult, cfg: AtelierConfig) -> None:
    parquet_path = Path(cfg.parquet_dir)
    if not parquet_path.is_absolute():
        parquet_path = _PROJECT_ROOT / parquet_path

    if parquet_path.exists():
        result.checks.append(CheckResult(
            name="parquet_dir_exists",
            status="pass",
            message=f"Parquet directory: {cfg.parquet_dir}",
        ))
    else:
        result.checks.append(CheckResult(
            name="parquet_dir_exists",
            status="warn",
            message=f"Parquet directory not found: {cfg.parquet_dir}",
            remediation="Create the directory or update data.parquet_dir in config.",
        ))


def _check_gpu(result: PreflightResult) -> None:
    """Advisory GPU detection — never blocks, just reports availability."""
    try:
        from atelier.classify.gpu import preflight_gpu
        gpu = preflight_gpu()
        if gpu.available:
            result.checks.append(CheckResult(
                name="gpu",
                status="pass",
                message=gpu.summary(),
            ))
        elif gpu.device_count > 0:
            result.checks.append(CheckResult(
                name="gpu",
                status="warn",
                message=gpu.summary(),
                remediation="; ".join(gpu.warnings) if gpu.warnings else "",
            ))
        else:
            result.checks.append(CheckResult(
                name="gpu",
                status="warn",
                message="No NVIDIA GPUs detected — embeddings will use CPU",
            ))
    except Exception:
        result.checks.append(CheckResult(
            name="gpu",
            status="warn",
            message="GPU detection skipped (torch not installed)",
        ))


# ── Helpers ──────────────────────────────────────────────────────


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a flat key=value env file into a dict."""
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result
