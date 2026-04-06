"""Configuration loading: HOCON + env vars + CLI overrides.

HOCON (config/base.conf) is the single source of truth for all application
config. No module should access os.environ directly for configuration
values. Environment variables are captured by HOCON via ${?VAR} substitution.

Precedence (highest wins):
  1. CLI arguments (passed as overrides dict)
  2. Environment variables (picked up by HOCON ${?VAR} substitution)
  3. config/base.conf defaults

Usage::

    # Load with defaults only
    cfg = load_config()

    # Load with CLI overrides
    cfg = load_config(overrides={"agent_model": "claude-opus-4-6"})

    # Materialize and validate resolved config
    materialize_config(cfg, "build/config/atelier.env")
    errors = validate_materialized_config("build/config/atelier.env")

Preflight:
    Run ``just resolve-config`` to materialize config, then
    ``just preflight`` to validate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONF = _PROJECT_ROOT / "config" / "base.conf"


# ── HOCON path → (field_name, type) mapping ──────────────────────

_HOCON_MAP: dict[str, tuple[str, type]] = {
    "grpc.host": ("grpc_host", str),
    "grpc.port": ("grpc_port", int),
    "grpc.max_workers": ("grpc_max_workers", int),
    "gateway.host": ("gateway_host", str),
    "gateway.port": ("gateway_port", int),
    "agents.api_key": ("anthropic_api_key", str),
    "agents.model": ("agent_model", str),
    "agents.permission_mode": ("agent_permission_mode", str),
    "db.url": ("db_url", str),
    "data.parquet_dir": ("parquet_dir", str),
    "cml.project_id": ("cml_project_id", str),
    "cml.domain": ("cml_domain", str),
    "cml.engine_id": ("cml_engine_id", str),
}

# Reverse: field_name → ENV var name
_FIELD_TO_ENV: dict[str, str] = {}
for _hocon_path, (_field, _) in _HOCON_MAP.items():
    _env = "ATELIER_" + _hocon_path.replace(".", "_").upper()
    # Special cases: preserve standard env var names
    if _field == "anthropic_api_key":
        _env = "ANTHROPIC_API_KEY"
    elif _field.startswith("cml_"):
        _env = "CDSW_" + _field[4:].upper()
    elif _field == "gateway_port":
        _env = "ATELIER_GATEWAY_PORT"
    _FIELD_TO_ENV[_field] = _env


@dataclass
class AtelierConfig:
    """Resolved application configuration."""

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 10

    # HTTP Gateway
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8090

    # Claude Agent SDK
    anthropic_api_key: str | None = None
    agent_model: str = "claude-sonnet-4-5-20250929"
    agent_permission_mode: str = "dontAsk"

    # Database
    db_url: str = "sqlite+pysqlite:///.app/state.db"

    # Data
    parquet_dir: str = "build/data"

    # CML
    cml_project_id: str | None = None
    cml_domain: str | None = None
    cml_engine_id: str | None = None

    @property
    def is_cml(self) -> bool:
        """True when running inside Cloudera ML."""
        return self.cml_project_id is not None


# ── HOCON loading ────────────────────────────────────────────────


def _coerce(val: Any, target_type: type) -> Any:
    """Coerce a HOCON value to the target Python type."""
    if val is None:
        return None
    if target_type is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
    if target_type is int:
        return int(val)
    if target_type is float:
        return float(val)
    return str(val)


def _hocon_to_dict(conf) -> dict[str, Any]:
    """Extract values from a pyhocon ConfigTree using the mapping table."""
    result: dict[str, Any] = {}
    for hocon_path, (field_name, field_type) in _HOCON_MAP.items():
        try:
            val = conf.get(hocon_path)
        except Exception:
            continue
        if val is None:
            continue
        result[field_name] = _coerce(val, field_type)
    return result


def load_config(
    conf_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AtelierConfig:
    """Load configuration with three-layer precedence.

    1. Start from config/base.conf defaults (with env var substitution)
    2. Apply explicit CLI overrides

    Args:
        conf_path: Path to HOCON config file. Defaults to config/base.conf.
        overrides: CLI overrides as ``{field_name: value}`` dict. Only
            non-None values are applied (highest precedence).
    """
    from pyhocon import ConfigFactory

    conf_path = Path(conf_path) if conf_path else _DEFAULT_CONF

    if conf_path.exists():
        conf = ConfigFactory.parse_file(str(conf_path))
    else:
        conf = ConfigFactory.parse_string("")

    values = _hocon_to_dict(conf)

    # Apply CLI overrides (highest precedence)
    if overrides:
        for key, val in overrides.items():
            if val is not None:
                values[key] = val

    return AtelierConfig(**values)


# ── Materialization ──────────────────────────────────────────────


def materialize_config(
    cfg: AtelierConfig,
    output_path: str | Path,
) -> Path:
    """Write resolved config as flat key=value file for shell consumption.

    Output format (sourceable by shell / ``env -i``)::

        ANTHROPIC_API_KEY=sk-ant-...
        ATELIER_GRPC_PORT=50051
        ...

    Returns:
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if val is None:
            continue

        env_name = _FIELD_TO_ENV.get(f.name, f"ATELIER_{f.name.upper()}")

        if isinstance(val, bool):
            val_str = "true" if val else "false"
        elif isinstance(val, list):
            val_str = ",".join(str(v) for v in val)
        else:
            val_str = str(val)

        lines.append(f"{env_name}={val_str}")

    output_path.write_text("\n".join(sorted(lines)) + "\n")
    return output_path


# ── Preflight validation ─────────────────────────────────────────

# Keys that must be present and non-empty in the materialized config
REQUIRED_KEYS: list[str] = [
    "ATELIER_GRPC_PORT",
    "ATELIER_GATEWAY_PORT",
]

_MATERIALIZED_PATH = _PROJECT_ROOT / "build" / "config" / "atelier.env"


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


def validate_materialized_config(
    path: str | Path | None = None,
) -> list[str]:
    """Validate the materialized config file for completeness.

    Returns a list of error strings. Empty list means valid.

    Args:
        path: Path to the materialized env file. Defaults to
            build/config/atelier.env.
    """
    path = Path(path) if path else _MATERIALIZED_PATH
    errors: list[str] = []

    if not path.exists():
        errors.append(
            f"{path} does not exist. Run 'just resolve-config' first."
        )
        return errors

    env = _parse_env_file(path)

    for key in REQUIRED_KEYS:
        if key not in env or not env[key]:
            errors.append(f"Missing required config key: {key}")

    return errors
