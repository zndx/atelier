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
import re
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
    "agents.aws_access_key_id": ("aws_access_key_id", str),
    "agents.aws_secret_access_key": ("aws_secret_access_key", str),
    "agents.aws_region": ("aws_region", str),
    "agents.aws_session_token": ("aws_session_token", str),
    "agents.permission_mode": ("agent_permission_mode", str),
    "agents.default_sonnet_model": ("agent_default_sonnet_model", str),
    "agents.default_haiku_model": ("agent_default_haiku_model", str),
    "agents.subagent_model": ("agent_subagent_model", str),
    "agents.disable_experimental_betas": ("agent_disable_experimental_betas", str),
    "agents.enable_tool_search": ("agent_enable_tool_search", str),
    "db.url": ("db_url", str),
    "qdrant.host": ("qdrant_host", str),
    "qdrant.http_port": ("qdrant_http_port", int),
    "qdrant.grpc_port": ("qdrant_grpc_port", int),
    "data.parquet_dir": ("parquet_dir", str),
    "cml.project_id": ("cml_project_id", str),
    "cml.domain": ("cml_domain", str),
    "cml.engine_id": ("cml_engine_id", str),
    "cml.data_connections": ("cml_data_connections", str),
    "classify.connection_name": ("classify_connection_name", str),
    "classify.database": ("classify_database", str),
    "classify.sample_size": ("classify_sample_size", int),
    "classify.tables_limit": ("classify_tables_limit", int),
    "classify.embedding_model": ("classify_embedding_model", str),
    "classify.auto_start": ("classify_auto_start", bool),
}

# Reverse: field_name → ENV var name
_FIELD_TO_ENV: dict[str, str] = {}
for _hocon_path, (_field, _) in _HOCON_MAP.items():
    _env = "ATELIER_" + _hocon_path.replace(".", "_").upper()
    # Special cases: preserve standard env var names
    if _field == "anthropic_api_key":
        _env = "ANTHROPIC_API_KEY"
    elif _field == "aws_access_key_id":
        _env = "AWS_ACCESS_KEY_ID"
    elif _field == "aws_secret_access_key":
        _env = "AWS_SECRET_ACCESS_KEY"
    elif _field == "aws_region":
        _env = "AWS_REGION"
    elif _field == "aws_session_token":
        _env = "AWS_SESSION_TOKEN"
    elif _field == "agent_default_sonnet_model":
        _env = "ANTHROPIC_DEFAULT_SONNET_MODEL"
    elif _field == "agent_default_haiku_model":
        _env = "ANTHROPIC_DEFAULT_HAIKU_MODEL"
    elif _field == "agent_subagent_model":
        _env = "CLAUDE_CODE_SUBAGENT_MODEL"
    elif _field == "agent_disable_experimental_betas":
        _env = "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"
    elif _field == "agent_enable_tool_search":
        _env = "ENABLE_TOOL_SEARCH"
    elif _field == "cml_data_connections":
        # Not a platform-provided var — this is our own knob.
        _env = "ATELIER_DATA_CONNECTIONS"
    elif _field.startswith("cml_"):
        _env = "CDSW_" + _field[4:].upper()
    elif _field == "gateway_port":
        _env = "ATELIER_GATEWAY_PORT"
    elif _field == "qdrant_host":
        _env = "QDRANT_HOST"
    elif _field == "qdrant_http_port":
        _env = "QDRANT_PORT"
    elif _field == "qdrant_grpc_port":
        _env = "QDRANT_GRPC_PORT"
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
    agent_model: str = "claude-opus-4-6"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    aws_session_token: str | None = None
    agent_permission_mode: str = "dontAsk"
    # Sub-model overrides for Bedrock/LiteLLM — claude CLI dispatches
    # internal calls to haiku/sonnet-sized models and its direct-API
    # defaults do not resolve on Bedrock.
    agent_default_sonnet_model: str | None = None
    agent_default_haiku_model: str | None = None
    agent_subagent_model: str | None = None
    # CLI feature flags — passed through to the SDK env when set.
    # None = rely on provider-specific defaults in _build_sdk_env.
    agent_disable_experimental_betas: str | None = None
    agent_enable_tool_search: str | None = None

    @property
    def has_anthropic(self) -> bool:
        """True when a direct Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_bedrock(self) -> bool:
        """True when AWS Bedrock credentials are configured."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    # Database
    db_url: str = "postgresql+psycopg://localhost:5533/atelier"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334

    # Data
    parquet_dir: str = "build/data"

    # Classification pipeline
    classify_connection_name: str = ""
    classify_database: str = "default"
    classify_sample_size: int = 50
    classify_tables_limit: int = 100
    classify_embedding_model: str = "all-MiniLM-L6-v2"
    classify_auto_start: bool = False

    # CML
    cml_project_id: str | None = None
    cml_domain: str | None = None
    cml_engine_id: str | None = None
    cml_data_connections: str = ""

    @property
    def is_cml(self) -> bool:
        """True when running inside Cloudera ML."""
        return self.cml_project_id is not None

    @property
    def cml_data_connection_names(self) -> list[str]:
        """Parsed list of CAI Data Platform connection names."""
        return [s.strip() for s in self.cml_data_connections.split(",") if s.strip()]

    @property
    def anthropic_model_id(self) -> str:
        """The agent_model normalized to a plain Anthropic model ID."""
        return extract_anthropic_model_id(self.agent_model)

    @property
    def model_family(self) -> str | None:
        """The model family tier (opus/sonnet/haiku) or None."""
        return extract_model_family(self.anthropic_model_id)


# ── Model ID extraction ─────────────────────────────────────────


def extract_anthropic_model_id(raw: str) -> str:
    """Extract a plain Anthropic model ID from a Bedrock ARN or model identifier.

    Handles:
    - Full ARN: arn:aws:bedrock:...:inference-profile/us.anthropic.claude-opus-4-6-v1
    - Bedrock model ID: us.anthropic.claude-opus-4-6-v1
    - On-demand: anthropic.claude-3-5-sonnet-20241022-v2:0
    - Plain Anthropic ID: claude-opus-4-6 (passthrough)
    """
    model_part = raw
    # Strip ARN prefix → everything after the last /
    if raw.startswith("arn:aws:bedrock:"):
        model_part = raw.rsplit("/", 1)[-1]
    # Strip regional/vendor prefix: (us.|eu.)anthropic. or bare anthropic.
    model_part = re.sub(r"^(?:[\w.-]+\.)?anthropic\.", "", model_part)
    # Strip Bedrock version suffix: -v1, -v2:0, etc.
    model_part = re.sub(r"-v\d+(?::\d+)?$", "", model_part)
    return model_part


def extract_model_family(model_id: str) -> str | None:
    """Extract the model family from an Anthropic model ID.

    Returns "opus", "sonnet", "haiku", or None.
    """
    match = re.search(r"(opus|sonnet|haiku)", model_id)
    return match.group(1) if match else None


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


_SECRET_FIELDS = frozenset({
    "anthropic_api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
})


def materialize_config_json(
    cfg: AtelierConfig,
    output_path: str | Path,
) -> Path:
    """Write resolved config as JSON for conftest policy validation.

    Secrets are redacted (set values become ``"***set***"``).
    Includes derived booleans: ``has_anthropic``, ``has_bedrock``, ``is_cml``.

    Returns:
        The output path.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if val is None:
            continue
        env_name = _FIELD_TO_ENV.get(f.name, f"ATELIER_{f.name.upper()}")
        if f.name in _SECRET_FIELDS:
            data[env_name] = "***set***"
        elif isinstance(val, bool):
            data[env_name] = val
        else:
            data[env_name] = val

    # Derived booleans for policy rules
    data["has_anthropic"] = cfg.has_anthropic
    data["has_bedrock"] = cfg.has_bedrock
    data["is_cml"] = cfg.is_cml

    output_path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return output_path


# ── Preflight validation ─────────────────────────────────────────

_MATERIALIZED_PATH = _PROJECT_ROOT / "build" / "config" / "atelier.env"


def validate_materialized_config(
    path: str | Path | None = None,
) -> list[str]:
    """Validate the materialized config file for completeness.

    Returns a list of error strings. Empty list means valid.
    Delegates to :func:`atelier.preflight.run_preflight` for structured checks.

    Args:
        path: Path to the materialized env file. Defaults to
            build/config/atelier.env.
    """
    from atelier.preflight import run_preflight

    env_path = Path(path) if path else _MATERIALIZED_PATH
    result = run_preflight(load_config(), env_path=env_path)
    return [c.message for c in result.denies]
