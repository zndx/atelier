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
    cfg = load_config(overrides={"agent_model": "claude-opus-4-7"})

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
    # Governance
    "governance.atlas.url": ("governance_atlas_url", str),
    "governance.atlas.username": ("governance_atlas_username", str),
    "governance.atlas.password": ("governance_atlas_password", str),
    "governance.ranger.url": ("governance_ranger_url", str),
    "governance.ranger.username": ("governance_ranger_username", str),
    "governance.ranger.password": ("governance_ranger_password", str),
    "governance.cluster_name": ("governance_cluster_name", str),
    "governance.verify_ssl": ("governance_verify_ssl", bool),
    "governance.auto_sync": ("governance_auto_sync", bool),
    "governance.dry_run": ("governance_dry_run", bool),
    # Overwatch
    "overwatch.enabled": ("overwatch_enabled", bool),
    "overwatch.autonomy": ("overwatch_autonomy", str),
    "overwatch.model": ("overwatch_model", str),
    "overwatch.github_app_id": ("overwatch_github_app_id", str),
    "overwatch.github_private_key_path": ("overwatch_github_private_key_path", str),
    "overwatch.github_repo": ("overwatch_github_repo", str),
    "classify.connection_name": ("classify_connection_name", str),
    "classify.database": ("classify_database", str),
    "classify.sample_size": ("classify_sample_size", int),
    "classify.column_sample_limit": ("classify_column_sample_limit", int),
    "classify.tables_limit": ("classify_tables_limit", int),
    "classify.embedding_model": ("classify_embedding_model", str),
    "classify.embedding_device": ("classify_embedding_device", str),
    "classify.embedding_batch_size": ("classify_embedding_batch_size", int),
    # GPU acceleration
    "classify.gpu.enabled": ("classify_gpu_enabled", str),
    "classify.gpu.shard_threshold": ("classify_gpu_shard_threshold", int),
    "classify.gpu.sage_chunk_permutations": ("classify_gpu_sage_chunk", int),
    "classify.auto_start": ("classify_auto_start", bool),
    "classify.default_source": ("classify_default_source", str),
    "classify.subagent_model": ("classify_subagent_model", str),
    "classify.meta_tagging_dir": ("classify_meta_tagging_dir", str),
    # ML classifier model paths
    "classify.catboost_model_path": ("classify_catboost_model_path", str),
    "classify.svm_model_path": ("classify_svm_model_path", str),
    # LLM backend for classification
    "classify.llm.backend": ("classify_llm_backend", str),
    "classify.llm.api_key": ("classify_llm_api_key", str),
    "classify.llm.model": ("classify_llm_model", str),
    "classify.llm.base_url": ("classify_llm_base_url", str),
    "classify.llm.max_tokens": ("classify_llm_max_tokens", int),
    "classify.llm.temperature": ("classify_llm_temperature", float),
    "classify.llm.columns_per_call": ("classify_llm_columns_per_call", int),
    "classify.llm.max_retries": ("classify_llm_max_retries", int),
    "classify.llm.disable_reasoning": ("classify_llm_disable_reasoning", bool),
    "classify.llm.reasoning_budget": ("classify_llm_reasoning_budget", int),
    "classify.llm.discount": ("classify_llm_discount", float),
    # DST fusion strategy
    "classify.fusion_strategy": ("classify_fusion_strategy", str),
    # Bootstrap convergence
    "classify.bootstrap.max_iterations": ("classify_bootstrap_max_iterations", int),
    "classify.bootstrap.k_threshold": ("classify_bootstrap_k_threshold", float),
    "classify.bootstrap.coverage_target": ("classify_bootstrap_coverage_target", float),
    "classify.bootstrap.max_total_llm_calls": ("classify_bootstrap_max_total_llm_calls", int),
    "classify.bootstrap.frontier_svm_retrain": ("classify_bootstrap_frontier_svm_retrain", bool),
    "classify.bootstrap.frontier_svm_min_labels": ("classify_bootstrap_frontier_svm_min_labels", int),
    "classify.bootstrap.gap_threshold": ("classify_bootstrap_gap_threshold", float),
    "classify.bootstrap.clarity_target": ("classify_bootstrap_clarity_target", float),
    "classify.bootstrap.bel_floor": ("classify_bootstrap_bel_floor", float),
    # DST discount factors
    "classify.discounts.cosine": ("classify_discount_cosine", float),
    "classify.discounts.svm": ("classify_discount_svm", float),
    "classify.discounts.pattern_theta": ("classify_discount_pattern_theta", float),
    "classify.discounts.name_match_exact": ("classify_discount_name_match_exact", float),
    "classify.discounts.name_match_code": ("classify_discount_name_match_code", float),
    "classify.discounts.name_match_alias": ("classify_discount_name_match_alias", float),
    "classify.discounts.name_match_overlap": ("classify_discount_name_match_overlap", float),
    "classify.discounts.catboost_base": ("classify_discount_catboost_base", float),
    "classify.discounts.catboost_variance_scale": ("classify_discount_catboost_variance_scale", float),
    "classify.discounts.catboost_max": ("classify_discount_catboost_max", float),
    "classify.discounts.catboost_fallback": ("classify_discount_catboost_fallback", float),
    "classify.discounts.confusable_ratio_threshold": ("classify_discount_confusable_ratio_threshold", float),
    # CatBoost training hyperparameters
    "classify.catboost.iterations": ("classify_catboost_iterations", int),
    "classify.catboost.depth": ("classify_catboost_depth", int),
    "classify.catboost.learning_rate": ("classify_catboost_learning_rate", float),
    "classify.catboost.fit_to_llm": ("classify_catboost_fit_to_llm", bool),
    "classify.catboost.fit_to_llm_min_labels": ("classify_catboost_fit_to_llm_min_labels", int),
    # SHAP explanations
    "classify.shap.enabled": ("classify_shap_enabled", bool),
    "classify.shap.top_k": ("classify_shap_top_k", int),
    # SAGE feature importance
    "classify.sage.enabled": ("classify_sage_enabled", bool),
    "classify.sage.permutations": ("classify_sage_permutations", int),
    # Agent-driven convergence
    "classify.agent.enabled": ("classify_agent_enabled", bool),
    "classify.agent.max_turns": ("classify_agent_max_turns", int),
    "classify.agent.model": ("classify_agent_model", str),
    # Monte Carlo sampling
    "classify.monte_carlo.min_corpus_size": ("mc_min_corpus_size", int),
    "classify.monte_carlo.sample_fraction": ("mc_sample_fraction", float),
    "classify.monte_carlo.min_per_stratum": ("mc_min_per_stratum", int),
    "classify.monte_carlo.max_frontier_columns": ("mc_max_frontier_columns", int),
    "classify.monte_carlo.propagation_threshold": ("mc_propagation_threshold", float),
    "classify.monte_carlo.propagation_discount": ("mc_propagation_discount", float),
    # Row-level Monte Carlo
    "classify.row_mc.enabled": ("row_mc_enabled", bool),
    "classify.row_mc.k": ("row_mc_k", int),
    "classify.row_mc.strategy": ("row_mc_strategy", str),
    "classify.row_mc.iterations": ("row_mc_iterations", int),
    "classify.row_mc.max_iterations": ("row_mc_max_iterations", int),
    "classify.row_mc.adaptive_escalation": ("row_mc_adaptive_escalation", bool),
    # Background feature analysis
    "classify.background_analysis": ("classify_background_analysis", bool),
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
    elif _field == "classify_subagent_model":
        _env = "ANTHROPIC_SUBAGENT_MODEL"
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
    agent_model: str = "claude-opus-4-7"
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
    classify_column_sample_limit: int = 1000
    classify_tables_limit: int = 100
    classify_embedding_model: str = "all-MiniLM-L6-v2"
    classify_embedding_device: str = "auto"
    classify_embedding_batch_size: int = 32
    # GPU acceleration
    classify_gpu_enabled: str = "auto"
    classify_gpu_shard_threshold: int = 200_000
    classify_gpu_sage_chunk: int = 16
    classify_auto_start: bool = False
    classify_default_source: str = ""  # empty = ootb-sample
    classify_subagent_model: str | None = None
    # Meta-tagging source — private reference data mount path.  Never
    # committed to git.  When empty, the module falls back to the
    # ATELIER_META_TAGGING_DIR env var and then to the maintainer
    # default at ~/local/tmp/meta-tagging.
    classify_meta_tagging_dir: str = ""

    # ML classifier model paths
    classify_catboost_model_path: str = "build/models/catboost.cbm"
    classify_svm_model_path: str = "build/models/svm.pkl"

    # Classification LLM backend
    classify_llm_backend: str = "openai_compatible"
    classify_llm_api_key: str | None = None
    classify_llm_model: str = "glm-4.7"
    classify_llm_base_url: str | None = None
    classify_llm_max_tokens: int = 65536
    classify_llm_temperature: float = 0.0
    classify_llm_columns_per_call: int = 25
    classify_llm_max_retries: int = 3
    classify_llm_disable_reasoning: bool = False
    classify_llm_reasoning_budget: int = 8192
    classify_llm_discount: float = 0.10
    # DST fusion strategy: "dempster" (default, normalizing) or "yager"
    # (redirect conflict to Θ).  Yager preserves epistemic honesty under
    # high conflict at the cost of higher ignorance mass.
    classify_fusion_strategy: str = "dempster"

    # Bootstrap convergence
    classify_bootstrap_max_iterations: int = 5
    classify_bootstrap_k_threshold: float = 0.2
    classify_bootstrap_coverage_target: float = 1.0
    classify_bootstrap_max_total_llm_calls: int = 5000
    classify_bootstrap_frontier_svm_retrain: bool = True
    classify_bootstrap_frontier_svm_min_labels: int = 20
    # Belief-gap convergence
    classify_bootstrap_gap_threshold: float = 0.15
    classify_bootstrap_clarity_target: float = 0.20
    classify_bootstrap_bel_floor: float = 0.50

    # DST discount factors
    classify_discount_cosine: float = 0.30
    classify_discount_svm: float = 0.20
    classify_discount_pattern_theta: float = 0.25
    classify_discount_name_match_exact: float = 0.70
    classify_discount_name_match_code: float = 0.50
    classify_discount_name_match_alias: float = 0.50
    classify_discount_name_match_overlap: float = 0.30
    classify_discount_catboost_base: float = 0.10
    classify_discount_catboost_variance_scale: float = 1.6
    classify_discount_catboost_max: float = 0.50
    classify_discount_catboost_fallback: float = 0.15
    classify_discount_confusable_ratio_threshold: float = 3.0

    # CatBoost training hyperparameters
    classify_catboost_iterations: int = 1000
    classify_catboost_depth: int = 6
    classify_catboost_learning_rate: float = 0.10
    classify_catboost_fit_to_llm: bool = False
    classify_catboost_fit_to_llm_min_labels: int = 30

    # SHAP explanations
    classify_shap_enabled: bool = True
    classify_shap_top_k: int = 3

    # SAGE feature importance
    classify_sage_enabled: bool = False
    classify_sage_permutations: int = 512

    # Agent-driven convergence
    classify_agent_enabled: bool = False
    classify_agent_max_turns: int = 10
    classify_agent_model: str | None = None  # falls back to agent_model

    # Monte Carlo sampling
    mc_min_corpus_size: int = 200
    mc_sample_fraction: float = 1.00
    mc_min_per_stratum: int = 3
    mc_max_frontier_columns: int = 500
    mc_propagation_threshold: float = 0.85
    mc_propagation_discount: float = 0.30

    # Row-level Monte Carlo
    row_mc_enabled: bool = False
    row_mc_k: int = 10
    row_mc_strategy: str = "stratified"
    row_mc_iterations: int = 3
    row_mc_max_iterations: int = 5
    row_mc_adaptive_escalation: bool = True

    # Background feature analysis
    classify_background_analysis: bool = True

    @property
    def has_classify_llm(self) -> bool:
        """True when an LLM backend is available for classification.

        Sources:
        1. Explicit classify LLM (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
        2. ANTHROPIC_SUBAGENT_MODEL (backend inferred from model format)
        """
        return bool(
            self.classify_llm_api_key
            or self.classify_llm_base_url
            or self.classify_subagent_model
        )

    # CML
    cml_project_id: str | None = None
    cml_domain: str | None = None
    cml_engine_id: str | None = None
    cml_data_connections: str = ""

    # Governance
    governance_atlas_url: str = ""
    governance_atlas_username: str = "admin"
    governance_atlas_password: str = ""
    governance_ranger_url: str = ""
    governance_ranger_username: str = "admin"
    governance_ranger_password: str = ""
    governance_cluster_name: str = "cm"
    governance_verify_ssl: bool = False
    governance_auto_sync: bool = False
    governance_dry_run: bool = False

    # Overwatch
    overwatch_enabled: bool = True
    overwatch_autonomy: str = "propose"
    overwatch_model: str = "claude-opus-4-7"
    overwatch_github_app_id: str = ""
    overwatch_github_private_key_path: str = ""
    overwatch_github_repo: str = ""

    @property
    def has_overwatch(self) -> bool:
        """True only when overwatch is enabled AND Anthropic API is available.

        Overwatch requires direct Anthropic API access for full Claude Code
        capabilities (/fast, worktrees, subagents). Bedrock-only deployments
        cannot activate overwatch.
        """
        return self.overwatch_enabled and self.has_anthropic

    @property
    def has_atlas(self) -> bool:
        """True when Atlas URL is configured."""
        return bool(self.governance_atlas_url)

    @property
    def has_ranger(self) -> bool:
        """True when Ranger URL is configured."""
        return bool(self.governance_ranger_url)

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


def is_bedrock_model(model_id: str) -> bool:
    """True when the model identifier is a Bedrock ARN or Bedrock model ID."""
    return model_id.startswith("arn:") or "anthropic." in model_id


def region_from_arn(model_id: str) -> str | None:
    """Extract the AWS region from a Bedrock ARN.

    Cross-region inference profiles encode their target region in the ARN
    (``arn:aws:bedrock:<region>:<account>:...``).  Without this, the boto3
    client connects to the default ``AWS_REGION``, which causes
    ``ResourceNotFoundException`` on ``invoke_model``.

    Returns the region string if *model_id* is an ARN, else ``None``.
    """
    if not model_id.startswith("arn:aws:bedrock:"):
        return None
    # arn:aws:bedrock:<region>:<account>:<resource-type>/<id>
    parts = model_id.split(":")
    if len(parts) >= 4:
        return parts[3]
    return None


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
    "classify_llm_api_key",
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
