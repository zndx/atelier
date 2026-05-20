# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
    "overwatch.github_app_id": ("overwatch_github_app_id", str),
    "overwatch.github_private_key_path": ("overwatch_github_private_key_path", str),
    "overwatch.github_repo": ("overwatch_github_repo", str),
    # Nautilus mid-run watcher
    "overwatch.nautilus.enabled": ("overwatch_nautilus_enabled", bool),
    "overwatch.nautilus.poll_interval_s": ("overwatch_nautilus_poll_interval_s", float),
    "overwatch.nautilus.stall_threshold_s": ("overwatch_nautilus_stall_threshold_s", float),
    "overwatch.nautilus.llm_sweep_threshold_s": ("overwatch_nautilus_llm_sweep_threshold_s", float),
    "overwatch.nautilus.failed_batch_threshold": ("overwatch_nautilus_failed_batch_threshold", int),
    "classify.connection_name": ("classify_connection_name", str),
    "classify.database": ("classify_database", str),
    "classify.sample_size": ("classify_sample_size", int),
    "classify.column_sample_limit": ("classify_column_sample_limit", int),
    "classify.tables_limit": ("classify_tables_limit", int),
    "classify.exclude_reference_columns": ("classify_exclude_reference_columns", bool),
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
    "classify.reference_uri": ("classify_reference_uri", str),
    # ML classifier model paths
    "classify.catboost_model_path": ("classify_catboost_model_path", str),
    "classify.svm_model_path": ("classify_svm_model_path", str),
    "classify.svm.hierarchical": ("classify_svm_hierarchical", bool),
    "classify.svm.nhsvm_temperature": ("classify_svm_nhsvm_temperature", float),
    "classify.svm.nhsvm_svd_components": ("classify_svm_nhsvm_svd_components", int),
    # LLM backend for classification
    "classify.llm.backend": ("classify_llm_backend", str),
    "classify.llm.api_key": ("classify_llm_api_key", str),
    "classify.llm.model": ("classify_llm_model", str),
    "classify.llm.base_url": ("classify_llm_base_url", str),
    "classify.llm.max_tokens": ("classify_llm_max_tokens", int),
    "classify.llm.temperature": ("classify_llm_temperature", float),
    "classify.llm.columns_per_call": ("classify_llm_columns_per_call", int),
    "classify.llm.min_columns_per_call": ("classify_llm_min_columns_per_call", int),
    "classify.llm.max_retries": ("classify_llm_max_retries", int),
    "classify.llm.disable_reasoning": ("classify_llm_disable_reasoning", bool),
    "classify.llm.reasoning_budget": ("classify_llm_reasoning_budget", int),
    "classify.llm.discount": ("classify_llm_discount", float),
    # DST fusion strategy
    "classify.fusion_strategy": ("classify_fusion_strategy", str),
    # Bootstrap convergence
    "app.display_name": ("app_display_name", str),
    "classify.bootstrap.max_iterations": ("classify_bootstrap_max_iterations", int),
    "classify.bootstrap.min_iterations": ("classify_bootstrap_min_iterations", int),
    "classify.cautious_review.enabled": ("classify_cautious_review_enabled", bool),
    "classify.cautious_review.bel_threshold": ("classify_cautious_review_bel_threshold", float),
    "classify.cautious_review.backend": ("classify_cautious_review_backend", str),
    "classify.cautious_review.shortlist_permissive": ("classify_cautious_review_shortlist_permissive", bool),
    "classify.cautious_review.exclude_opaque_siblings": ("classify_cautious_review_exclude_opaque_siblings", bool),
    "classify.cautious_review.stability_guard_enabled": ("classify_cautious_review_stability_guard_enabled", bool),
    "classify.cautious_review.stability_guard_llm_conf": ("classify_cautious_review_stability_guard_llm_conf", float),
    "classify.resolve_llm_annotation_mnemonic": ("classify_resolve_llm_annotation_mnemonic", bool),
    "classify.exclude_temp_tables": ("classify_exclude_temp_tables", bool),
    "classify.bootstrap.k_threshold": ("classify_bootstrap_k_threshold", float),
    "classify.bootstrap.coverage_target": ("classify_bootstrap_coverage_target", float),
    "classify.bootstrap.max_total_llm_calls": ("classify_bootstrap_max_total_llm_calls", int),
    "classify.bootstrap.max_total_llm_attempts": ("classify_bootstrap_max_total_llm_attempts", int),
    "classify.bootstrap.sweep_deadline_s": ("classify_bootstrap_sweep_deadline_s", float),
    "classify.bootstrap.max_consecutive_halve_failures": ("classify_bootstrap_max_consecutive_halve_failures", int),
    "classify.bootstrap.gap_threshold": ("classify_bootstrap_gap_threshold", float),
    "classify.bootstrap.clarity_target": ("classify_bootstrap_clarity_target", float),
    "classify.bootstrap.bel_floor": ("classify_bootstrap_bel_floor", float),
    "classify.bootstrap.indep_revisit_mass_threshold": ("classify_bootstrap_indep_revisit_mass_threshold", float),
    "classify.bootstrap.top1_margin_threshold": ("classify_bootstrap_top1_margin_threshold", float),
    # DST discount factors
    "classify.discounts.cosine": ("classify_discount_cosine", float),
    # Late-interaction ColBERT cosine via Qdrant — feature-flag gated
    "classify.cosine.late_interaction.enabled": ("classify_cosine_late_interaction_enabled", bool),
    "classify.cosine.late_interaction.model": ("classify_colbert_model", str),
    "classify.discounts.svm": ("classify_discount_svm", float),
    "classify.subsumption_alignment.score_threshold": ("classify_subsumption_score_threshold", float),
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
    "classify.shap.method": ("classify_shap_method", str),
    "classify.taxonomy.strict_validation": ("classify_taxonomy_strict_validation", bool),
    "classify.liveness_probe_timeout_s": ("classify_liveness_probe_timeout_s", float),
    "classify.metadata_probe_timeout_s": ("classify_metadata_probe_timeout_s", float),
    "classify.phase_heartbeat_interval_s": ("classify_phase_heartbeat_interval_s", float),
    # SAGE feature importance
    "classify.sage.enabled": ("classify_sage_enabled", bool),
    "classify.sage.permutations": ("classify_sage_permutations", int),
    # Agent-driven convergence
    "classify.agent.enabled": ("classify_agent_enabled", bool),
    "classify.agent.max_turns": ("classify_agent_max_turns", int),
    "classify.agent.model": ("classify_agent_model", str),
    # Per-iteration scoring against the Opus-crafted reference
    "classify.evaluation.agent_mediated_path": ("classify_evaluation_agent_mediated_path", str),
    "classify.evaluation.enabled": ("classify_evaluation_enabled", bool),
    # Disk-space guard
    "classify.disk_guard.enabled": ("classify_disk_guard_enabled", bool),
    "classify.disk_guard.headroom_multiplier": ("classify_disk_guard_headroom_multiplier", float),
    "classify.disk_guard.bootstrap_floor_bytes": ("classify_disk_guard_bootstrap_floor_bytes", int),
    "classify.disk_guard.min_runs_for_stats": ("classify_disk_guard_min_runs_for_stats", int),
    # Monte Carlo sampling
    "classify.monte_carlo.min_corpus_size": ("mc_min_corpus_size", int),
    "classify.monte_carlo.sample_fraction": ("mc_sample_fraction", float),
    "classify.monte_carlo.min_per_stratum": ("mc_min_per_stratum", int),
    "classify.monte_carlo.max_sampled_columns": ("mc_max_sampled_columns", int),
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
    # Annotation enrichment (LLM-mediated, provider-co-located with classify)
    "enrichment.model_override": ("enrichment_model_override", str),
    "enrichment.reasoning_budget": ("enrichment_reasoning_budget", int),
    "enrichment.max_tokens": ("enrichment_max_tokens", int),
    "enrichment.max_attempts": ("enrichment_max_attempts", int),
    "enrichment.temperature": ("enrichment_temperature", float),
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
    # Governance fields use shortened names in HOCON ${?VAR} substitution
    # (config/base.conf:494-517) — mirror those here so materialization
    # round-trips correctly under `env -i $(cat build/config/atelier.env...)`.
    elif _field == "governance_atlas_url":
        _env = "ATELIER_ATLAS_URL"
    elif _field == "governance_atlas_username":
        _env = "ATELIER_ATLAS_USER"
    elif _field == "governance_atlas_password":
        _env = "ATELIER_ATLAS_PASSWORD"
    elif _field == "governance_ranger_url":
        _env = "ATELIER_RANGER_URL"
    elif _field == "governance_ranger_username":
        _env = "ATELIER_RANGER_USER"
    elif _field == "governance_ranger_password":
        _env = "ATELIER_RANGER_PASSWORD"
    elif _field == "governance_cluster_name":
        _env = "ATELIER_CLUSTER_NAME"
    _FIELD_TO_ENV[_field] = _env


@dataclass
class AtelierConfig:
    """Resolved application configuration."""

    # Application identity (surfaced to UI as a lightweight rebrand
    # knob; defaults to "Atelier" upstream, override via SOPS dotenv
    # ``ATELIER_APP_DISPLAY_NAME`` for deployment-specific naming).
    app_display_name: str = "Atelier"

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
    # Reference-column exclusion — see config/base.conf for the
    # regex pattern and rationale.  Kept as a toggle so UAT reviewers
    # can demonstrate classifier quality in both configurations on
    # the synth corpus that motivated it.
    classify_exclude_reference_columns: bool = True
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
    # committed to git.  When empty, the resolver walks:
    # ATELIER_META_TAGGING_DIR env var → <repo>/build/meta-tagging/ (UAT
    # snapshot, gitignored) → ~/local/tmp/meta-tagging/ (legacy default).
    classify_meta_tagging_dir: str = ""
    # Optional curated-reference CSV — per-column expected codes used
    # by evaluation_report + overwatch.  Set to a path or file:// URI.
    # Contents never checked in; deployments point at their own CSV.
    classify_reference_uri: str = ""

    # ML classifier model paths
    classify_catboost_model_path: str = "build/models/catboost.cbm"
    classify_svm_model_path: str = "build/models/svm.pkl"
    classify_svm_hierarchical: bool = True
    classify_svm_nhsvm_temperature: float = 1.0
    classify_svm_nhsvm_svd_components: int = 200

    # Classification LLM backend
    classify_llm_backend: str = "openai_compatible"
    classify_llm_api_key: str | None = None
    classify_llm_model: str = "glm-4.7"
    classify_llm_base_url: str | None = None
    classify_llm_max_tokens: int = 65536
    classify_llm_temperature: float = 0.0
    classify_llm_columns_per_call: int = 25
    # Exhaustive-halving floor: a failing batch halves recursively until
    # this size, at which point a per-batch failure is recorded rather
    # than the columns being silently dropped.  1 = per-column fallback.
    classify_llm_min_columns_per_call: int = 1
    classify_llm_max_retries: int = 3
    classify_llm_disable_reasoning: bool = False
    classify_llm_reasoning_budget: int = 8192
    classify_llm_discount: float = 0.15
    # DST fusion strategy: "dempster" (default, normalizing) or "yager"
    # (redirect conflict to Θ).  Yager preserves epistemic honesty under
    # high conflict at the cost of higher ignorance mass.
    classify_fusion_strategy: str = "dempster"

    # Bootstrap convergence
    classify_bootstrap_max_iterations: int = 5
    classify_bootstrap_min_iterations: int = 2
    # Cautious-code review — PROVEN HARMFUL.  Empirically destroys
    # accuracy (ce4f3777: 76% reroute miss rate, −13.6pp vs LLM-only).
    # Default OFF (enabled=False) with bel_threshold=0.0 (unreachable)
    # as a belt-and-suspenders guard.  See cautious_review.py docstring.
    classify_cautious_review_enabled: bool = False
    classify_cautious_review_bel_threshold: float = 0.0
    classify_cautious_review_backend: str = "default"
    # The following sub-knobs are part of the proven-harmful cautious
    # review system.  All prior remediations (R2a, R2c, R3) failed to
    # overcome the fundamental accuracy-destruction problem.  Retained
    # only for controlled re-validation experiments.
    classify_cautious_review_shortlist_permissive: bool = True
    classify_cautious_review_exclude_opaque_siblings: bool = True
    classify_cautious_review_stability_guard_enabled: bool = True
    classify_cautious_review_stability_guard_llm_conf: float = 0.80
    # R1: annotation-mnemonic fallback in mass_functions
    # (_resolve_to_focal_element).  Default on — strictly recovers LLM
    # evidence for ~95 columns / 33% of corpus in 8d67b1ed (audit
    # P0 expanded).  Toggle off only for ablation runs.
    classify_resolve_llm_annotation_mnemonic: bool = True
    # R6: skip Hive/Hue temp tables (``__tmp_*`` prefix) at discovery.
    # 14 hallucinations from one such table in 8d67b1ed.
    classify_exclude_temp_tables: bool = True
    classify_bootstrap_k_threshold: float = 0.2
    classify_bootstrap_coverage_target: float = 1.0
    classify_bootstrap_max_total_llm_calls: int = 5000
    classify_bootstrap_max_total_llm_attempts: int = 10000
    classify_bootstrap_sweep_deadline_s: float = 0.0
    classify_bootstrap_max_consecutive_halve_failures: int = 8
    # Belief-gap convergence — defaults mirror config/base.conf.
    # Recalibrated 2026-05-03 for the parent-aware DST frame; see the
    # comment in base.conf for the per-knob rationale.
    classify_bootstrap_gap_threshold: float = 0.18
    classify_bootstrap_clarity_target: float = 0.25
    classify_bootstrap_bel_floor: float = 0.45
    classify_bootstrap_indep_revisit_mass_threshold: float = 0.45
    # DST sensitivity Rec 3 — rank-instability revisit gate.
    classify_bootstrap_top1_margin_threshold: float = 0.05

    # DST discount factors.  ``catboost_*`` defaults are calibrated
    # well above the cosine discount because that source is LLM-
    # derivative at the per-column level (``fit_to_llm`` mode trains
    # on the live run's LLM labels — see
    # ml_train.fit_catboost_to_llm_labels).  ``svm`` sits between
    # fully-independent and CatBoost: features are independent (TF-IDF)
    # and labels come from the synth generators, but the prediction-
    # to-frame-element mapping passes through the LLM-mediated
    # alignment in classify.ontology_alignment.  Per Shafer 1976
    # §11.3 + Denoeux 2008, non-distinct evidence requires a
    # discount to avoid double-counting under Dempster's rule; the
    # current SVM default reflects the weakly-non-distinct regime.
    classify_discount_cosine: float = 0.20
    # Late-interaction multi-vector cosine via Qdrant is the production
    # cosine evidence source under the DST-independence architecture.
    # Default ON: this is the path that restores independence among
    # evidence sources.  Disabling it routes the pipeline back through
    # the single-vector cosine path which is known to under-discriminate
    # on adversarial corpora and is retained only as a transitional
    # emergency fallback during deployment rollout.  When this flag is
    # True and the late-interaction path cannot run (no enriched
    # collection registered, Qdrant unreachable, qdrant-client missing),
    # the pipeline logs a WARNING and marks the run as degraded — that
    # condition is a deployment issue, not a normal operating mode.
    # See docs/src/architecture/late-interaction-cosine.md.
    classify_cosine_late_interaction_enabled: bool = True
    classify_colbert_model: str = "colbert-ir/colbertv2.0"
    classify_discount_svm: float = 0.22
    classify_subsumption_score_threshold: float = 0.35
    classify_discount_pattern_theta: float = 0.25
    classify_discount_name_match_exact: float = 0.70
    classify_discount_name_match_code: float = 0.50
    classify_discount_name_match_alias: float = 0.50
    classify_discount_name_match_overlap: float = 0.30
    classify_discount_catboost_base: float = 0.55
    classify_discount_catboost_variance_scale: float = 1.6
    classify_discount_catboost_max: float = 0.75
    classify_discount_catboost_fallback: float = 0.55
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
    classify_shap_method: str = "auto"
    classify_taxonomy_strict_validation: bool = False
    # Pre-LLM phase observability — proof-of-progress paradigm.
    # Liveness / metadata probes have short hard timeouts (fail-fast
    # on dead connections); heavy queries themselves run unbounded
    # but observable via phase_heartbeat advancing FSM.updated_at.
    classify_liveness_probe_timeout_s: float = 5.0
    classify_metadata_probe_timeout_s: float = 10.0
    classify_phase_heartbeat_interval_s: float = 5.0

    # SAGE feature importance
    classify_sage_enabled: bool = False
    classify_sage_permutations: int = 512

    # Agent-driven convergence
    classify_agent_enabled: bool = False
    classify_agent_max_turns: int = 10
    classify_agent_model: str | None = None  # falls back to agent_model

    # Per-iteration scoring against the Opus-crafted reference (see
    # atelier.classify.incremental_scoring).  When agent_mediated_path is
    # empty, incremental scoring auto-disables with a single log line.
    classify_evaluation_agent_mediated_path: str = ""
    classify_evaluation_enabled: bool = True

    # Disk-space guard (atelier.classify.incremental_scoring.DiskGuardConfig).
    # When enabled, the pipeline refuses to start (and refuses to advance
    # past a bootstrap iteration) when projected free space falls short
    # of mean+2σ × headroom of historical run sizes.
    classify_disk_guard_enabled: bool = True
    classify_disk_guard_headroom_multiplier: float = 1.25
    classify_disk_guard_bootstrap_floor_bytes: int = 209715200  # 200 MiB
    classify_disk_guard_min_runs_for_stats: int = 3

    # Monte Carlo sampling
    mc_min_corpus_size: int = 200
    mc_sample_fraction: float = 1.00
    mc_min_per_stratum: int = 3
    mc_max_sampled_columns: int = 500
    mc_propagation_threshold: float = 0.80
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
    overwatch_github_app_id: str = ""
    overwatch_github_private_key_path: str = ""
    overwatch_github_repo: str = ""
    # Nautilus mid-run watcher — polls FSM + batch_audit between LLM
    # calls and fires interventions on stall / slow-sweep / accumulated
    # batch failures.  Cooperative cancellation only (sets a flag on the
    # live BootstrapState); no SIGKILL of the pipeline thread.
    overwatch_nautilus_enabled: bool = True
    overwatch_nautilus_poll_interval_s: float = 10.0
    overwatch_nautilus_stall_threshold_s: float = 120.0
    overwatch_nautilus_llm_sweep_threshold_s: float = 300.0
    overwatch_nautilus_failed_batch_threshold: int = 10

    # Annotation enrichment — LLM-mediated multi-vector profile
    # generation for the late-interaction cosine architecture.  The
    # backend is NOT a separate knob here: enrichment derives its
    # provider from classify_llm_backend (single operator-facing
    # selection of credentials + cost regime).  Only the model and
    # reasoning depth are tunable separately, because enrichment is
    # single-shot per taxonomy node and can afford the apex
    # reasoning model from whatever provider classify is using.
    enrichment_model_override: str | None = None
    enrichment_reasoning_budget: int = 16384
    enrichment_max_tokens: int = 8192
    enrichment_max_attempts: int = 3
    enrichment_temperature: float = 0.0

    @property
    def has_overwatch(self) -> bool:
        """True when overwatch is enabled AND the Web Terminal Agent has a
        runnable model.

        Overwatch follows the WTA's selected model — operators configure
        provider + model once and both surfaces pick it up.  When the WTA
        can't run (no Anthropic key AND no Bedrock creds), Overwatch
        also can't run; we deliberately do NOT silently fall back to a
        weaker default since a half-functional Overwatch report would
        do more harm than no report.
        """
        if not self.overwatch_enabled:
            return False
        # Same gate the WTA picker uses to decide if any catalog entry is
        # available.  Importing here avoids a circular import at module
        # load (terminal_catalog reads cfg).
        #
        # Fail-closed: if the catalog itself raises, we treat Overwatch
        # as unavailable rather than falling back to a credential-only
        # check.  The user-visible promise is "Overwatch runs whatever
        # WTA is configured to run" — degrading silently to a path that
        # WTA wouldn't accept would surprise reviewers and risk
        # reputational harm.
        try:
            from atelier.terminal_catalog import available_models
            return any(m.available for m in available_models(self))
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "has_overwatch: WTA catalog probe failed; reporting unavailable"
            )
            return False

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
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)
    if target_type is int:
        return int(val)
    if target_type is float:
        return float(val)
    # Strings get whitespace trimmed — AMP env forms and shell dotenv
    # round-trips can introduce leading/trailing whitespace that becomes
    # an illegal HTTP header when e.g. an API key reaches httpx.
    return str(val).strip()


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
