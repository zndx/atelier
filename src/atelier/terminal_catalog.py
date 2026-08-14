"""Frontier model catalog for the Web Terminal Agent panel.

A static list of (provider, model) entries the operator can pick
from. Availability per entry is derived at call time from the
current config's credential presence (``cfg.has_bedrock`` /
``cfg.has_anthropic``).  Bedrock entries resolve their concrete
``model_ref`` from the ATELIER_AGENT_MODEL / ANTHROPIC_DEFAULT_*
env-layer values so a per-deploy override via ``.env.cai.enc`` still
flows through without editing the catalog.

The catalog is deliberately small — two Opus-class frontier entries
(one per provider). The Web Terminal Agent is the operator's
interactive surface; Sonnet/Haiku tiers are reserved for pipeline
internals (subagent dispatch, classification LLM, tool search) and
route via ``ANTHROPIC_DEFAULT_SONNET_MODEL`` / ``_HAIKU_MODEL`` /
``CLAUDE_CODE_SUBAGENT_MODEL``, not this catalog. Operators wanting
something outside this list still route via the ``ATELIER_AGENT_MODEL``
env var directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _CatalogEntry:
    """Template entry — ``model_ref`` may be a placeholder that
    resolves at availability-probe time against a live cfg field."""

    id: str
    label: str
    provider: str  # "bedrock" | "anthropic"
    model_ref_source: str  # either a concrete ref or an "@attr:..." placeholder
    context_window: int
    max_output_tokens: int
    thinking: str  # "adaptive" | "extended" | "none"
    notes: str = ""
    # True → additionally requires cfg.anthropic_base_url (corporate
    # gateway entries make no sense against api.anthropic.com).
    requires_base_url: bool = False


@dataclass(frozen=True)
class ModelEntry:
    """Catalog entry resolved against a live config.

    Same shape as ``_CatalogEntry`` plus ``available`` (credential
    presence) and ``model_ref`` (the concrete string the SDK
    receives — placeholders resolved).
    """

    id: str
    label: str
    provider: str
    model_ref: str
    context_window: int
    max_output_tokens: int
    thinking: str
    available: bool
    unavailable_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model_ref": self.model_ref,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "thinking": self.thinking,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "notes": self.notes,
        }


# ── Catalog (frozen) ────────────────────────────────────────────
#
# Bedrock entries use the ``@attr:<field>`` placeholder pointing at
# an AtelierConfig field so the live .env.cai.enc value is what
# actually flows to the SDK.  Anthropic entries carry concrete
# direct-API model IDs.

_CATALOG: tuple[_CatalogEntry, ...] = (
    _CatalogEntry(
        id="bedrock-opus-4-6",
        label="Opus 4.6",
        provider="bedrock",
        model_ref_source="@attr:agent_model",
        context_window=200_000,
        max_output_tokens=32_000,
        thinking="extended",
        notes="Current ATELIER_AGENT_MODEL default.",
    ),
    _CatalogEntry(
        id="anthropic-opus-4-8",
        label="Opus 4.8",
        provider="anthropic",
        model_ref_source="claude-opus-4-8",
        context_window=1_000_000,
        max_output_tokens=64_000,
        thinking="adaptive",
        notes="Latest Opus on the direct API; 1M context window.",
    ),
    _CatalogEntry(
        id="anthropic-opus-4-7",
        label="Opus 4.7",
        provider="anthropic",
        model_ref_source="claude-opus-4-7",
        context_window=1_000_000,
        max_output_tokens=64_000,
        thinking="adaptive",
        notes="Previous-generation Opus on the direct API (still active; pinnable).",
    ),
    _CatalogEntry(
        id="gateway-agent-model",
        label="Gateway",
        provider="anthropic",
        model_ref_source="@attr:agent_model",
        context_window=200_000,
        max_output_tokens=64_000,
        thinking="extended",
        notes=(
            "Routes ATELIER_AGENT_MODEL through the configured gateway "
            "(ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)."
        ),
        requires_base_url=True,
    ),
)


def _resolve_ref(entry: _CatalogEntry, cfg) -> str:
    """Resolve ``@attr:<field>`` placeholders against cfg."""
    src = entry.model_ref_source
    if src.startswith("@attr:"):
        attr = src[len("@attr:"):]
        return getattr(cfg, attr, "") or ""
    return src


def _availability(entry: _CatalogEntry, resolved_ref: str, cfg) -> tuple[bool, str]:
    """Return (available, reason-if-not) for an entry given cfg."""
    if entry.provider == "bedrock":
        if not cfg.has_bedrock:
            return False, "Bedrock credentials not configured"
        if not resolved_ref:
            return False, f"Bedrock model ARN not set ({entry.model_ref_source!r})"
        return True, ""
    if entry.provider == "anthropic":
        if not cfg.has_anthropic:
            return False, "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN not set"
        if entry.requires_base_url and not getattr(cfg, "anthropic_base_url", None):
            return False, "ANTHROPIC_BASE_URL not set"
        return True, ""
    return False, f"unknown provider: {entry.provider}"


def available_models(cfg) -> list[ModelEntry]:
    """Return the full catalog stamped with live availability + refs.

    The list order is stable (matches ``_CATALOG``) so the UI radio
    group renders deterministically.  Rows where the provider isn't
    configured appear as unavailable with a human-readable reason.
    """
    out: list[ModelEntry] = []
    for e in _CATALOG:
        ref = _resolve_ref(e, cfg)
        ok, reason = _availability(e, ref, cfg)
        out.append(ModelEntry(
            id=e.id,
            label=e.label,
            provider=e.provider,
            model_ref=ref,
            context_window=e.context_window,
            max_output_tokens=e.max_output_tokens,
            thinking=e.thinking,
            available=ok,
            unavailable_reason=reason,
            notes=e.notes,
        ))
    return out


def find_by_id(entry_id: str, cfg) -> ModelEntry | None:
    """Look up a single resolved entry; ``None`` when the id is unknown."""
    for m in available_models(cfg):
        if m.id == entry_id:
            return m
    return None


def derive_from_agent_model(cfg) -> ModelEntry | None:
    """Find the catalog entry whose ``model_ref`` matches ``cfg.agent_model``.

    Used when no explicit selection has been made — the terminal
    should reflect what the deploy is currently wired to.  Returns
    ``None`` when ``cfg.agent_model`` doesn't correspond to any
    catalog entry (custom deployment).
    """
    target = (getattr(cfg, "agent_model", "") or "").strip()
    if not target:
        return None
    for m in available_models(cfg):
        # Unavailable entries can't be launched, so they can't be "what
        # the deploy is wired to". Without this, the Bedrock entry's
        # @attr:agent_model ref — equal to cfg.agent_model by
        # construction — claims 'active' on direct-API deploys that
        # have no Bedrock credentials at all.
        if not m.available:
            continue
        if m.model_ref == target:
            return m
    return None
