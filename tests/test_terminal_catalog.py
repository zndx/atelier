"""Terminal model catalog — active-entry derivation semantics.

The Bedrock catalog entry resolves ``@attr:agent_model``, which equals
``cfg.agent_model`` by construction — so derivation must never let an
UNAVAILABLE entry claim 'active' (the direct-API-deploy misreport).
"""
from __future__ import annotations

from dataclasses import dataclass

from atelier.terminal_catalog import available_models, derive_from_agent_model


@dataclass
class _Cfg:
    agent_model: str = "claude-opus-4-8"
    has_bedrock: bool = False
    has_anthropic: bool = True


def test_direct_api_deploy_derives_anthropic_entry():
    # Bedrock unavailable: its @attr ref matches agent_model but must be
    # skipped; the anthropic apex entry is the deploy's actual wiring.
    active = derive_from_agent_model(_Cfg())
    assert active is not None
    assert active.id == "anthropic-opus-4-8"
    assert active.available


def test_bedrock_deploy_keeps_bedrock_entry():
    # With Bedrock creds + an ARN in agent_model, the bedrock entry is
    # both available and ref-matching — it correctly claims active.
    cfg = _Cfg(
        agent_model="arn:aws:bedrock:us-east-1::inference-profile/opus",
        has_bedrock=True,
    )
    active = derive_from_agent_model(cfg)
    assert active is not None
    assert active.provider == "bedrock"


def test_catalog_lists_latest_opus_first_among_anthropic():
    ids = [m.id for m in available_models(_Cfg())]
    assert ids.index("anthropic-opus-4-8") < ids.index("anthropic-opus-4-7")
