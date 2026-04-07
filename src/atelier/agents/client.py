"""Claude Agent SDK client — credential validation and smoke test.

Supports multiple concurrent providers: direct Anthropic API, AWS Bedrock
(and eventually Vertex AI). Both can coexist — ANTHROPIC_API_KEY serves as
overwatch/bootstrap capability alongside production Bedrock credentials.

All functions take AtelierConfig to follow the project's config-loading
convention. Never access os.environ directly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.config import AtelierConfig


def _build_anthropic_client(cfg: AtelierConfig):
    """Build direct Anthropic API client."""
    import anthropic
    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def _build_bedrock_client(cfg: AtelierConfig):
    """Build AWS Bedrock client."""
    import anthropic
    return anthropic.AnthropicBedrock(
        aws_access_key=cfg.aws_access_key_id,
        aws_secret_key=cfg.aws_secret_access_key,
        aws_region=cfg.aws_region,
        aws_session_token=cfg.aws_session_token,
    )


def _validate_single(client, model: str, provider: str) -> dict:
    """Validate a single provider with a minimal messages call."""
    import anthropic

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with only: ok"}],
        )
        text = resp.content[0].text if resp.content else ""
        return {
            "provider": provider,
            "valid": True,
            "model": resp.model,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            "reply": text.strip(),
        }
    except anthropic.AuthenticationError as e:
        return {"provider": provider, "valid": False, "error": f"Authentication failed: {e}"}
    except anthropic.APIError as e:
        return {"provider": provider, "valid": False, "error": f"API error: {e}"}
    except Exception as e:
        return {"provider": provider, "valid": False, "error": str(e)}


def validate_credentials(cfg: AtelierConfig) -> dict:
    """Validate all configured providers.

    Tests every provider that has credentials present.
    Returns a dict with per-provider results and an overall summary.

    Returns::

        {
            "providers": {
                "anthropic": {"valid": True, "model": "...", ...},
                "bedrock": {"valid": True, "model": "...", ...},
            },
            "any_valid": True,
            "configured": ["anthropic", "bedrock"],
        }
    """
    providers: dict[str, dict] = {}

    if cfg.has_anthropic:
        client = _build_anthropic_client(cfg)
        providers["anthropic"] = _validate_single(client, cfg.agent_model, "anthropic")

    if cfg.has_bedrock:
        client = _build_bedrock_client(cfg)
        providers["bedrock"] = _validate_single(client, cfg.agent_model, "bedrock")

    if not providers:
        return {
            "providers": {},
            "any_valid": False,
            "configured": [],
            "error": "No credentials configured. Set ANTHROPIC_API_KEY and/or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY.",
        }

    return {
        "providers": providers,
        "any_valid": any(p["valid"] for p in providers.values()),
        "configured": list(providers.keys()),
    }


# Keep backward-compatible alias used by gateway and BDD
validate_api_key = validate_credentials


def _build_sdk_env(cfg: AtelierConfig) -> dict[str, str]:
    """Build environment dict for ClaudeAgentOptions.

    Passes all available credentials — they don't conflict.
    The SDK / CLI resolves which provider to use based on model format.
    """
    env: dict[str, str] = {}
    if cfg.anthropic_api_key:
        env["ANTHROPIC_API_KEY"] = cfg.anthropic_api_key
    if cfg.aws_access_key_id:
        env["AWS_ACCESS_KEY_ID"] = cfg.aws_access_key_id
    if cfg.aws_secret_access_key:
        env["AWS_SECRET_ACCESS_KEY"] = cfg.aws_secret_access_key
    if cfg.aws_region:
        env["AWS_REGION"] = cfg.aws_region
    if cfg.aws_session_token:
        env["AWS_SESSION_TOKEN"] = cfg.aws_session_token
    return env


def run_smoke_test(cfg: AtelierConfig) -> dict:
    """Run a minimal Claude Agent SDK query to prove the pipeline works.

    Uses the SDK's query() function with max_turns=1 and a trivial prompt.

    Returns:
        {"success": True, "duration_ms": ..., "session_id": ..., ...} or
        {"success": False, "error": "..."}
    """
    return asyncio.run(_run_smoke_test_async(cfg))


async def _run_smoke_test_async(cfg: AtelierConfig) -> dict:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        AssistantMessage,
        ResultMessage,
        TextBlock,
    )

    env = _build_sdk_env(cfg)
    if not env:
        return {"success": False, "error": "No credentials configured for any provider"}

    options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="dontAsk",
        model=cfg.agent_model,
        max_turns=1,
        max_budget_usd=0.05,
        env=env,
    )

    texts: list[str] = []
    result_meta: dict = {}

    try:
        async for message in query(
            prompt="Reply with exactly: Atelier agent SDK operational",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_meta = {
                    "duration_ms": message.duration_ms,
                    "num_turns": message.num_turns,
                    "session_id": message.session_id,
                    "total_cost_usd": message.total_cost_usd,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "reply": " ".join(texts).strip(),
        **result_meta,
    }
