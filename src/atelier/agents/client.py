"""Claude Agent SDK client — credential validation and smoke test.

Supports multiple concurrent providers: direct Anthropic API, AWS Bedrock
(and eventually Vertex AI). Both can coexist — ANTHROPIC_API_KEY serves as
overwatch/bootstrap capability alongside production Bedrock credentials.

All functions take AtelierConfig to follow the project's config-loading
convention. Never access os.environ directly.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.config import AtelierConfig


def _build_anthropic_client(cfg: AtelierConfig):
    """Build direct Anthropic API client."""
    import anthropic
    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def _build_bedrock_client(cfg: AtelierConfig):
    """Build AWS Bedrock client.

    Uses the region embedded in the model ARN when present, falling back
    to ``cfg.aws_region``.  Cross-region inference profiles encode their
    target region in the ARN; without this the client connects to the
    default region and gets ``ResourceNotFoundException``.
    """
    import anthropic
    from atelier.config import region_from_arn

    region = region_from_arn(cfg.agent_model) or cfg.aws_region
    return anthropic.AnthropicBedrock(
        aws_access_key=cfg.aws_access_key_id,
        aws_secret_key=cfg.aws_secret_access_key,
        aws_region=region,
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

    Passes all available credentials — they don't conflict at the SDK
    layer. BUT the underlying ``claude`` CLI does NOT auto-detect
    Bedrock from model format; it requires the explicit
    ``CLAUDE_CODE_USE_BEDROCK=1`` flag, otherwise it tries to use a
    (nonexistent) login session and returns "Not logged in · Please
    run /login" with exit code 1.

    Provider selection rules:
    - If the model is a Bedrock ARN OR only bedrock creds are present,
      set ``CLAUDE_CODE_USE_BEDROCK=1``.
    - If only ``ANTHROPIC_API_KEY`` is present, leave the flag unset
      (CLI defaults to direct API).
    - When both are configured, prefer the direct API unless the model
      is explicitly a Bedrock ARN — that's the production signal.
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

    from atelier.config import is_bedrock_model, region_from_arn

    model_is_bedrock = is_bedrock_model(cfg.agent_model)
    prefer_bedrock = model_is_bedrock or (cfg.has_bedrock and not cfg.has_anthropic)
    if prefer_bedrock and cfg.has_bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        # Cross-region inference profiles embed the target region in the
        # ARN.  Prefer that over cfg.aws_region so the CLI connects to
        # the correct endpoint.
        arn_region = region_from_arn(cfg.agent_model)
        effective_region = arn_region or cfg.aws_region
        if effective_region:
            env["AWS_REGION"] = effective_region
            env["AWS_DEFAULT_REGION"] = effective_region

        # ── Bedrock sub-model pins ───────────────────────────────
        # The CLI internally uses haiku-sized models for tool search
        # and sonnet-sized models for subagent dispatch. Its direct-API
        # defaults do NOT resolve on Bedrock; without these pins,
        # internal calls fail silently and interactive sessions (with
        # tools, max_turns>1) hang or crash. If the operator hasn't
        # pinned per-size Bedrock model IDs, fall back to the primary
        # model so internal calls at least go to a known-good endpoint
        # (oversized is better than broken).
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = (
            cfg.agent_default_sonnet_model or cfg.agent_model
        )
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = (
            cfg.agent_default_haiku_model or cfg.agent_model
        )
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = (
            cfg.agent_subagent_model or cfg.agent_model
        )

    # ── CLI feature flags ────────────────────────────────────────
    # Precedence: explicit cfg value > provider-specific default.
    # On Bedrock, experimental betas / tool search are safe to disable
    # by default because they dispatch to direct-API sub-models that
    # won't resolve. On direct Anthropic we leave them untouched so
    # the CLI's native behavior wins.
    disable_betas = cfg.agent_disable_experimental_betas
    if disable_betas is None and prefer_bedrock and cfg.has_bedrock:
        disable_betas = "1"
    if disable_betas is not None:
        env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = disable_betas

    tool_search = cfg.agent_enable_tool_search
    if tool_search is None and prefer_bedrock and cfg.has_bedrock:
        tool_search = "false"
    if tool_search is not None:
        env["ENABLE_TOOL_SEARCH"] = tool_search

    return env


def run_smoke_test(cfg: AtelierConfig) -> dict:
    """Run a minimal Claude Agent SDK query to prove the pipeline works.

    Uses the SDK's query() function with max_turns=1 and a trivial prompt.
    Handles both sync and async calling contexts.

    Returns:
        {"success": True, "duration_ms": ..., "session_id": ..., ...} or
        {"success": False, "error": "..."}
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. FastAPI, behave).
        # Schedule the coroutine on the running loop and return a future.
        # The caller (gateway) must await the result.
        raise _NeedsAsync()
    return asyncio.run(_run_smoke_test_async(cfg))


class _NeedsAsync(Exception):
    """Sentinel: caller should use run_smoke_test_async() instead."""


async def run_smoke_test_async(cfg: AtelierConfig) -> dict:
    """Async version for use inside an event loop (FastAPI endpoints)."""
    return await _run_smoke_test_async(cfg)


async def _run_smoke_test_async(cfg: AtelierConfig) -> dict:
    from pathlib import Path

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

    # Project root contains .claude/commands/ — explicit cwd + setting_sources
    # ensures the SDK discovers our 9 keystone skills at runtime. Without
    # setting_sources, the SDK defaults to None and silently skips filesystem
    # slash commands.
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    provider = "bedrock" if env.get("CLAUDE_CODE_USE_BEDROCK") else "anthropic"

    last_error: str | None = None

    # Retry once on cold-start failure. The claude CLI sometimes exits with
    # code 1 on its first invocation (session directory initialization, config
    # discovery, etc.) and succeeds on the immediate retry.
    for attempt in range(2):
        stderr_lines: list[str] = []

        def _capture_stderr(line: str) -> None:
            stderr_lines.append(line)

        options = ClaudeAgentOptions(
            allowed_tools=[],
            permission_mode="dontAsk",
            model=cfg.agent_model,
            max_turns=1,
            max_budget_usd=0.05,
            cwd=str(project_root),
            setting_sources=["project"],
            env=env,
            stderr=_capture_stderr,
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
            return {
                "success": True,
                "reply": " ".join(texts).strip(),
                "retried": attempt > 0,
                **result_meta,
            }
        except Exception as e:
            exit_code = getattr(e, "exit_code", None)
            detail = f"{type(e).__name__}: {e}"
            if exit_code is not None:
                detail += f" (exit {exit_code})"
            if stderr_lines:
                tail = stderr_lines[-20:]
                detail += "\n\nstderr:\n" + "\n".join(tail)
            last_error = detail

            if attempt == 0:
                await asyncio.sleep(1)

    return {
        "success": False,
        "error": last_error,
        "provider": provider,
        "model": cfg.agent_model,
    }


# ── Model discovery ─────────────────────────────────────────────

_model_cache: tuple[float, dict] | None = None
_CACHE_TTL = 3600  # 1 hour


def check_model_upgrade(cfg: AtelierConfig) -> dict:
    """Check if a newer model is available in the same family.

    Requires ``has_anthropic`` (``models.list()`` needs direct API key).
    Results are cached for 1 hour.
    """
    global _model_cache

    if _model_cache is not None:
        cached_at, cached_result = _model_cache
        if time.monotonic() - cached_at < _CACHE_TTL:
            return {**cached_result, "cached": True}

    result = _do_model_discovery(cfg)
    _model_cache = (time.monotonic(), result)
    return {**result, "cached": False}


def _do_model_discovery(cfg: AtelierConfig) -> dict:
    """Perform model discovery (uncached)."""
    from atelier.config import extract_anthropic_model_id, extract_model_family

    current_id = extract_anthropic_model_id(cfg.agent_model)
    family = extract_model_family(current_id)
    is_arn = cfg.agent_model.startswith("arn:") or "anthropic." in cfg.agent_model

    base = {
        "current_model": current_id,
        "current_family": family,
        "source": "bedrock_arn" if is_arn else "direct",
    }

    if not cfg.has_anthropic:
        return {**base, "upgrade_available": False, "reason": "no_anthropic_key"}

    if family is None:
        return {**base, "upgrade_available": False, "reason": "unknown_family"}

    try:
        client = _build_anthropic_client(cfg)
        page = client.models.list(limit=100)
    except Exception as e:
        return {**base, "upgrade_available": False, "reason": "api_error", "error": str(e)}

    # Filter to same family, sort by created_at descending
    family_models = [
        m for m in page.data
        if extract_model_family(m.id) == family
    ]
    family_models.sort(key=lambda m: m.created_at, reverse=True)

    if not family_models:
        return {**base, "upgrade_available": False, "reason": "no_family_models"}

    latest = family_models[0]

    if latest.id == current_id:
        return {**base, "upgrade_available": False, "reason": "already_latest"}

    # Check if latest is actually newer
    current_info = next((m for m in page.data if m.id == current_id), None)
    if current_info and latest.created_at <= current_info.created_at:
        return {**base, "upgrade_available": False, "reason": "already_latest"}

    return {
        **base,
        "upgrade_available": True,
        "latest_model": latest.id,
        "latest_display_name": latest.display_name,
        "latest_created_at": latest.created_at.isoformat()
        if hasattr(latest.created_at, "isoformat")
        else str(latest.created_at),
    }
