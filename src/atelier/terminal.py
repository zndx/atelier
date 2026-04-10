"""Terminal REPL — line-buffered session bridging to the Claude Agent SDK.

Each WebSocket connection gets a TerminalSession. Input is buffered
character-by-character; on Enter the completed line is dispatched to the
SDK via ``query()``. Tokens stream back as ``{"type": "text"}`` frames.

Graceful degradation: if the SDK is not installed or no credentials are
configured, the terminal still renders and responds to local commands.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

# ANSI helpers
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_CYAN = "\x1b[36m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"
_WHITE = "\x1b[97m"

_PROMPT = f"{_CYAN}\u25b8{_RESET} "

_HELP_TEXT = f"""\r
{_BOLD}Commands{_RESET}
  {_WHITE}help{_RESET}      {_DIM}Show this message{_RESET}
  {_WHITE}status{_RESET}    {_DIM}SDK, credentials, and model info{_RESET}
  {_WHITE}clear{_RESET}     {_DIM}Clear the screen{_RESET}

{_DIM}Anything else is sent to the Claude Agent SDK as a prompt.{_RESET}
{_DIM}Responses stream back in real time. Ctrl-C to interrupt.{_RESET}
"""


def _text(data: str) -> dict:
    return {"type": "text", "data": data}


class TerminalSession:
    """Line-buffered REPL backed by the Claude Agent SDK."""

    def __init__(self) -> None:
        self._line_buffer: list[str] = []
        self._session_id = str(uuid.uuid4())
        self._busy = False
        self._sdk_available: bool | None = None
        self._creds_available: bool | None = None

    def _check_sdk(self) -> bool:
        if self._sdk_available is None:
            try:
                import claude_agent_sdk  # noqa: F401
                self._sdk_available = True
            except ImportError:
                self._sdk_available = False
        return self._sdk_available

    def _check_creds(self) -> bool:
        if self._creds_available is None:
            try:
                from atelier.config import load_config
                cfg = load_config()
                self._creds_available = cfg.has_anthropic or cfg.has_bedrock
            except Exception:
                self._creds_available = False
        return self._creds_available

    def _get_model_name(self) -> str:
        try:
            from atelier.config import load_config
            return load_config().agent_model
        except Exception:
            return "unknown"

    def welcome(self) -> list[dict]:
        """Generate welcome banner frames."""
        has_sdk = self._check_sdk()
        has_creds = self._check_creds()
        model = self._get_model_name() if has_sdk else ""

        lines = ["\r\n"]

        # Header
        lines.append(f"  {_BOLD}{_WHITE}Atelier Terminal{_RESET}\r\n")
        lines.append(f"  {_DIM}Claude Agent SDK \u2022 Interactive session{_RESET}\r\n")
        lines.append(f"  {_DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{_RESET}\r\n")

        if not has_sdk:
            lines.append(f"  {_RED}\u2717{_RESET} SDK not installed ")
            lines.append(f"{_DIM}(pip install atelier[agents]){_RESET}\r\n")
        elif not has_creds:
            lines.append(f"  {_GREEN}\u2713{_RESET} SDK installed  ")
            lines.append(f"  {_RED}\u2717{_RESET} No credentials ")
            lines.append(f"{_DIM}\u2014 visit /status{_RESET}\r\n")
        else:
            lines.append(f"  {_GREEN}\u2713{_RESET} {model}  ")
            lines.append(f"{_GREEN}\u2713{_RESET} credentials configured\r\n")

        lines.append(f"  {_DIM}Type 'help' for commands{_RESET}\r\n")
        lines.append(f"\r\n{_PROMPT}")
        return [_text("".join(lines))]

    async def feed(self, data: str) -> AsyncIterator[dict]:
        """Process terminal input character-by-character."""
        for ch in data:
            if ch in ("\r", "\n"):
                line = "".join(self._line_buffer).strip()
                self._line_buffer.clear()
                yield _text("\r\n")

                if line:
                    async for frame in self._dispatch(line):
                        yield frame

                yield _text(_PROMPT)

            elif ch in ("\x7f", "\x08"):  # Backspace
                if self._line_buffer:
                    self._line_buffer.pop()
                    yield _text("\x08 \x08")

            elif ch == "\x03":  # Ctrl-C
                if self._busy:
                    self._busy = False
                    yield _text(f"^C\r\n{_DIM}(interrupted){_RESET}\r\n{_PROMPT}")
                else:
                    self._line_buffer.clear()
                    yield _text(f"^C\r\n{_PROMPT}")

            elif ch >= " " or ch == "\t":  # Printable
                self._line_buffer.append(ch)
                yield _text(ch)

    async def _dispatch(self, line: str) -> AsyncIterator[dict]:
        """Dispatch a completed line."""
        cmd = line.lower().strip()

        if cmd == "help":
            yield _text(_HELP_TEXT)
            return

        if cmd == "clear":
            yield _text("\x1b[2J\x1b[H")  # Clear screen + cursor home
            return

        if cmd == "status":
            async for frame in self._status():
                yield frame
            return

        # Everything else goes to the SDK
        async for frame in self._query_sdk(line):
            yield frame

    async def _status(self) -> AsyncIterator[dict]:
        """Show SDK and credential status."""
        has_sdk = self._check_sdk()
        has_creds = self._check_creds()

        sdk_icon = f"{_GREEN}\u2713{_RESET}" if has_sdk else f"{_RED}\u2717{_RESET}"
        cred_icon = f"{_GREEN}\u2713{_RESET}" if has_creds else f"{_RED}\u2717{_RESET}"

        lines = [f"\r\n  {_BOLD}{_WHITE}Status{_RESET}\r\n"]
        lines.append(f"  {_DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{_RESET}\r\n")
        lines.append(f"  {sdk_icon} SDK         {'installed' if has_sdk else 'not installed'}\r\n")
        lines.append(f"  {cred_icon} Credentials {'configured' if has_creds else 'not configured'}\r\n")

        try:
            from atelier.config import load_config
            cfg = load_config()
            lines.append(f"  \u2022 Model       {_DIM}{cfg.agent_model}{_RESET}\r\n")
            providers = []
            if cfg.has_anthropic:
                providers.append("Anthropic")
            if cfg.has_bedrock:
                providers.append("Bedrock")
            if providers:
                lines.append(f"  \u2022 Providers   {_DIM}{', '.join(providers)}{_RESET}\r\n")
        except Exception:
            pass

        lines.append(f"  \u2022 Session     {_DIM}{self._session_id[:8]}...{_RESET}\r\n")
        yield _text("".join(lines))

    async def _query_sdk(self, prompt: str) -> AsyncIterator[dict]:
        """Send prompt to Claude Agent SDK and stream response."""
        if not self._check_sdk():
            yield _text(
                f"{_RED}SDK not available.{_RESET} "
                f"{_DIM}Install with: pip install atelier[agents]{_RESET}\r\n"
            )
            return

        if not self._check_creds():
            yield _text(
                f"{_RED}No credentials configured.{_RESET} "
                f"{_DIM}Set ANTHROPIC_API_KEY or AWS credentials.{_RESET}\r\n"
            )
            return

        self._busy = True
        yield _text(f"{_DIM}thinking...{_RESET}")

        try:
            from pathlib import Path

            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                AssistantMessage,
                ResultMessage,
                TextBlock,
            )
            from atelier.config import load_config
            from atelier.agents.client import _build_sdk_env

            cfg = load_config()
            env = _build_sdk_env(cfg)

            # Load project-level .claude/commands so the interactive session
            # exposes our 9 keystone skills as slash commands.
            project_root = Path(__file__).resolve().parent.parent.parent

            options = ClaudeAgentOptions(
                allowed_tools=[],
                permission_mode="dontAsk",
                model=cfg.agent_model,
                max_turns=5,
                max_budget_usd=0.25,
                cwd=str(project_root),
                setting_sources=["project"],
                env=env,
            )

            # Clear "thinking..." line
            yield _text("\r\x1b[K")

            async for message in query(prompt=prompt, options=options):
                if not self._busy:
                    break  # Ctrl-C interrupted
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Convert LF to CR+LF for terminal display
                            text = block.text.replace("\n", "\r\n")
                            yield _text(text)
                elif isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", None)
                    duration = getattr(message, "duration_ms", None)
                    meta_parts = []
                    if duration is not None:
                        meta_parts.append(f"{duration}ms")
                    if cost is not None:
                        meta_parts.append(f"${cost:.4f}")
                    if meta_parts:
                        yield _text(
                            f"\r\n{_DIM}({', '.join(meta_parts)}){_RESET}\r\n"
                        )

        except asyncio.CancelledError:
            yield _text(f"\r\n{_DIM}(cancelled){_RESET}\r\n")
        except Exception as e:
            yield _text(f"\r\n{_RED}Error: {e}{_RESET}\r\n")
        finally:
            self._busy = False
