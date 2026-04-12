"""Terminal REPL — line-buffered session bridging to the Claude Agent SDK.

Each WebSocket connection gets a TerminalSession. Input is buffered
character-by-character; on Enter the completed line is dispatched to
the SDK via ``query()`` as a *background task*. Ctrl-C cancels the
running task, giving operators Claude Code-style pause/redirect mid
query.

Graceful degradation: if the SDK is not installed or no credentials
are configured, the terminal still renders and responds to local
commands.

Architecture note
-----------------
Frames are emitted via an async callback registered by the gateway
(``set_emit``) rather than via an async generator. This matters
because the websocket read loop must stay concurrent with any
in-flight SDK query — if the read loop were blocked iterating an
``async for`` over the session, Ctrl-C bytes from the client would
never reach us until the query completed. With the callback pattern,
``handle_input`` is a regular async method that kicks off a task and
returns immediately, so the next ``receive_text()`` fires right away
and subsequent Ctrl-C can cancel the task in flight.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Awaitable, Callable

# ANSI helpers
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_CYAN = "\x1b[36m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"
_WHITE = "\x1b[97m"

_PROMPT = f"{_CYAN}\u25b8{_RESET} "

# Braille dots spinner frames — the same clockwise orbit used by
# Claude Code's thinking indicator (rattles `Dots` preset).
_SPINNER_FRAMES = ("\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f")
_SPINNER_INTERVAL = 0.08  # 80ms per frame

_HELP_TEXT = f"""\r
{_BOLD}Commands{_RESET}
  {_WHITE}help{_RESET}      {_DIM}Show this message{_RESET}
  {_WHITE}status{_RESET}    {_DIM}SDK, credentials, and model info{_RESET}
  {_WHITE}clear{_RESET}     {_DIM}Clear the screen{_RESET}

{_DIM}Anything else is sent to the Claude Agent SDK as a prompt.{_RESET}
{_DIM}Press Ctrl-C to interrupt a running query.{_RESET}
"""


EmitFn = Callable[[dict], Awaitable[None]]


def _text(data: str) -> dict:
    return {"type": "text", "data": data}


class TerminalSession:
    """Line-buffered REPL backed by the Claude Agent SDK."""

    def __init__(self) -> None:
        self._line_buffer: list[str] = []
        self._session_id = str(uuid.uuid4())
        self._sdk_available: bool | None = None
        self._creds_available: bool | None = None
        self._emit: EmitFn | None = None
        self._current_task: asyncio.Task | None = None

    # ── Gateway wiring ───────────────────────────────────────────

    def set_emit(self, emit: EmitFn) -> None:
        """Register the async callback used to push frames to the client."""
        self._emit = emit

    @property
    def _busy(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    async def _send(self, data: str) -> None:
        if self._emit is not None:
            await self._emit(_text(data))

    async def shutdown(self) -> None:
        """Cancel any in-flight query and wait for it to unwind."""
        if self._busy:
            assert self._current_task is not None
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass

    # ── SDK / credential probes ──────────────────────────────────

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

    # ── Welcome banner ───────────────────────────────────────────

    def welcome(self) -> list[dict]:
        """Generate welcome banner frames (caller emits synchronously)."""
        has_sdk = self._check_sdk()
        has_creds = self._check_creds()
        model = self._get_model_name() if has_sdk else ""

        lines = ["\r\n"]
        lines.append(f"  {_BOLD}{_WHITE}Atelier Terminal{_RESET}\r\n")
        lines.append(f"  {_DIM}Claude Agent SDK \u2022 Interactive session{_RESET}\r\n")
        lines.append(f"  {_DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{_RESET}\r\n")

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

    # ── Input handling ───────────────────────────────────────────

    async def handle_input(self, data: str) -> None:
        """Process terminal input.

        Returns immediately after scheduling any SDK query, so the
        websocket read loop can pick up the next frame (including
        ^C) without waiting for the query to finish.

        While a query is running, only Ctrl-C is accepted — all
        other input is silently dropped so interleaved SDK output
        doesn't fight with a half-typed next prompt.
        """
        # ── Paste fast-path ──────────────────────────────────────
        # ghostty-web (and xterm.js) deliver pasted content as a
        # single onData() frame. When data has length > 1 and
        # contains a line break we preserve the multi-line structure
        # rather than dispatching on each \n like we do for
        # interactive input.
        if len(data) > 1 and ("\n" in data or "\r" in data):
            if self._busy:
                await self._send(
                    f"\r\n{_DIM}(busy \u2014 press Ctrl-C to interrupt){_RESET}\r\n"
                )
                return
            normalized = data.replace("\r\n", "\n").replace("\r", "\n")
            trailing_newline = normalized.endswith("\n")
            # Strip only the trailing newline — keep internal \n intact.
            body = normalized[:-1] if trailing_newline else normalized
            self._line_buffer.append(body)
            # Echo with CR+LF so the terminal renders the paste the
            # same way an interactive user would see it.
            await self._send(body.replace("\n", "\r\n"))
            if trailing_newline:
                line = "".join(self._line_buffer).strip()
                self._line_buffer.clear()
                await self._send("\r\n")
                if line:
                    await self._dispatch(line)
                else:
                    await self._send(_PROMPT)
            return

        for ch in data:
            if ch == "\x03":  # Ctrl-C — always handled, even while busy
                if self._busy:
                    assert self._current_task is not None
                    # Cancellation propagates into the SDK query
                    # coroutine; the task wrapper emits the
                    # interrupted notice and restores the prompt.
                    self._current_task.cancel()
                else:
                    self._line_buffer.clear()
                    await self._send(f"^C\r\n{_PROMPT}")
                continue

            if self._busy:
                # While a query is running, silently drop other
                # input. Otherwise keystrokes would interleave with
                # SDK output and confuse the user. They can type
                # their next prompt after Ctrl-C.
                continue

            if ch in ("\r", "\n"):
                line = "".join(self._line_buffer).strip()
                self._line_buffer.clear()
                await self._send("\r\n")
                if line:
                    await self._dispatch(line)
                else:
                    await self._send(_PROMPT)

            elif ch in ("\x7f", "\x08"):  # Backspace
                if self._line_buffer:
                    self._line_buffer.pop()
                    await self._send("\x08 \x08")

            elif ch >= " " or ch == "\t":  # Printable
                self._line_buffer.append(ch)
                await self._send(ch)

    # ── Dispatch ─────────────────────────────────────────────────

    async def _dispatch(self, line: str) -> None:
        """Dispatch a completed line.

        Local commands (help/clear/status) run inline and restore the
        prompt themselves. SDK queries are scheduled as background
        tasks via ``_run_query_task`` so the websocket read loop can
        keep receiving Ctrl-C bytes mid-query.
        """
        cmd = line.lower().strip()

        if cmd == "help":
            await self._send(_HELP_TEXT)
            await self._send(_PROMPT)
            return

        if cmd == "clear":
            await self._send("\x1b[2J\x1b[H")  # Clear screen + cursor home
            await self._send(_PROMPT)
            return

        if cmd == "status":
            await self._status()
            await self._send(_PROMPT)
            return

        # Everything else goes to the SDK as a background task.
        self._current_task = asyncio.create_task(self._run_query_task(line))

    async def _run_query_task(self, line: str) -> None:
        """Wrapper that runs the SDK query and always restores the prompt.

        Splitting this out from ``_query_sdk`` keeps the cancel/prompt
        logic in a single finally block regardless of how the query
        terminates (success, exception, or cancellation).
        """
        try:
            await self._query_sdk(line)
        except asyncio.CancelledError:
            try:
                await self._send(f"\r\n{_DIM}(interrupted){_RESET}\r\n")
            except Exception:
                pass
            raise
        finally:
            try:
                await self._send(_PROMPT)
            except Exception:
                pass

    # ── Status command ───────────────────────────────────────────

    async def _status(self) -> None:
        """Show SDK and credential status."""
        has_sdk = self._check_sdk()
        has_creds = self._check_creds()

        sdk_icon = f"{_GREEN}\u2713{_RESET}" if has_sdk else f"{_RED}\u2717{_RESET}"
        cred_icon = f"{_GREEN}\u2713{_RESET}" if has_creds else f"{_RED}\u2717{_RESET}"

        lines = [f"\r\n  {_BOLD}{_WHITE}Status{_RESET}\r\n"]
        lines.append(f"  {_DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{_RESET}\r\n")
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
        await self._send("".join(lines))

    # ── Spinner ───────────────────────────────────────────────────

    async def _animate_spinner(
        self, label: str, stop: asyncio.Event
    ) -> None:
        """Cycle braille spinner frames until *stop* is set.

        Each frame overwrites the current line via ``\\r … \\x1b[K``.
        The caller signals *stop* when real content arrives; the
        method then clears the spinner line and returns.
        """
        i = 0
        n = len(_SPINNER_FRAMES)
        try:
            while not stop.is_set():
                frame = _SPINNER_FRAMES[i % n]
                await self._send(
                    f"\r{_CYAN}{frame}{_RESET} {_DIM}{label}{_RESET}\x1b[K"
                )
                i += 1
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=_SPINNER_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await self._send("\r\x1b[K")
            except Exception:
                pass

    # ── SDK query ────────────────────────────────────────────────

    async def _query_sdk(self, prompt: str) -> None:
        """Send prompt to Claude Agent SDK and stream response.

        Runs inside the background task spawned by ``_dispatch``.
        Emits frames via ``self._send``. Cancellation propagates from
        ``_run_query_task`` — no manual busy-flag check is needed.
        """
        if not self._check_sdk():
            await self._send(
                f"{_RED}SDK not available.{_RESET} "
                f"{_DIM}Install with: pip install atelier[agents]{_RESET}\r\n"
            )
            return

        if not self._check_creds():
            await self._send(
                f"{_RED}No credentials configured.{_RESET} "
                f"{_DIM}Set ANTHROPIC_API_KEY or AWS credentials.{_RESET}\r\n"
            )
            return

        # Animated thinking spinner — cycles braille dots until the
        # first SDK message arrives, then clears the line.
        stop_spinner = asyncio.Event()
        spinner_task: asyncio.Task | None = asyncio.create_task(
            self._animate_spinner("thinking...", stop_spinner)
        )

        # Capture CLI stderr so we can actually tell the operator *why*
        # the subprocess died. Without this callback, claude-agent-sdk
        # lets the child's stderr drain to the parent process's stderr
        # and constructs ProcessError with a placeholder string
        # ("Check stderr output for details") — the operator sees the
        # placeholder and has no idea what happened. With the callback,
        # we accumulate real lines and surface them on failure.
        stderr_lines: list[str] = []

        def _capture_stderr(line: str) -> None:
            stderr_lines.append(line)

        async def _stop_spinner() -> None:
            nonlocal spinner_task
            if spinner_task is not None and not stop_spinner.is_set():
                stop_spinner.set()
                try:
                    await spinner_task
                except (asyncio.CancelledError, Exception):
                    pass
                spinner_task = None

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

            # Load project-level .claude/commands so the interactive
            # session exposes our 9 keystone skills as slash commands.
            project_root = Path(__file__).resolve().parent.parent.parent

            # ``bypassPermissions`` is the CLI's explicit "allow all
            # tools, never prompt" mode. The web terminal is an
            # authenticated operator surface so full bypass is
            # appropriate; no human is around to approve individual
            # tool calls anyway.
            #
            # No budget/turn caps on the interactive terminal — this
            # is an authenticated operator surface. A nominal 1250
            # USD budget is high enough to be effectively unlimited
            # for a single session; ``max_turns=None`` disables the
            # turn gate so tool-heavy prompts don't trip a default.
            options = ClaudeAgentOptions(
                allowed_tools=[],
                permission_mode="bypassPermissions",
                model=cfg.agent_model,
                max_turns=None,
                max_budget_usd=1250.0,
                cwd=str(project_root),
                setting_sources=["project"],
                env=env,
                stderr=_capture_stderr,
            )

            async for message in query(prompt=prompt, options=options):
                # Stop the spinner on the first message of any type.
                if spinner_task is not None:
                    await _stop_spinner()

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Convert LF to CR+LF for terminal display.
                            text = block.text.replace("\n", "\r\n")
                            await self._send(text)
                elif isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", None)
                    duration = getattr(message, "duration_ms", None)
                    meta_parts = []
                    if duration is not None:
                        meta_parts.append(f"{duration}ms")
                    if cost is not None:
                        meta_parts.append(f"${cost:.4f}")
                    if meta_parts:
                        await self._send(
                            f"\r\n{_DIM}({', '.join(meta_parts)}){_RESET}\r\n"
                        )

        except asyncio.CancelledError:
            # Let _run_query_task handle the user-facing notice so
            # the cancellation text lives in exactly one place.
            raise
        except Exception as e:
            # Surface the raw exception plus any stderr lines we
            # captured. ProcessError.exit_code is useful; its
            # ``stderr`` attribute is a hardcoded placeholder so we
            # ignore it in favor of our callback-collected lines.
            exit_code = getattr(e, "exit_code", None)
            header = f"Error: {type(e).__name__}: {e}"
            if exit_code is not None:
                header += f" (exit {exit_code})"
            await self._send(f"\r\n{_RED}{header}{_RESET}\r\n")
            if stderr_lines:
                # Show the last ~20 lines — enough for actionable
                # signal without flooding the terminal on runaway
                # stderr.
                tail = stderr_lines[-20:]
                await self._send(f"{_DIM}stderr:{_RESET}\r\n")
                for line in tail:
                    await self._send(f"{_DIM}  {line}{_RESET}\r\n")
        finally:
            # Always clean up the spinner — catches early import
            # errors, cancellation, and any unexpected exit path.
            await _stop_spinner()
