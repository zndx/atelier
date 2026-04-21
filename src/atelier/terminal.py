"""Terminal REPL — persistent session bridging to the Claude Agent SDK.

Sessions survive WebSocket disconnects (page navigation, reload).
The gateway connects clients to sessions via ``/ws/terminal/{session_id}``;
on disconnect the session stays alive and output accumulates in a ring
buffer. On reconnect the buffer is replayed so the user sees everything
that happened while they were away.

Input is buffered character-by-character; on Enter the completed line is
dispatched to the SDK via ``query()`` as a *background task*. Ctrl-C
cancels the running task, giving operators Claude Code-style pause/redirect
mid query.

Graceful degradation: if the SDK is not installed or no credentials
are configured, the terminal still renders and responds to local
commands.

Architecture note
-----------------
Frames are emitted via an async callback registered by the gateway
(``set_emit`` / ``attach``) rather than via an async generator. This
matters because the websocket read loop must stay concurrent with any
in-flight SDK query — if the read loop were blocked iterating an
``async for`` over the session, Ctrl-C bytes from the client would
never reach us until the query completed. With the callback pattern,
``handle_input`` is a regular async method that kicks off a task and
returns immediately, so the next ``receive_text()`` fires right away
and subsequent Ctrl-C can cancel the task in flight.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
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
_YELLOW = "\x1b[33m"

# ── Inline annotation prefixes ───────────────────────────────
_TOOL_PREFIX = f"{_DIM}  \u2699 "        # ⚙ tool call
_TOOL_ERR_PREFIX = f"{_RED}  \u2717 "    # ✗ tool error
_THINK_PREFIX = f"{_DIM}  \U0001f4ad "   # 💭 thinking
_AGENT_PREFIX = f"{_CYAN}  \u25c8 "      # ◈ agent/task event
_DONE_PREFIX = f"{_DIM}  \u2713 "        # ✓ done summary


def _format_tool_summary(name: str, tool_input: dict) -> str:
    """Build a compact one-line summary for a tool invocation."""
    if name in ("Read", "read"):
        path = tool_input.get("file_path", "")
        offset = tool_input.get("offset")
        suffix = f" (lines {offset}-{offset + tool_input.get('limit', 200)})" if offset else ""
        return f"{path}{suffix}"
    if name in ("Edit", "edit"):
        return tool_input.get("file_path", "")
    if name in ("Write", "write"):
        return tool_input.get("file_path", "")
    if name in ("Bash", "bash"):
        cmd = tool_input.get("command", "")
        return cmd[:60] + ("\u2026" if len(cmd) > 60 else "")
    if name in ("Grep", "grep"):
        pat = tool_input.get("pattern", "")
        path = tool_input.get("path", ".")
        return f'"{pat}" in {path}'
    if name in ("Glob", "glob"):
        return tool_input.get("pattern", "")
    if name in ("Agent", "agent"):
        desc = tool_input.get("description", "")
        return desc
    return name


def _format_tokens(n: int) -> str:
    """Format token count as compact string (e.g. 12.4k, 1.2M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# Braille dots spinner frames — the same clockwise orbit used by
# Claude Code's thinking indicator (rattles `Dots` preset).
_SPINNER_FRAMES = ("\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f")
_SPINNER_INTERVAL = 0.08  # 80ms per frame

_HELP_TEXT = f"""\r
{_BOLD}Commands{_RESET}
  {_WHITE}help{_RESET}      {_DIM}Show this message{_RESET}
  {_WHITE}status{_RESET}    {_DIM}SDK, credentials, and model info{_RESET}
  {_WHITE}chk{_RESET}       {_DIM}Discovery sitrep — probe all reachable services{_RESET}
  {_WHITE}clear{_RESET}     {_DIM}Clear the screen{_RESET}

{_DIM}Anything else is sent to the Claude Agent SDK as a prompt.{_RESET}
{_DIM}Press Ctrl-C to interrupt a running query.{_RESET}
"""


EmitFn = Callable[[dict], Awaitable[None]]


def _text(data: str) -> dict:
    return {"type": "text", "data": data}


class TerminalSession:
    """Persistent line-buffered REPL backed by the Claude Agent SDK.

    Sessions survive WebSocket disconnects. Output is buffered in a ring
    buffer and replayed on reconnect via :meth:`attach`.
    """

    def __init__(self, session_id: str | None = None) -> None:
        # ── Line editor state ──────────────────────────────────────
        # ``_line_buffer`` holds one character per slot so cursor
        # arithmetic stays O(1).  Every edit goes through the helpers
        # below so echo to the client and the internal model stay in
        # sync — bypassing the helpers breaks cursor tracking after a
        # mid-line insert or kill.
        self._line_buffer: list[str] = []
        self._cursor: int = 0
        # Command history — most recent last, capped at HISTORY_LIMIT.
        # Up/Down arrows browse; typing any edit exits browse mode.
        self._history: list[str] = []
        self._history_pos: int | None = None
        self._saved_edit: list[str] = []
        self._session_id = session_id or str(uuid.uuid4())
        self._sdk_available: bool | None = None
        self._creds_available: bool | None = None
        self._emit: EmitFn | None = None
        self._current_task: asyncio.Task | None = None
        self._output_buffer: deque[str] = deque(maxlen=65536)
        self._detached_at: float | None = None
        # SDK conversation continuity — track whether we've had a
        # successful query so subsequent prompts continue the conversation.
        self._sdk_session_id: str | None = None
        self._has_conversed: bool = False

    # ── Gateway wiring ───────────────────────────────────────────

    def set_emit(self, emit: EmitFn) -> None:
        """Register the async callback used to push frames to the client."""
        self._emit = emit
        self._detached_at = None

    def detach(self) -> None:
        """Detach emit callback (WS disconnected). Session stays alive."""
        self._emit = None
        self._detached_at = time.monotonic()

    def attach(self, emit: EmitFn) -> list[dict]:
        """Re-attach emit callback and return buffered output for replay.

        If the buffer is non-empty, returns a single frame containing all
        buffered output. Otherwise returns the welcome banner.
        """
        self._emit = emit
        self._detached_at = None
        if self._output_buffer:
            replay = "".join(self._output_buffer)
            return [_text(replay)]
        return self.welcome()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def _busy(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    @property
    def idle_seconds(self) -> float:
        """Seconds since last client disconnected. 0.0 if connected."""
        if self._detached_at is None:
            return 0.0
        return time.monotonic() - self._detached_at

    async def _send(self, data: str) -> None:
        """Buffer output and forward to connected client (if any)."""
        self._output_buffer.append(data)
        if self._emit is not None:
            try:
                await self._emit(_text(data))
            except Exception:
                # Client disconnected mid-send — detach silently
                self._emit = None

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
            from atelier.terminal_selection import active_model_ref
            return active_model_ref(load_config())
        except Exception:
            return "unknown"

    # ── Welcome banner ───────────────────────────────────────────

    def welcome(self) -> list[dict]:
        """Generate welcome banner frames (caller emits synchronously)."""
        has_sdk = self._check_sdk()
        has_creds = self._check_creds()
        model = self._get_model_name() if has_sdk else ""

        # Clear the screen + home cursor so a re-attach after a
        # terminal-model change doesn't overlay the new banner on top
        # of a stale one (observed: a new 'claude-opus-4-7' banner
        # rendered over an older Bedrock-ARN banner, leaving the tail
        # of the ARN visible as garble).
        lines = ["\x1b[2J\x1b[H"]
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

    # ── Line editor primitives ───────────────────────────────────
    #
    # Every edit goes through these helpers so the internal line buffer
    # and the on-wire cursor position stay consistent.  Echoes use
    # ANSI CSI sequences (``\x1b[…``) understood by every VT100-ish
    # terminal emulator — ghostty-web, xterm.js, an embedded tmux pane.
    #
    # Cursor model: ``_cursor`` is the index where the next typed char
    # would land (0..len(_line_buffer)).  The client's visible cursor
    # must always match this value after every helper returns.

    _HISTORY_LIMIT = 500

    def _exit_history_browse(self) -> None:
        """Commit the current line as the user's own edit.

        Called whenever the user edits (insert, backspace, delete, kill)
        while browsing history.  After this, Up arrow starts a new
        browse from the most recent entry rather than resuming mid-browse.
        """
        self._history_pos = None
        self._saved_edit = []

    async def _emit_insert(self, ch: str) -> None:
        """Insert ``ch`` at cursor; redraw the tail if present."""
        self._line_buffer.insert(self._cursor, ch)
        self._cursor += 1
        tail = "".join(self._line_buffer[self._cursor:])
        if tail:
            await self._send(ch + tail + f"\x1b[{len(tail)}D")
        else:
            await self._send(ch)

    async def _emit_backspace(self) -> None:
        """Delete the char *before* the cursor (classic backspace)."""
        if self._cursor == 0:
            return
        self._cursor -= 1
        del self._line_buffer[self._cursor]
        tail = "".join(self._line_buffer[self._cursor:])
        await self._send("\b\x1b[K" + tail)
        if tail:
            await self._send(f"\x1b[{len(tail)}D")

    async def _emit_delete(self) -> None:
        """Delete the char *at* the cursor (the Delete key / ``^D`` on
        some layouts).  No-op when the cursor sits at end of line."""
        if self._cursor >= len(self._line_buffer):
            return
        del self._line_buffer[self._cursor]
        tail = "".join(self._line_buffer[self._cursor:])
        await self._send("\x1b[K" + tail)
        if tail:
            await self._send(f"\x1b[{len(tail)}D")

    async def _move_cursor(self, delta: int) -> None:
        """Move cursor by ``delta`` (+ right, − left), clamped to [0, len]."""
        new_pos = max(0, min(len(self._line_buffer), self._cursor + delta))
        actual = new_pos - self._cursor
        if actual == 0:
            return
        if actual > 0:
            await self._send(f"\x1b[{actual}C")
        else:
            await self._send(f"\x1b[{-actual}D")
        self._cursor = new_pos

    async def _cursor_to(self, pos: int) -> None:
        """Absolute cursor move — used by Home/End and Ctrl-A/E."""
        await self._move_cursor(pos - self._cursor)

    async def _kill_to_end(self) -> None:
        """Ctrl-K — delete from cursor to end of line."""
        if self._cursor >= len(self._line_buffer):
            return
        del self._line_buffer[self._cursor:]
        await self._send("\x1b[K")

    async def _kill_to_start(self) -> None:
        """Ctrl-U — delete from cursor to start of line."""
        if self._cursor == 0:
            return
        killed = self._cursor
        del self._line_buffer[:self._cursor]
        self._cursor = 0
        # Backspace over the killed region, clear to EOL, redraw tail.
        await self._send("\b" * killed + "\x1b[K")
        tail = "".join(self._line_buffer)
        if tail:
            await self._send(tail + f"\x1b[{len(tail)}D")

    async def _delete_word_back(self) -> None:
        """Ctrl-W — delete one whitespace-delimited word before cursor.

        Matches bash/readline: skip trailing whitespace first, then
        non-whitespace, so ``foo bar `` with cursor at end deletes
        ``bar `` (leaving ``foo ``) rather than the trailing space alone.
        """
        if self._cursor == 0:
            return
        j = self._cursor
        while j > 0 and self._line_buffer[j - 1].isspace():
            j -= 1
        while j > 0 and not self._line_buffer[j - 1].isspace():
            j -= 1
        if j == self._cursor:
            return
        killed = self._cursor - j
        del self._line_buffer[j:self._cursor]
        self._cursor = j
        await self._send("\b" * killed + "\x1b[K")
        tail = "".join(self._line_buffer[self._cursor:])
        if tail:
            await self._send(tail + f"\x1b[{len(tail)}D")

    async def _replace_line(self, new_line: str) -> None:
        """Replace the entire current line with ``new_line``.

        Used by history navigation (Up/Down) and any other operation
        that wants to swap the whole input.  Cursor ends at the end of
        the new content (matches readline's history-recall behavior).
        """
        if self._cursor:
            await self._send("\b" * self._cursor)
        await self._send("\x1b[K")
        self._line_buffer = list(new_line)
        self._cursor = len(new_line)
        if new_line:
            await self._send(new_line)

    async def _history_prev(self) -> None:
        """Up arrow — walk backwards through history."""
        if not self._history:
            return
        if self._history_pos is None:
            # Starting a browse — stash the in-progress edit so Down
            # arrow past the newest entry can restore it.
            self._saved_edit = list(self._line_buffer)
            self._history_pos = len(self._history) - 1
        elif self._history_pos > 0:
            self._history_pos -= 1
        else:
            return  # already at oldest — no wraparound
        await self._replace_line(self._history[self._history_pos])

    async def _history_next(self) -> None:
        """Down arrow — walk forwards; past the newest entry restores
        whatever the user was typing before they started browsing."""
        if self._history_pos is None:
            return
        if self._history_pos < len(self._history) - 1:
            self._history_pos += 1
            await self._replace_line(self._history[self._history_pos])
        else:
            saved = "".join(self._saved_edit)
            self._history_pos = None
            self._saved_edit = []
            await self._replace_line(saved)

    def _record_history(self, line: str) -> None:
        """Append ``line`` to history, deduping against the last entry."""
        if not line:
            return
        if self._history and self._history[-1] == line:
            return
        self._history.append(line)
        if len(self._history) > self._HISTORY_LIMIT:
            # Trim from the oldest end so browse order is preserved.
            del self._history[: len(self._history) - self._HISTORY_LIMIT]

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
        # interactive input.  Paste always lands at end-of-line to
        # keep the cursor model simple — embedded \n in the payload
        # breaks the single-line invariant anyway.
        if len(data) > 1 and ("\n" in data or "\r" in data):
            if self._busy:
                await self._send(
                    f"\r\n{_DIM}(busy \u2014 press Ctrl-C to interrupt){_RESET}\r\n"
                )
                return
            self._exit_history_browse()
            if self._cursor < len(self._line_buffer):
                await self._cursor_to(len(self._line_buffer))
            normalized = data.replace("\r\n", "\n").replace("\r", "\n")
            trailing_newline = normalized.endswith("\n")
            body = normalized[:-1] if trailing_newline else normalized
            # One char per buffer slot so subsequent cursor arithmetic works.
            self._line_buffer.extend(body)
            self._cursor = len(self._line_buffer)
            await self._send(body.replace("\n", "\r\n"))
            if trailing_newline:
                line = "".join(self._line_buffer).strip()
                self._line_buffer.clear()
                self._cursor = 0
                self._exit_history_browse()
                await self._send("\r\n")
                if line:
                    self._record_history(line)
                    await self._dispatch(line)
                else:
                    await self._send(_PROMPT)
            return

        i = 0
        while i < len(data):
            ch = data[i]

            if ch == "\x03":  # Ctrl-C — always handled, even while busy
                if self._busy:
                    assert self._current_task is not None
                    self._current_task.cancel()
                else:
                    self._line_buffer.clear()
                    self._cursor = 0
                    self._exit_history_browse()
                    await self._send(f"^C\r\n{_PROMPT}")
                i += 1
                continue

            # ── Escape sequences (arrow keys, Delete, Home, End) ──
            if ch == "\x1b" and i + 1 < len(data) and data[i + 1] == "[":
                # CSI sequence: \x1b[ followed by params + final byte
                j = i + 2
                while j < len(data) and data[j] in "0123456789;":
                    j += 1
                if j < len(data):
                    final = data[j]
                    params = data[i + 2 : j]
                    j += 1  # consume final byte
                else:
                    # Incomplete sequence — discard
                    i = len(data)
                    continue

                i = j

                if self._busy:
                    continue

                # Dispatch on final byte (with optional ``params`` for
                # the tilde-terminated navigation keys).
                if final == "A":       # ↑  — history back
                    await self._history_prev()
                elif final == "B":     # ↓  — history forward
                    await self._history_next()
                elif final == "C":     # →  — cursor right
                    await self._move_cursor(1)
                elif final == "D":     # ←  — cursor left
                    await self._move_cursor(-1)
                elif final == "H" or (final == "~" and params in ("1", "7")):
                    await self._cursor_to(0)
                elif final == "F" or (final == "~" and params in ("4", "8")):
                    await self._cursor_to(len(self._line_buffer))
                elif final == "~" and params == "3":
                    await self._emit_delete()
                # else: unknown CSI — ignore silently
                continue

            i += 1

            if self._busy:
                # While a query is running, silently drop other
                # input. Otherwise keystrokes would interleave with
                # SDK output and confuse the user.
                continue

            # ── Control-key shortcuts (readline-style) ───────────
            if ch == "\x01":          # Ctrl-A — start of line
                await self._cursor_to(0)
            elif ch == "\x05":        # Ctrl-E — end of line
                await self._cursor_to(len(self._line_buffer))
            elif ch == "\x0b":        # Ctrl-K — kill to end
                self._exit_history_browse()
                await self._kill_to_end()
            elif ch == "\x15":        # Ctrl-U — kill to start
                self._exit_history_browse()
                await self._kill_to_start()
            elif ch == "\x17":        # Ctrl-W — delete word back
                self._exit_history_browse()
                await self._delete_word_back()
            elif ch in ("\r", "\n"):
                line = "".join(self._line_buffer).strip()
                self._line_buffer.clear()
                self._cursor = 0
                self._exit_history_browse()
                await self._send("\r\n")
                if line:
                    self._record_history(line)
                    await self._dispatch(line)
                else:
                    await self._send(_PROMPT)
            elif ch in ("\x7f", "\x08"):  # Backspace
                self._exit_history_browse()
                await self._emit_backspace()
            elif ch == "\x1b":
                # Lone escape (no CSI follows) — ignore.
                pass
            elif ch >= " " or ch == "\t":  # Printable (or Tab)
                self._exit_history_browse()
                await self._emit_insert(ch)

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
            # Drop conversation continuity too — ``clear`` is the
            # natural "start fresh" affordance; operators shouldn't
            # have to restart the terminal to break out of a wedged
            # slash-command session.
            self._has_conversed = False
            self._sdk_session_id = None
            await self._send(_PROMPT)
            return

        if cmd == "status":
            await self._status()
            await self._send(_PROMPT)
            return

        # Discovery sitrep runs locally — probes the gateway + config
        # directly.  The SDK-dispatched ``/health-check`` slash command
        # was unreliable on Bedrock (the bundled CLI resolved it
        # client-side with zero assistant output); running the probes
        # in-process makes ``chk`` deterministic and independent of
        # model + SDK + CLI interactions.
        if cmd in ("chk", "health-check"):
            await self._health_check()
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
        if self._has_conversed:
            lines.append(f"  \u2022 Context     {_DIM}active (conversation continues across prompts){_RESET}\r\n")
        else:
            lines.append(f"  \u2022 Context     {_DIM}fresh (first query will start a new conversation){_RESET}\r\n")
        await self._send("".join(lines))

    # ── Discovery sitrep ─────────────────────────────────────────

    async def _health_check(self) -> None:
        """Run probes directly and render a sitrep — no SDK involved.

        Mirrors the probes in ``.claude/commands/health-check.md`` so
        the slash-command and the in-process command stay in sync on
        intent.  The in-process path is the reliable one: SDK dispatch
        of ``/health-check`` was returning 0-turn / 0-duration silent
        completions on Bedrock 4.6 because the bundled CLI resolved
        the command client-side without hitting the API.
        """
        import asyncio as _asyncio
        import json as _json
        from urllib.request import urlopen

        gateway = "http://localhost:8090"

        def _probe_json(path: str, timeout: float = 5.0) -> dict | None:
            try:
                return _json.loads(urlopen(f"{gateway}{path}", timeout=timeout).read())
            except Exception:
                return None

        # Fan out probes in the thread pool — urllib is blocking and
        # we don't want the terminal to hang while they run.
        status_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/status"))
        sources_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/data-sources"))
        vocab_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/vocabulary/stats"))
        fsm_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/fsm/status"))
        runs_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/fsm/runs"))
        overwatch_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/overwatch/status"))
        gov_task = _asyncio.create_task(_asyncio.to_thread(_probe_json, "/api/governance/status", 10.0))

        status = await status_task
        sources = await sources_task
        vocab = await vocab_task
        fsm_state = await fsm_task
        runs = await runs_task
        overwatch = await overwatch_task
        governance = await gov_task

        try:
            from atelier.config import load_config
            cfg = load_config()
        except Exception:
            cfg = None
        try:
            from atelier import __version__
            version = __version__
        except Exception:
            version = "unknown"

        def _tick(ok: bool | None) -> str:
            if ok is True:
                return f"{_GREEN}\u2713{_RESET}"
            if ok is False:
                return f"{_RED}\u2717{_RESET}"
            return f"{_DIM}?{_RESET}"

        lines: list[str] = []
        lines.append(f"\r\n  {_BOLD}{_WHITE}Sitrep — Atelier v{version}{_RESET}\r\n")
        lines.append(f"  {_DIM}" + "\u2500" * 38 + f"{_RESET}\r\n\r\n")

        # Core services
        lines.append(f"  {_BOLD}Core services{_RESET}\r\n")
        if status:
            for svc, key in (("gRPC", "grpc"), ("PostgreSQL", "postgres"), ("Qdrant", "qdrant")):
                ok = status.get(key, {}).get("ok")
                lines.append(f"    {_tick(ok)} {svc:<12}{_RESET}\r\n")
        else:
            lines.append(f"    {_RED}\u2717{_RESET} gateway {_DIM}(unreachable at {gateway}){_RESET}\r\n")

        # Credentials
        lines.append(f"\r\n  {_BOLD}Credentials{_RESET}\r\n")
        if cfg is not None:
            lines.append(f"    {_tick(bool(cfg.has_anthropic))} Anthropic   {_DIM}{'configured' if cfg.has_anthropic else 'not configured'}{_RESET}\r\n")
            lines.append(f"    {_tick(bool(cfg.has_bedrock))} Bedrock     {_DIM}{'configured' if cfg.has_bedrock else 'not configured'}{_RESET}\r\n")
            lines.append(f"    \u2022 Agent model {_DIM}{cfg.agent_model}{_RESET}\r\n")
            lines.append(f"    {_tick(bool(cfg.has_classify_llm))} Classify LLM {_DIM}{'configured' if cfg.has_classify_llm else 'not configured'}{_RESET}\r\n")
        else:
            lines.append(f"    {_RED}\u2717{_RESET} config load failed\r\n")

        # Data sources + vocabulary
        lines.append(f"\r\n  {_BOLD}Data{_RESET}\r\n")
        if sources and isinstance(sources.get("sources"), list):
            src_list = sources["sources"]
            lines.append(f"    \u2022 Sources     {_DIM}{len(src_list)} registered{_RESET}\r\n")
            for s in src_list[:5]:
                sid = s.get("id", "?")
                stype = s.get("source_type", "?")
                lines.append(f"        - {sid} {_DIM}({stype}){_RESET}\r\n")
        else:
            lines.append(f"    {_DIM}\u2022 Sources     unavailable{_RESET}\r\n")
        if vocab:
            lines.append(f"    \u2022 Terms       {_DIM}{vocab.get('terms', 0)}{_RESET}\r\n")

        # Pipeline
        lines.append(f"\r\n  {_BOLD}Pipeline{_RESET}\r\n")
        if fsm_state:
            state = fsm_state.get("state", "UNKNOWN")
            rid = fsm_state.get("id", "")
            progress = fsm_state.get("progress") or {}
            lines.append(f"    \u2022 State       {_DIM}{state}{_RESET}\r\n")
            if rid:
                lines.append(f"    \u2022 Run ID      {_DIM}{rid[:8]}\u2026{_RESET}\r\n")
            acc = progress.get("accuracy")
            if acc is not None:
                lines.append(f"    \u2022 Accuracy    {_DIM}{acc}{_RESET}\r\n")
            cov = progress.get("bootstrap_coverage")
            if cov is not None:
                lines.append(f"    \u2022 Coverage    {_DIM}{cov}{_RESET}\r\n")
        else:
            lines.append(f"    {_DIM}\u2022 State       unavailable{_RESET}\r\n")
        if runs and isinstance(runs.get("runs"), list):
            lines.append(f"    \u2022 Recent runs {_DIM}{len(runs['runs'])}{_RESET}\r\n")

        # Overwatch
        lines.append(f"\r\n  {_BOLD}Overwatch{_RESET}\r\n")
        if overwatch:
            enabled = overwatch.get("enabled")
            lines.append(f"    {_tick(bool(enabled))} Enabled     {_DIM}{enabled}{_RESET}\r\n")
            autonomy = overwatch.get("autonomy")
            if autonomy:
                lines.append(f"    \u2022 Autonomy    {_DIM}{autonomy}{_RESET}\r\n")
            ow_model = overwatch.get("model")
            if ow_model:
                lines.append(f"    \u2022 Model       {_DIM}{ow_model}{_RESET}\r\n")
        else:
            lines.append(f"    {_DIM}\u2022 status      unavailable{_RESET}\r\n")

        # Governance
        lines.append(f"\r\n  {_BOLD}Governance{_RESET}\r\n")
        if governance and "error" not in governance:
            atlas = governance.get("atlas") or {}
            ranger = governance.get("ranger") or {}
            lines.append(f"    {_tick(bool(atlas.get('reachable')))} Atlas       {_DIM}{atlas.get('url', 'not configured')}{_RESET}\r\n")
            lines.append(f"    {_tick(bool(ranger.get('reachable')))} Ranger      {_DIM}{ranger.get('url', 'not configured')}{_RESET}\r\n")
        else:
            lines.append(f"    {_DIM}\u2022 status      unavailable{_RESET}\r\n")

        lines.append("\r\n")
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

        # ── Restartable spinner ───────────────────────────────────
        # The spinner persists across tool-use turns and only clears
        # when actual text output arrives.  Between text segments it
        # restarts with a contextual label ("thinking…", tool name, …).
        _stop_ev = asyncio.Event()
        _spinner_task: asyncio.Task | None = None

        async def _start_spinner(label: str = "thinking...") -> None:
            nonlocal _spinner_task, _stop_ev
            await _stop_spinner_inner()
            _stop_ev = asyncio.Event()
            _spinner_task = asyncio.create_task(
                self._animate_spinner(label, _stop_ev)
            )

        async def _stop_spinner_inner() -> None:
            nonlocal _spinner_task
            if _spinner_task is not None and not _stop_ev.is_set():
                _stop_ev.set()
                try:
                    await _spinner_task
                except (asyncio.CancelledError, Exception):
                    pass
            _spinner_task = None

        await _start_spinner()

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

        try:
            from pathlib import Path

            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                AssistantMessage,
                ResultMessage,
                TextBlock,
            )
            # Optional types — import what's available
            try:
                from claude_agent_sdk import (
                    ThinkingBlock,
                    ToolUseBlock,
                    ToolResultBlock,
                )
            except ImportError:
                ThinkingBlock = None  # type: ignore[assignment,misc]
                ToolUseBlock = None  # type: ignore[assignment,misc]
                ToolResultBlock = None  # type: ignore[assignment,misc]
            try:
                from claude_agent_sdk.types import (
                    TaskStartedMessage,
                    TaskProgressMessage,
                    TaskNotificationMessage,
                )
            except ImportError:
                TaskStartedMessage = None  # type: ignore[assignment,misc]
                TaskProgressMessage = None  # type: ignore[assignment,misc]
                TaskNotificationMessage = None  # type: ignore[assignment,misc]
            from atelier.config import load_config
            from atelier.agents.client import _build_sdk_env

            cfg = load_config()

            # Resolve the active model first — the picker can override
            # the deployment default, and the env-builder needs to see
            # the live selection to decide whether to set
            # CLAUDE_CODE_USE_BEDROCK. Without this, picking a direct-
            # API model on a deploy that has Bedrock creds still routes
            # through Bedrock (because cfg.agent_model is a Bedrock ARN).
            from atelier.model_compat import requires_adaptive_thinking
            from atelier.terminal_selection import get_active, active_model_ref
            active_entry = get_active(cfg)
            resolved_model = active_model_ref(cfg)

            env = _build_sdk_env(cfg, selected_model=resolved_model)

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
            # Opus 4.7+ (direct API) only accepts thinking.type=adaptive
            # + output_config.effort.  Opus 4.6 on Bedrock still uses
            # the legacy enabled+budget_tokens shape.
            #
            # SDK v0.1.56 has a latent bug: passing thinking={"type":
            # "adaptive"} causes it to emit `--max-thinking-tokens
            # 32000` to the bundled CLI (v2.1.92), which in turn sends
            # `thinking.type=enabled` to the API — rejected by 4.7.
            # Workaround: pass max_thinking_tokens=0 + effort=<level>
            # so the CLI emits `--max-thinking-tokens 0 --effort
            # <level>`, the API sees only `output_config.effort` and
            # 4.7 is happy.  Remove this workaround after upgrading
            # claude-agent-sdk to a release that fixes the mapping.
            # ``active_entry`` / ``resolved_model`` computed above
            # (before env-build) so provider selection sees the picker
            # choice; the catalog entry (when one matches) tells us
            # which rolling-stats bucket to write to when this query
            # completes.
            thinking_kwargs: dict = {}
            if requires_adaptive_thinking(resolved_model):
                thinking_kwargs["max_thinking_tokens"] = 0
                thinking_kwargs["effort"] = "medium"

            options = ClaudeAgentOptions(
                allowed_tools=[],
                permission_mode="bypassPermissions",
                model=resolved_model,
                max_turns=None,
                max_budget_usd=1250.0,
                cwd=str(project_root),
                setting_sources=["project"],
                env=env,
                stderr=_capture_stderr,
                # Conversation continuity — after the first query,
                # subsequent prompts continue the same conversation so
                # Claude retains context across terminal interactions.
                # NOTE: only pass continue_conversation (not session_id)
                # because the SDK CLI rejects --session-id + --continue
                # without --fork-session.
                #
                # Slash commands are explicit fresh-intent invocations;
                # chaining them onto a prior conversation makes the CLI
                # treat them as client-side no-ops in some cases
                # (observed with /health-check on Bedrock 4.6, which
                # returned 6 turns of 0ms with no assistant output).
                # Reset continuity when the prompt is a slash command.
                continue_conversation=(
                    self._has_conversed and not prompt.lstrip().startswith("/")
                ),
                **thinking_kwargs,
            )

            # Track state across the query for the summary line.
            emitted_text = False
            tools_used: list[str] = []
            tool_errors = 0
            is_thinking = False

            # TTFT / duration measurement — anchored at the start of
            # message iteration.  First AssistantMessage TextBlock
            # stamps the TTFT; ResultMessage branch below records the
            # final sample to the rolling-stats buffer.
            import time as _time
            query_started_at = _time.monotonic()
            ttft_ms: float | None = None

            async for message in query(prompt=prompt, options=options):
                # Capture session_id early for conversation continuity
                msg_sid = getattr(message, "session_id", None)
                if msg_sid and not self._sdk_session_id:
                    self._sdk_session_id = msg_sid

                if isinstance(message, AssistantMessage):
                    text_blocks = []
                    thinking_blocks = []
                    tool_use_blocks = []
                    tool_result_blocks = []
                    for b in message.content:
                        if isinstance(b, TextBlock):
                            text_blocks.append(b)
                        elif ThinkingBlock and isinstance(b, ThinkingBlock):
                            thinking_blocks.append(b)
                        elif ToolUseBlock and isinstance(b, ToolUseBlock):
                            tool_use_blocks.append(b)
                        elif ToolResultBlock and isinstance(b, ToolResultBlock):
                            tool_result_blocks.append(b)
                        elif hasattr(b, "name"):
                            tool_use_blocks.append(b)

                    # ── Thinking indicator ──
                    if thinking_blocks and not is_thinking:
                        is_thinking = True
                        await _stop_spinner_inner()
                        await self._send(
                            f"\r\n{_THINK_PREFIX}thinking\u2026{_RESET}\r\n"
                        )

                    # ── Tool call annotations ──
                    if tool_use_blocks:
                        await _stop_spinner_inner()
                        is_thinking = False
                        for tb in tool_use_blocks:
                            name = tb.name
                            inp = tb.input if hasattr(tb, "input") else {}
                            summary = _format_tool_summary(name, inp)
                            tools_used.append(name)
                            await self._send(
                                f"\r\n{_TOOL_PREFIX}{name}: {summary}{_RESET}\r\n"
                            )
                        await _start_spinner(f"{tool_use_blocks[-1].name}\u2026")

                    # ── Tool result errors ──
                    for tr in tool_result_blocks:
                        if getattr(tr, "is_error", False):
                            tool_errors += 1
                            err_text = str(getattr(tr, "content", ""))[:80]
                            await self._send(
                                f"\r\n{_TOOL_ERR_PREFIX}failed: {err_text}{_RESET}\r\n"
                            )

                    # ── Text output ──
                    if text_blocks:
                        await _stop_spinner_inner()
                        is_thinking = False
                        if ttft_ms is None:
                            # First assistant text block = TTFT anchor.
                            ttft_ms = (_time.monotonic() - query_started_at) * 1000.0
                        if emitted_text:
                            await self._send("\r\n\r\n")
                        for block in text_blocks:
                            text = block.text.replace("\n", "\r\n")
                            await self._send(text)
                        emitted_text = True

                    elif not tool_use_blocks and not thinking_blocks:
                        if _spinner_task is None or _stop_ev.is_set():
                            await _start_spinner()

                # ── Subagent / task events ──
                elif TaskStartedMessage and isinstance(message, TaskStartedMessage):
                    await _stop_spinner_inner()
                    desc = getattr(message, "description", "")
                    await self._send(
                        f"\r\n{_AGENT_PREFIX}Agent spawned: \"{desc}\"{_RESET}\r\n"
                    )
                    await _start_spinner("agent\u2026")

                elif TaskProgressMessage and isinstance(message, TaskProgressMessage):
                    last_tool = getattr(message, "last_tool_name", None)
                    usage = getattr(message, "usage", None)
                    parts = []
                    if last_tool:
                        parts.append(last_tool)
                    if usage:
                        parts.append(f"{getattr(usage, 'tool_uses', '?')} tools")
                        parts.append(f"{getattr(usage, 'duration_ms', 0)}ms")
                    if parts:
                        label = ", ".join(parts)
                        await _start_spinner(f"agent: {label}")

                elif TaskNotificationMessage and isinstance(message, TaskNotificationMessage):
                    await _stop_spinner_inner()
                    status = getattr(message, "status", "")
                    summary = (getattr(message, "summary", "") or "")[:80]
                    usage = getattr(message, "usage", None)
                    tok = ""
                    if usage:
                        total = getattr(usage, "total_tokens", 0)
                        if total:
                            tok = f", {_format_tokens(total)} tokens"
                    icon = "\u2713" if status == "completed" else "\u2717"
                    color = _CYAN if status == "completed" else _YELLOW
                    await self._send(
                        f"\r\n{color}  {icon} Agent {status}: {summary}{tok}{_RESET}\r\n"
                    )

                # ── Result summary ──
                elif isinstance(message, ResultMessage):
                    await _stop_spinner_inner()
                    # Capture session_id for conversation continuity
                    result_sid = getattr(message, "session_id", None)
                    if result_sid:
                        self._sdk_session_id = result_sid
                        self._has_conversed = True
                    duration = getattr(message, "duration_ms", None)
                    num_turns = getattr(message, "num_turns", None)
                    usage = getattr(message, "usage", None)

                    # Rolling-stats observation.  Record only when the
                    # active entry is in the catalog — "custom" agent_model
                    # values produce no catalog match and no stats bucket.
                    if active_entry is not None:
                        try:
                            from atelier.terminal_stats import record as _rec_stat
                            dur_ms = float(duration) if duration is not None else \
                                (_time.monotonic() - query_started_at) * 1000.0
                            inp_t = 0
                            out_t = 0
                            if isinstance(usage, dict):
                                inp_t = int(usage.get("input_tokens", 0) or 0)
                                out_t = int(usage.get("output_tokens", 0) or 0)
                            _rec_stat(
                                active_entry.provider, active_entry.id,
                                ttft_ms=ttft_ms if ttft_ms is not None else 0.0,
                                duration_ms=dur_ms,
                                input_tokens=inp_t,
                                output_tokens=out_t,
                            )
                        except Exception:
                            # Stats are advisory — never let a measurement
                            # failure break the user's terminal session.
                            pass

                    parts = []
                    if num_turns is not None and num_turns > 1:
                        parts.append(f"{num_turns} turns")
                    if usage:
                        inp = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
                        out = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
                        if inp or out:
                            parts.append(f"{_format_tokens(inp)} in / {_format_tokens(out)} out")
                        cache_read = usage.get("cache_read_input_tokens", 0) if isinstance(usage, dict) else 0
                        if cache_read:
                            parts.append(f"+{_format_tokens(cache_read)} cache")
                    if duration is not None:
                        secs = duration / 1000
                        parts.append(f"{secs:.1f}s" if secs < 60 else f"{secs / 60:.1f}m")
                    if tool_errors:
                        parts.append(f"{tool_errors} error{'s' if tool_errors > 1 else ''}")

                    summary = " \u00b7 ".join(parts) if parts else "done"

                    # Silent-run diagnostic: when the SDK reports turns
                    # happened but nothing flowed to the terminal (no
                    # text, no tool annotations), the operator sees a
                    # blank prompt and concludes the terminal is dead.
                    # Usually this is a slash command that the bundled
                    # CLI resolved client-side without hitting the API
                    # (e.g. /health-check on Bedrock with a stale
                    # conversation continuation).  Tell the operator
                    # explicitly so they can retry cleanly.
                    if not emitted_text and not tools_used and num_turns:
                        hint = (
                            "no assistant output — the SDK may have "
                            "resolved the prompt client-side. If you "
                            "used a slash command, try typing 'clear' "
                            "and running it again."
                        )
                        await self._send(
                            f"\r\n{_DIM}  \u26a0 {hint}{_RESET}\r\n"
                        )

                    await self._send(
                        f"\r\n{_DONE_PREFIX}{summary}{_RESET}\r\n"
                    )

                else:
                    # Other messages — ensure spinner is active
                    if _spinner_task is None or _stop_ev.is_set():
                        await _start_spinner()

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
            await _stop_spinner_inner()


# ── Session registry ──────────────────────────────────────────────
#
# Module-level dict keeps sessions alive across WebSocket reconnects.
# The gateway's cleanup task sweeps idle sessions periodically.

_sessions: dict[str, TerminalSession] = {}
_IDLE_TIMEOUT = 1800  # 30 minutes


def get_or_create_session(session_id: str) -> tuple[TerminalSession, bool]:
    """Return ``(session, is_new)``. Creates if not found."""
    if session_id in _sessions:
        return _sessions[session_id], False
    session = TerminalSession(session_id=session_id)
    _sessions[session_id] = session
    return session, True


async def cleanup_idle_sessions() -> None:
    """Remove sessions idle longer than ``_IDLE_TIMEOUT``."""
    to_remove = [
        sid for sid, s in _sessions.items()
        if s._emit is None and s.idle_seconds > _IDLE_TIMEOUT
    ]
    for sid in to_remove:
        session = _sessions.pop(sid)
        await session.shutdown()


def list_sessions() -> list[dict]:
    """Return session metadata (for ``/api/terminal/sessions``)."""
    return [
        {
            "session_id": sid,
            "busy": s._busy,
            "connected": s._emit is not None,
            "idle_seconds": round(s.idle_seconds, 1),
        }
        for sid, s in _sessions.items()
    ]
