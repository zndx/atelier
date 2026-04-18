"""Step defs for features/gateway/terminal_line_editor.feature.

Exercises the line editor on ``TerminalSession`` in-process.  A noop
emit is registered so ``_send`` doesn't raise, and we inspect the
internal state directly — the wire-level ANSI echo is covered by
the smoke test in the scenario-verification stage of the fix plan,
not re-asserted here (bytes-on-wire assertions would just duplicate
the helpers).
"""

from __future__ import annotations

import asyncio
import codecs

from behave import given, when, then


def _run(coro):
    """Run an async coroutine synchronously from a sync behave step."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _session(context):
    if not hasattr(context, "session") or context.session is None:
        from atelier.terminal import TerminalSession
        s = TerminalSession("bdd")

        async def _noop(_frame):
            return None

        s.set_emit(_noop)
        context.session = s
    return context.session


# ── Background / fixture setup ─────────────────────────────────


@given("a fresh TerminalSession")
def step_fresh_session(context):
    context.session = None
    _session(context)


@given('the terminal has history entries {entries}')
def step_seed_history(context, entries):
    # Parse comma-separated double-quoted strings, e.g. `"first", "second"`
    s = _session(context)
    parsed: list[str] = []
    buf = ""
    in_str = False
    for ch in entries:
        if ch == '"':
            if in_str:
                parsed.append(buf)
                buf = ""
                in_str = False
            else:
                in_str = True
        elif in_str:
            buf += ch
    s._history = parsed


# ── Sending input to the session ───────────────────────────────


@when('the terminal receives "{text}"')
def step_send_text(context, text):
    s = _session(context)
    _run(s.handle_input(text))


@when("the terminal receives an arrow-{direction} key")
@when("the terminal receives an arrow-{direction} key {count:d} times")
def step_arrow(context, direction, count=1):
    s = _session(context)
    csi = {"up": "A", "down": "B", "right": "C", "left": "D"}[direction]
    _run(s.handle_input(f"\x1b[{csi}" * count))


@when("the terminal receives a backspace key")
def step_backspace(context):
    _run(_session(context).handle_input("\x7f"))


@when("the terminal receives a delete key")
def step_delete(context):
    _run(_session(context).handle_input("\x1b[3~"))


@when("the terminal receives ctrl-{letter}")
def step_ctrl(context, letter):
    s = _session(context)
    control_char = chr(ord(letter.lower()) - ord("a") + 1)
    _run(s.handle_input(control_char))


@when('the terminal receives a raw escape sequence "{seq}"')
def step_raw_seq(context, seq):
    # behave steps arrive with literal backslash-escapes; decode them
    # so "\x1b[99Z" becomes the one-byte ESC + "[99Z".
    decoded = codecs.decode(seq, "unicode_escape")
    _run(_session(context).handle_input(decoded))


@when('a command "{line}" is recorded in history')
def step_record_history(context, line):
    _session(context)._record_history(line)


# ── Assertions on buffer / cursor / history ─────────────────────


@then('the line buffer is "{expected}"')
def step_buffer_equals(context, expected):
    got = "".join(_session(context)._line_buffer)
    assert got == expected, f"buffer = {got!r}, expected {expected!r}"


@then("the cursor is at column {col:d}")
def step_cursor_at(context, col):
    got = _session(context)._cursor
    assert got == col, f"cursor = {got}, expected {col}"


@then("the history has {count:d} entries")
def step_history_count(context, count):
    got = len(_session(context)._history)
    assert got == count, f"history len = {got}, expected {count}"


@then('the newest history entry is "{line}"')
def step_history_newest(context, line):
    hist = _session(context)._history
    assert hist, "history is empty"
    assert hist[-1] == line, f"newest = {hist[-1]!r}, expected {line!r}"
