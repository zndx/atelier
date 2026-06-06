# Persistent Terminal Sessions

**Date:** 2026-04-14 (continued 2026-04-15)

## Problem

The Ghostty WASM terminal + Claude Agent SDK REPL session died on page
navigation or browser reload. The WebSocket disconnect triggered
`session.shutdown()`, cancelling any in-flight SDK query and losing all
context and output.

## Solution

Server-side session persistence with ring buffer replay:

1. **Session registry** (`terminal.py`): Module-level `_sessions` dict keeps
   `TerminalSession` objects alive across WebSocket reconnects. Ring buffer
   (`deque` maxlen=65536) captures all output regardless of client connection.

2. **Detach/attach pattern**: On WS disconnect, `detach()` nulls the emit
   callback but keeps the session alive. SDK queries continue running. On
   reconnect, `attach()` replays the buffer and re-registers the callback.

3. **Idle cleanup**: Background asyncio task sweeps sessions with no client
   for >30 minutes.

4. **Client-side session ID**: `localStorage` persists the session UUID so
   the same session is reconnected across page loads and navigation.

5. **Dedicated `/terminal` page**: Full-screen terminal with `fullHeight`
   Layout, alongside the embedded preview on the Landing page.

6. **Header navigation**: Added nav bar to Layout with links to all pages
   (Agents, Workflows, Terminal, Embeddings, Status) with active state.

## Files Changed

- `src/atelier/terminal.py` — Ring buffer, detach/attach, session registry, cleanup
- `src/atelier/gateway.py` — `/ws/terminal/{session_id}`, cleanup task, sessions API
- `ui/src/components/Terminal.tsx` — sessionId prop, localStorage, reconnect notice
- `ui/src/pages/TerminalPage.tsx` — New full-screen terminal page
- `ui/src/App.tsx` — `/terminal` route with fullHeight Layout
- `ui/src/components/Layout.tsx` — Header nav bar with active state highlighting

## Verification

- TypeScript check: clean (`npx tsc -b --noEmit`)
- 98 tier-0 BDD scenarios pass (0 failed)
- Manual: navigate away during SDK query, return, see buffered output
