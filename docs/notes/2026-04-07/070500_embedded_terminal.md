# Embedded Terminal — ghostty-web WASM + Claude Agent SDK

## What Changed

Added an interactive terminal to the Landing page, backed by the Claude Agent SDK.
The terminal uses ghostty-web (pre-built WASM from npm, same as the cybersec project)
for rendering, connected to a line-buffered REPL via WebSocket.

## Architecture

```
ghostty-web (browser WASM) → WebSocket /ws/terminal → FastAPI gateway → Claude Agent SDK
```

- Terminal renders instantly regardless of credential configuration
- WASM component loads locally (~400KB), no API call needed for rendering
- SDK interaction only happens when the user types a message and presses Enter
- Graceful degradation: SDK not installed or no creds → helpful error messages
- Built-in commands: `help`, `status`, `clear`

## New Files

| File | Purpose |
|------|---------|
| `ui/src/components/Terminal.tsx` | React component wrapping ghostty-web with WebSocket, reconnection, ResizeObserver |
| `src/atelier/terminal.py` | TerminalSession class — line buffer, character processing, SDK dispatch, ANSI formatting |
| `ui/public/ghostty/` | ghostty-web.js + ghostty-vt.wasm (derived, gitignored) |

## Modified Files

| File | Change |
|------|--------|
| `ui/package.json` | Added ghostty-web devDep, postinstall script to copy WASM assets |
| `ui/src/pages/Landing.tsx` | Added terminal Card section below feature cards (lazy loaded) |
| `ui/vite.config.ts` | Added /ws WebSocket proxy rule for dev server |
| `src/atelier/gateway.py` | Added /ws/terminal WebSocket endpoint, /ghostty static mount |
| `.gitignore` | Added ui/public/ghostty/ |

## Key Design Decisions

- **ghostty-web from npm** (not compiled from source) — proven in cybersec project
- **No separate PTY proxy process** — WebSocket handler lives in the existing gateway
- **No database persistence** — session lives in WebSocket connection lifetime
- **Dynamic `<script type="module">`** for WASM loading — same pattern as cybersec's loader.js shim
- **max_turns=5, max_budget_usd=0.25** — reasonable defaults for interactive use
- **Lazy-loaded** Terminal component — WASM init is async, doesn't block page render

## Ported from Cybersec

Pattern adapted from `~/local/src/cldr/cybersec/`:
- `zarf/images/otel-navigator.py` (GhosttyTerminal ReactiveHTML class) → React component
- `cybersec/pty_proxy/repl.py` (line buffer, ANSI rendering) → terminal.py TerminalSession
- `cybersec/pty_proxy/server.py` (WebSocket server) → gateway.py WebSocket endpoint

Simplified: removed gRPC engine layer, param_update frames, Navigator-specific commands.
The Claude Agent SDK is the engine — no intermediate protocol needed.
