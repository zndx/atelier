# Waffle flashed “unreachable” during Ghostty / engine connect

Peers showed, then “Federation surfaces unreachable” while ghostty-web
WASM timed out (15s) on the same origin, then peers returned.

Causes: each route mounted its own Layout (waffle remounted empty);
`AbortSignal.timeout(4000)` fired while Vite was busy on the WASM
download; a successful empty/error refresh could replace a good roster.

Layout now wraps the router. Last-good roster is kept in the module and
in `collect_peer_surfaces_cached` (8s TTL; empty/failed walk serves the
previous list). No 4s abort on the waffle fetch.
