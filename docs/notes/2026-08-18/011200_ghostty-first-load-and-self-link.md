# Ghostty first-load timeout + waffle self-link about:blank

First web-terminal visit hit “ghostty-web loading timed out.” (second
worked). The effect subscribed to `ghostty-ready` and, under Strict Mode,
unsubscribed on the immediate remount; WASM finished with no listener and
the 15s timer painted the error. Loader is now a session Promise
(`ensureGhostty`), started from `main.tsx`, with a poll fallback. The
terminal waits on “Loading terminal…” instead of a hard timeout.

Atelier’s waffle self-link still opened `about:blank` because
`window.open(url, "_blank", "noopener")` returns null — the tab is
created and never navigated. Left-click now `window.open("", "_blank")`
then `location.replace` on the handle. Other peers were already fine
(different port).
