# Waffle instant + systemd/devenv share one compose

Signals waffle is instant because it Statuses local PEERS and skips
offline hints. Atelier still showed “Looking for peers…” because collect
was a sequential BFS on the request path and the rail only fetched on
open (loading copy stayed until that returned).

- `collect_peer_surfaces`: parallel Status of the contract directory,
  ServerQuery PEERS only on live engines, 1s Status / 2s Query, no
  grpc retries. Offline synth/metabase skipped this round.
- WaffleMenu: prefetch on mount, keep last roster, no “Looking for
  peers…” copy (Signals chrome.js).
- systemd wrap pins `XDG_RUNTIME_DIR=/run/user/<uid>` (Gaius
  `export_unit_runtime`). Skip-up only when login-shell `devenv
  processes` can see the compose. Leftover `/tmp/devenv-*` daemons for
  this checkout are reaped on start/stop so `systemctl restart atelier`
  and `devenv processes restart gateway` are the same graph.
