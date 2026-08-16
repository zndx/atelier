# Gaius-style wrap for atelier.service

systemd is only the `signals.target` hook. `ExecStart` runs
`devenv up -d`. The lattice engine is devenv process
`capability-engine` (`scripts/processes/capability-engine.sh`).

Same graph on Linux and macOS — no laptop/server gate. Accept remains
`Engine/Status` at bind (no vLLM SERVING). Stop is `devenv processes
down` + reap foreign `atelier.engine`; no `/tmp/zndx-gpu-leases` wipe.

Live `:50251` on this host is still the old setsid unit until
`systemctl restart atelier` (migrates the listener into process-compose).

Ægir can follow this wrap next.
