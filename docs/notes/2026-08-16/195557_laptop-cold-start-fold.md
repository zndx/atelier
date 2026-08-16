# Fold GitHub laptop cold-start onto GPU-ready trunk

Rebased local `peer-unit@atelier` (`734be28` → `6bc4c03`) onto
`origin/trunk` (`b4e5bd9`):

- `ee9eb27` SDG first-run substrate + llama.cpp turn-key classify
- `605a793` SDG-era docs
- `b4e5bd9` hold `sdg-corpora` at `b24ef9f6` (populated collections)

## Kept separate (do not collapse)

| Runtime | Path |
|---------|------|
| Laptop cold start | `devenv up` + llama.cpp `:8080` + `just sdg-sample` |
| Linux GPU lattice | `atelier.service` → `python -m atelier.engine.server` `:50251` |

`devenv up` does **not** start the capability engine (dual-bind).
llama.cpp skips when `:8080` is already listening (Gaius vLLM on this host).

Did **not** apply the local stash `wip: full-stack unit rewrite` — that
would have made systemd start the whole product stack and drifted the
engine-only lattice unit.

## Verify

- `tests/engine` + `test_gpu_probe.py`: 24 passed
- `atelier.sdg.sample` profiles: macbook / workstation / cluster
- live `Engine/Status` on `:50251` still `project=atelier`
