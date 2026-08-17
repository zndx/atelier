# llama-cpp is darwin-only

Linux `devenv up -d` / `atelier.service` evaluated `llama-cpp` (CUDA
compat) and failed. Classify on this host is capability-engine →
shared vLLM.

`devenv.nix` now scopes `llama-cpp`, the `llama` process, and
`llama-serve` with `lib.optionals isDarwin` / `optionalAttrs`.
`devenv info` on linux lists capability-engine, not llama.
