# Documentation Overhaul

## Summary

Comprehensive mdbook documentation update to bring docs in sync with
implementation through M8 (GPU acceleration, Monte Carlo sampling, agent loop).

## Commits

1. `772e58b` — Monte Carlo sampling for scalable classification
   - New: `monte_carlo.py` (514 lines), `monte_carlo_steps.py`
   - Modified: pipeline.py, bootstrap.py, config.py, base.conf, classification.feature
   - 5 new BDD scenarios for MC sampling

2. `1cbe976` — Comprehensive documentation overhaul
   - 3 pages rewritten (introduction, agents, gRPC)
   - 3 pages created (monte-carlo.md, gpu.md, synth.md)
   - 4 pages updated (classification, overview, testing, SUMMARY)
   - 971 lines added across 10 files

## Key Findings from Audit

- Docs claimed "30 scenarios, 10 features, 2 domains"
- Actual: 127 scenarios, 31 features, 4 domains
- agents.md was a 14-line speculative stub; now 150 lines covering actual
  5-tool agent convergence loop implementation
- grpc.md was 23 lines; now 130 lines with full endpoint table (25+ REST
  endpoints, 2 WebSocket endpoints, 7 gRPC RPCs)
- No coverage of: Monte Carlo, GPU, synth framework, agent loop — now all documented

## Documentation Structure

```
Introduction           — Value proposition, architecture, OOTB, quick start
Architecture (11 pages):
  System Overview      — d2 component diagram
  Deployment           — CAI AMP + local dev
  gRPC & Gateway       — Proto contract, REST endpoints, config lifecycle
  Keystone Agents      — 5-tool convergence loop, LLM backend matrix
  Classification       — DST methodology, 6 evidence sources, bootstrap
  Monte Carlo Sampling — Stratified sampling, label propagation, scaling
  GPU Acceleration     — CUDA detection, nix symlink, batch encoding
  Synth & Training     — 316+ generators, CatBoost + SVM, SAGE/SHAP
  Embeddings           — embedding-atlas visualization
  Data Sources         — Source-aware versioning, OOTB sample
  Proposed Integrations — MLflow, Hive (future)
Scenarios (4 pages):
  Overview             — 127 scenarios, 4 domains
  Test Infrastructure  — 31-file feature tree, tier system
  Deployment Modalities — AMP, Application, Studio
  Runtime Profile      — Pre-push validation
```
