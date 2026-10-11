# d3 can replace the classification sweep

2026-10-11. Review of Signals, signals-engine, Gaius, Ægir, and
`src/atelier/classify` against Decision 3.0 `vllm-sr/d3` (26.09B,
finetune of `Qwen/Qwen3.8-27B`, Apache-2.0).

## Verdict

Yes for the initial sweep (`instruct`). No as a drop-in model id on
Engine/Complete, and no as a replacement for the `thinking` revisit
that drives DST convergence.

## Where the sweep sits

- Signals is the platform (THS, Atlas / OpenLineage, YuniKorn).
- signals-engine serves `zndx.engine.v1` Complete. `thinking` is
  Qwen3.8-27B with thinking on and effort `xhigh`. `instruct` is the
  same checkpoint with thinking on and effort `low`. Callers never
  see the vLLM URL.
- Gaius hosts that checkpoint. Atelier's engine on `:50251` forwards
  `instruct` and `thinking`; it hosts `referee` (Nemotron) locally.
- Ægir's sdg-corpora vocabulary is the SKOS frame: 1,036 nodes,
  940 leaves, 96 internal, depth at most 5, seven roots. Widest
  sibling set is `SDG.PROCESS` with 180 children. Largest subtree
  is `SDG.GDC` at 412 nodes. No node has more than 255 children.
- Atelier fuses six mass functions. `name_match`, `pattern`, and
  `maxsim` are independent of the LLM. CatBoost is fit-to-llm.
  The registered NHSVM is trained offline. The sweep is
  `CompleteLLMBackend` with `llm_capability_for_sweep(revisit=False)`
  → `instruct`. Revisit passes use `thinking`.

## Why the sweep contract matches

`llm_to_mass` consumes a code, a confidence, and up to three
alternatives. d3's `POST /v1/systemone` returns a softmax over the
options of a Choice (also Noul, and Score as 2–10 ordered levels).
No tokens are generated. Text median on the model card is 55 ms
on one MI325X. Map top-1 to `category_code` / `confidence` and the
next three probabilities to `alternatives`. Put column name, type,
samples, siblings, and ontology priors in `state`. Keep other DST
channel scores out of that state.

## What blocks a model-id swap

- d3 is a code readout (`readout.safetensors`, shape `[255, hidden]`),
  served by its own runtime (`d3_server.py`), not `vllm serve` chat
  completions. `SIGNALS_VLLM_MODEL=vllm-sr/d3` will not answer
  Complete.
- One Choice accepts 1–255 options. The vocabulary is 1,036 nodes.
  A flat question 422s. A walk fits: every sibling set is ≤ 180.
- A greedy walk drops cross-subtree alternatives. DST wants those.
  The question plan has to keep the root distribution and expand
  more than the winning branch.
- Parent-vs-leaf is prose in the current system prompt. On d3 it
  has to be an explicit option or a Noul at each internal node.
- Option keys must be the codes. Labels are not unique (1,026 of
  1,036).
- d3 emits no evidence string and no reasoning trace. Fusion does
  not need the string. The thinking revisit and Overwatch still do.
- d3 is the same backbone family as `thinking`. Replacing `instruct`
  with it does not create a new independent source. The independent
  tier stays `name_match`, `pattern`, `maxsim`. CatBoost will refit
  to d3 labels, which is the existing fit-to-llm design.
- BF16 is about 52 GB plus the vision tower. Column classification
  does not need images or video. Zero-share GPU leases mean d3
  needs its own card beside the Qwen3.8 chat replica.
- The HTTP server handles one request at a time. Questions inside
  a request share one `state` and run `batch_size` 8 per forward.
  A cascade step depends on the previous answer, so a column is
  several serial requests. Still far cheaper than thinking
  generation, and a multi-state batch is the throughput follow-up.
- Softmax confidence is not the self-reported 0–1 the discount of
  0.15 was tuned on. Calibrate before treating raw probabilities
  as mass.

## Leave in place

`thinking` revisit, belief-gap / bel-floor convergence, channel
agreement, the five non-sweep mass functions, the Nemotron referee,
and Overwatch.
