# The classification sweep used decisions

2026-10-12. A live classification run called Engine/Decide and stored
the label. Run `d5b12441` finished `CONVERGED` with
`convergence_reason=sweep_only`. The entity `billing.invoice_id`
was classified `SDG.A.1` (invoice identifier). Evidence is
`dst(llm→SDG.A.1)`. Three Decide calls, model `vllm-sr/d3-flash`.

## What the sweep reads

Capability `decisions` is Engine/Decide, not Engine/Complete. The
HTTP body behind Decide is `{model, state, questions}` to
`POST /v1/systemone`. A live choice answer looks like

```json
{"type": "choice", "choice": "SDG.A", "probabilities": {"SDG.A": 0.55, "SDG.B": 0.45}, "confidence": 1e-5}
```

The walk ranks `probabilities`. The `confidence` number on that
object is a different score and is not the option probability. A
flat `{option: probability}` map is still accepted. The thinking
revisit stays Engine/Complete capability `thinking`. There is no
instruct fallback when decisions is unhealthy.

The opening Choice is the structural genera of the loaded taxonomy.
NHSVM membership is not a gate on this walk. The 0.15 LLM discount
is uncalibrated against the decision probability.

## The GPU mix that served the run

One model owns each GPU. Thinking (`Qwen/Qwen3.8-27B`) stays on
GPUs 0–3, whole cards, tensor parallel 4. The decisions replicas
are `d3-flash-0` on GPU 5 port 8096 and `d3-flash-1` on GPU 4 port
8097, each one card. When Signals admitted that configuration, the
model already on the light cards (the CLT worker) was stopped
before `d3_server` started. Memory is not sliced so two models can
share a card.

The engine that served the run is pid 2536073, started 02:33:19 UTC,
and was left running. Gaius `595ef5d` is the yield. Atelier
`62e9f29` is the sweep. Signals `c3ad478` keeps a kind-scoped
activity list from walking ended history inside the 15s RPC.

The proof process set `ATELIER_LATTICE_HOST` to this host's LAN
address because the peer-contract hostname does not resolve here.
MaxSim was turned off for that process only: the synthetic taxonomy
has no current Qdrant collection, and the product default stays on.
SVM had no registered head, which the run recorded and continued
past. Those two channels are later work.
