# `decisions` has its own endpoint and queue

2026-10-11. Follow-up to `211905_decisions-capability-and-k8s-metaflow.md`.
`thinking` and `decisions` are different capabilities, so they are
different endpoints and different YuniKorn leaves. On this host the
two leaves cannot be in force together.

## Two endpoints

`thinking` stays `gaius.endpoint.thinking`: `Qwen/Qwen3.8-27B`, tensor
parallel 4, `/v1/chat/completions`. `instruct` stays an operating
profile of that process.

`decisions` is `gaius.endpoint.decisions`: its own vLLM process, its
own port, model `vllm-sr/d3`, tensor parallel 4, capability
`decisions`, route `POST /v1/decisions`. It is not a command-line
change to the thinking process. `WorkloadOffer` lists `decisions` only
while that process is serving.

## Two queues

`root.internal.inference.heavy` stays the standing leaf: guaranteed 4,
max 4, `maxapplications: 1`, fenced, example `gaius.thinking`.

`root.internal.inference.decisions` is a new leaf in
`federation-queues.yaml`, promoted through `write-scratch` / `promote`.
Peers still do not write that file. Placement is `create: false`, so
the leaf exists before anyone submits. Shape: class
`internal.inference.decisions`, max 4, `maxapplications: 1`, fenced,
standing guaranteed 0, examples `gaius.decisions;vllm-sr/d3;tp4`. The
floor of 4 is asserted by the classification activity and retracted
when the activity ends, the same way `agent-rtc` is asserted only
while a session runs.

`class_for_gpu_tokens(4)` returns heavy today
(`sentinel_claim.py`), and `ModelDeploymentProfile.resource_class`
uses that. A decisions profile with four tokens has to name
`root.internal.inference.decisions` itself. Four GPUs do not mean the
heavy queue.

## Mutually exclusive on this host

Six RTX 4090s. Parent GPU max is 6. Each endpoint is tensor parallel 4.
Standing floors are heavy 4 and extract 1. A decisions floor of 4 fits
only while heavy's floor is 0 (4 + 4 + extract is 9 and the capacity
gate rejects it). The pair is the same kind of exclusion the arbiter
already applies to the leftover leaves, at the TP=4 size.

The classification activity claims `decisions` at 4 and precludes
`root.internal.inference.heavy`. While it is `RUNNING` the arbiter
retracts heavy's floor to 0 and asserts decisions at 4. Gaius stops
the thinking endpoint, because that process is what holds the cards,
then starts the decisions endpoint on those four GPUs. The two
processes do not overlap. Release, lapse, or failure stops decisions,
restores heavy's floor to 4, and starts thinking again.

Cards 4 and 5 stay outside the pair (extract, light, agent-rtc).
Moshi and an in-force interactive session stay up.

The Metaflow DAG does not change shape. `@kubernetes` sweep pods call
`decisions` only after that endpoint is serving. Revisit pods call
`thinking` only after it is serving again. The phases cannot overlap,
because the queues cannot.
