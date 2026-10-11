# Capability `decisions`, and ClassificationFlow on RKE2

2026-10-11. Follow-up to `211100_d3-imperative-airflow-reconfiguration.md`.
Two corrections.

## The capability is `decisions`

The lattice name matches the route. vLLM serves `POST /v1/decisions`.
`WorkloadOffer` advertises `decisions`. The sweep asks for `decisions`.
The singular `decision` in the previous note is the same effect under
this name.

`thinking` and `instruct` stay operating profiles of `Qwen/Qwen3.8-27B`
on `/v1/chat/completions`. `decisions` is advertised only while the
heavy replace effect is serving `vllm-sr/d3` at tensor parallel 4.
A request for `decisions` while that process is the Qwen checkpoint
fails closed. `Engine/Complete` stays the chat contract. The hosting
engine is the only caller of `/v1/decisions`.

The vLLM effect on the activity is:

```
endpoint:         gaius.endpoint.heavy
action:           replace
model:            vllm-sr/d3
tensor_parallel:  4
capability:       decisions
route:            /v1/decisions
```

## ClassificationFlow is `@kubernetes` on the Signals RKE2 platform

Signals is the RKE2 platform. Platform Metaflow's only Airflow compute
is a `@kubernetes` pod (`KubernetesPodOperator`, not `@batch`), in
namespace `metaflow`, admitted by YuniKorn. `airflow create` is how the
flow becomes the DAG `atelier_sdg_classify`.

`ClassificationFlow` is not that yet. `start`, `probe`, `precondition`,
`load`, `sweep`, `fuse`, and `end` are plain `@step`. The parameter
`kubernetes-preferred` defaults true and nothing reads it. The module
docstring treats a local step as the fallback. `resident.py` already
reloads config and vocabulary from `source_id` so the artifacts stay
pickleable for a pod. `require_signals_metaflow` already fail-closes
when the Signals metadata service is down.

The platform path drops the local fallback. Every step is
`@kubernetes`. A user run that cannot schedule on RKE2 fails closed.
The activity from the previous note is what starts the DAG `airflow
create` produced. The coordination bracket stays outside the flow:
`apply_vllm` until `decisions` is serving, the Metaflow DAG, then
`restore_vllm` (`all_done`) until `thinking` is serving, then the
revisit. Metaflow does not swap vLLM.

The `decisions` process is the Gaius heavy endpoint, not a step pod.
Sweep and revisit pods call the engine. They do not request the four
GPUs. Precondition, when it needs a GPU, is the existing light claim.
That pod is one YuniKorn application stamped with the light queue and
the flow's workload id. One application per claim. Heavy stays the
standing four-GPU application whose command line the effect replaced.
