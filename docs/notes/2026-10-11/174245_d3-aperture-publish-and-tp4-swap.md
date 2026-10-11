# Publish one aperture, swap one TP-4 endpoint

2026-10-11. Follow-up to `174500_d3-genus-sweep-and-gpu-borrow.md`.
The genus-first Choice stands. Serving is one unquantized `vllm-sr/d3`
process on the standing heavy claim.

## The aperture ships with the ontology

`aegir.lineup sync` already publishes the entailment chain. Realized
OMN is the logical source, SKOS is the content model, SHACL is the
projection (`vocab_lowering.py`: OWL ⊨ SKOS ⊨ SHACL). The gate is
schema-CI and OQuaRE and sufficiency. The share is the `sdg-corpora`
submodule.

That share stops at ontology, `vocabulary/annotations.*`, and DDL.
`AperturePoints` is absent from the Atelier pin. Three checkouts
therefore answer "what is the aperture" differently:

- Ægir keeps membership in `aperture_layer.ttl` and the MaxSim index
  live on `:6355` (`sdg_aperture`, 33 points). Those payloads still
  say LIMS `16.2`; `sdg_domains` has the same IRI as code `9`.
- Gaius loads `external/sdg-strategy` (`aperture.snapshot.json`,
  `admission_filter.json`) and says it does not invent a second
  aperture. The loader records that encoder weights and the live
  Qdrant index are not in that pin. The embedding profile is
  `lightonai/ColBERT-Zero`, threshold regime, 512-token grain.
- Atelier classifies blind against `vocabulary/annotations.parquet`.
  NHSVM heads are keyed by those `SDG.*` codes. `domain-concepts.ttl`
  already ships the scheme tops and records that a top-concept
  graduation does not admit the concept to the aperture.

The d3 genus Choice is the occasion to extend the same sync with one
more product, regenerated from the realized ontology:

- Genus SKOS terms: code, IRI, prefLabel, definition, parent,
  children. The codes are the NHSVM `code_to_idx` codes, so the walk
  and the head name the same nodes.
- `AperturePoints` as an explicit SKOS collection. Point-hood stays a
  published fact, separate from notation and from scheme tops.
- ColBERT-Zero identity: encoder `lightonai/ColBERT-Zero`, τ, token
  limit, collection name, vector sha per anchor. The weights stay on
  the Hub. `:6355` is rebuilt from the pin.
- SHACL that checks that membership. Constituent neighborhoods stay
  retrieval payload and stay out of the Choice.

Ægir, Gaius, and Atelier then pin one submodule. The first Choice
reads that published genus set. It fits the 255-option cap. The live
33-point index stops being the source of the question.

## One unquantized process, TP=4

The standing heavy profile (`DEPLOYMENT_PROFILES["thinking"]` in
Gaius) is `Qwen/Qwen3.8-27B`, `gpu_tokens=4`, `tensor_parallel=4`,
capabilities `thinking` and `instruct`. Gaius starts it as
`vllm serve <model> --tensor-parallel-size 4` on GPUs 0–3. Heavy
stays guaranteed 4, fenced, `maxapplications: 1`. Cards 4 and 5 stay
with the Gaius process and Moshi. `federation-queues.yaml` stays as
it is.

A d3 phase is a deployment-config swap inside that claim, the same
shape as `plan_swap_for_reasoning` / `plan_restore_default`. Airflow
owns the activity, YuniKorn keeps the heavy share, Metaflow runs the
sweep:

1. Record the standing profile. Hold the heavy share.
2. Stop the Qwen process. Start one unquantized `vllm-sr/d3` process
   on the same four GPUs, `--tensor-parallel-size 4`, BF16 weights.
   About 52 GB across four 24 GB cards is the same memory family as
   the standing workers (~21 GB each).
3. Metaflow runs the genus Choice and the walk while that process is
   up. `ClassificationFlow.sweep` currently runs the whole pipeline,
   including the thinking revisit; this phase runs the d3 sweep only.
4. On success and on failure, stop d3 and start the Qwen3.8 profile
   again. The thinking revisit runs after that restore, on the
   disagreement set.

`instruct` is an operating profile of the same Qwen process, so it is
down for the swap. The sweep talks to the d3 process directly.

The published `d3_runtime.py` loads with `device_map` onto one device
and serves `POST /v1/systemone`. Stock `vllm serve vllm-sr/d3` loads
the Qwen backbone as a chat model. The TP-4 process is the
deployment; the 255-way readout has to be the model class on that
process before the sweep can call it. The weights stay the
unquantized Hub checkpoint.

Later vLLM configuration changes use this same Airflow + YuniKorn +
Metaflow swap and stay inside the heavy claim.
