"""Atelier capability/gRPC engine — local model serving ("inference capability").

Minimal mirror of the Ægir engine (`aegir/src/aegir/engine/`), which is itself a
minimal mirror of the Gaius engine — the established cross-project template.

**Strict layering:** the engine is the SOLE vLLM client and owns the
capability→model mapping; workloads connect ONLY to the gRPC engine
(`atelier.engine.client.complete`) — never to vLLM, never handed an endpoint URL.
(Inc-0 exception, mirroring Ægir's current practice: the classification
pipeline's ``OpenAICompatibleBackend`` may point at the engine's vLLM port
directly until the OpenAI-compat proxy over gRPC lands.)

Capability boundary (referee independence, see
docs/notes/2026-07-03/140044_engine_inference_capability_spec.md):

- ``instruct`` — the runtime LLM evidence channel (Qwen3.6-35B-A3B-FP8).
- ``referee`` — agent-mediated curation ONLY (Nemotron-3-Nano-30B-A3B-BF16).

Nemotron never serves runtime evidence; Qwen never referees. Distinct model
families keep the agent-mediated reference architecturally independent of the
ensemble it steers.

Federation seam: ``ATELIER_ENGINE_FEDERATE`` (via HOCON ``engine.federate``)
delegates capability requests to a sibling engine (Ægir :50151, Gaius :50051)
— a config change, not a rework.
"""
