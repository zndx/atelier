"""Late-interaction enrichment pipeline.

Produces multi-vector enriched annotation profiles in Qdrant for the
late-interaction cosine evidence source.  See
``docs/src/architecture/late-interaction-cosine.md`` for the architectural
design and ``scripts/enrich_annotations.py`` for the CLI entry point.

Package layout
--------------

- :mod:`verifiers` — deterministic per-field verifier functions.  Run after
  each LLM generation; reject + retry on failure.  No LLM in this module.
- :mod:`qdrant_writer` — Qdrant client wrapper that writes one multi-vector
  point per annotation with structured JSON payload.  Content-addressed
  caching to support idempotent rebuilds.
- :mod:`llm_generator` — LLM enrichment generation (Anthropic backend with
  the option to swap in claude-agent-sdk-based harness later).  Pluggable
  via the ``EnrichmentGenerator`` protocol.
- :mod:`loop` — orchestration: read taxonomy → generate enrichment →
  verify → embed → write to Qdrant → update registry.

The package satisfies the LLM-mediated reference artifact pattern
documented in the memory: every output is procedurally reproducible from
its inputs (content-addressed cache key) and falsifiable by the verifier
suite.  Operators inspect via on-demand exports
(``scripts/export_enriched_annotations.py``); edits write back to Qdrant
through a structured CLI with audit logging in the point's payload.
"""
