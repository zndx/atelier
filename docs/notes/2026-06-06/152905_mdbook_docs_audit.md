# Atelier mdbook Documentation Audit — 2026-06-06

**Scope:** all 32 published pages under `docs/src/` (the rendered book), audited
against the **current source tree as ground truth**. Method: five parallel
read-only audit agents, each given an identical verified architecture-of-record
brief; every flagged claim was cross-checked against `src/` before reporting.
`docs/notes/**` and `docs/scratch/**` (63 working notes) are intentionally
out of scope — they are dated historical records, expected to drift.

---

## 1. Executive summary

The published docs narrate a classification stack that is **two generations
stale**. The project's two flagship research-engineering advances exist in code
but are described in the docs as their *predecessors*:

| Capability | Docs still say (stale) | Code actually does (current) |
|---|---|---|
| MaxSim source | "Cosine similarity" — single-vector all-MiniLM-L6-v2 embedding, a DST channel | **`maxsim`** — ColBERT v2 per-token multi-vectors, **Qdrant native MaxSim** late-interaction, fail-fast (`MaxSimUnavailable`, no silent fallback) |
| SVM source | "Sparse TF-IDF char/word n-grams → LinearSVC → Platt scaling" | **ModernBERT mean-pool → factorized fully-hierarchical NHSVM** (Choi 2015; per-node weights, root-to-leaf path scores, non-leaf nodes first-class, calibrated softmax temperature, registry-promoted) |

This is the exact confusion you're hearing reported. **~40 findings**, of which
**7 are peer-review BLOCKERs** (they misrepresent what the system fundamentally
*is*), concentrated in the highest-visibility pages a reviewer reads first
(`introduction.md`, `classification.md`, `dst-evidence-independence.md`,
`synth.md`) and — most damagingly — **inside `maxsim-channel.md` itself**, the
one doc dedicated to the new work.

**On the ColPali / ColVision worry:** confirmed **absent from both code and
docs** — zero references anywhere. Every ColBERT mention is correctly scoped as
*text* late-interaction. So the ColPali/ColVision chatter is **not coming from
the committed documentation** — it is stale memory or ColBERT→ColPali
conflation. The audit did surface **one real phantom**: a `CerebrasBackend`
class that does not exist (Cerebras is served via `OpenAICompatibleBackend`),
cited in `agents.md` and `classification.md`.

Net: the underlying work is real and strong; the docs simply haven't caught up.
The remediation is well-bounded and listed in §8.

---

## 2. Ground truth — architecture of record (verified from `src/`, 2026-06-06)

Dempster–Shafer fusion of **exactly 6 evidence sources** (`pipeline.py`
~L2718–2875), named precisely: `name_match`, `pattern`, **`maxsim`**, `llm`,
`catboost`, **`svm`**.

- **`maxsim`** = ColBERT v2 (`colbert-ir/colbertv2.0`, BERT + 768→128 linear
  projection → per-token 128-dim multi-vectors, `colbert_encoder.py`). A single
  `colbert` multi-vector field in Qdrant (`MultiVectorConfig(comparator=MAX_SIM)`);
  Qdrant performs native MaxSim. `maxsim_bridge.py` converts top-K → DST mass.
  **Replaced** the legacy single-vector cosine source (`cosine_to_mass` removed
  2026-05-25). **Fail-fast:** enabled-but-can't-run → `MaxSimUnavailable` →
  FSM ERROR; **no silent fallback** to single-vector cosine (−13.6pp measured).
- **`svm`** = ModernBERT mean-pool dense embeddings → **factorized NHSVM**
  (`factorized_nhsvm.py`, `registry/nhsvm_head.py`; default `classify.svm.source
  = "registered"`). Per-node weights + per-node alphas; path score over
  root-to-leaf ancestors; **non-leaf nodes are first-class prediction targets**
  (authentic fully-hierarchical); calibrated softmax temperature; emits user
  codes natively (no runtime alignment). The TF-IDF + LinearSVC + Platt
  (`CalibratedClassifierCV`) + SVD path in `svm_classifier.py` is the
  **legacy/baseline** (`per_vocab_legacy`); dense embeddings catastrophically
  fail the *old* Kronecker NHSVM (98.9% TF-IDF vs 4.3% naïve-dense) — which is
  *why* the factorized form exists.
- **`cosine`** survives legitimately only as the *metric* the MiniLM
  single-vector encoder (`embedding.py`) uses for CatBoost/SVM feature
  embeddings and the Monte-Carlo M0 pre-classifier. That usage is correct and
  must **not** be "fixed" to maxsim.

---

## 3. Theme A — cosine-channel → maxsim (dominant staleness)

The retired single-vector cosine source is still named as a live DST channel in:

| Doc | Location | Sev | Fix |
|---|---|---|---|
| `introduction.md` | "Six Evidence Sources" table, "Cosine similarity" row (~L92) | **BLOCKER** | Replace with MaxSim (ColBERT late-interaction) row |
| `classification.md` | Evidence Sources table "Cosine similarity / MiniLM / `classify.discounts.cosine`" (~L49) | **BLOCKER** | → MaxSim row; key `classify.discounts.maxsim` |
| `classification.md` | "Evidence Independence" §, ~140 lines built on "cosine vs CatBoost share the dense embedding" (~L78–222) | **BLOCKER** | Rewrite: maxsim is the independent semantic channel |
| `classification.md` | Discounts HOCON `cosine = 0.30`, `ATELIER_DISCOUNT_COSINE` (~L622) | MISLEADING | `classify.discounts.cosine` is a **rejected** key → use `maxsim = 0.20` |
| `dst-evidence-independence.md` | Canonical six-source list, item 3 = `cosine` (~L92) | **BLOCKER** | item 3 → `maxsim`; reconcile independence argument |
| `dst-evidence-independence.md` | "Cosine reliability shaping" §, built on `cosine_to_mass` (**removed**) (~L287–366) | MISLEADING | Retitle "MaxSim reliability shaping"; rebind to `maxsim_to_mass`/`_maxsim_positive_mass` (math is still correct) |
| `dst-evidence-independence.md` | discount table `cosine` row 0.20 (~L177) | MINOR | rename row → `maxsim` |
| `pipeline-phases.md` | CLASSIFYING lists `...cosine...` in 6-source set (~L52) | MISLEADING | → `maxsim` |
| `embeddings-reviewer-guide.md` | "(name match, pattern, **cosine**, LLM, CatBoost, SVM)" (~L19); "fall back to cosine alone" (~L184) | MISLEADING | → `maxsim` |
| `synth.md` | "the dense embedding used by **cosine** and CatBoost" (~L84) | MISLEADING | drop cosine-as-source |
| `deployment-ontology-agnosticism.md` | "cosine `0.20`" discount (~L118) | MISLEADING | → `maxsim 0.20` (see also §6 for the other stale numbers) |
| `integrations.md` | proposed MLflow `discount_cosine` key (~L61) | MINOR | → `discount_maxsim` when built |

> **Do NOT over-correct:** `monte-carlo.md` (`classify_cosine_batch`, MC M0
> pre-filter) and `embeddings.md` cosine references are the legitimate *metric*
> on the MiniLM encoder — leave them.

---

## 4. Theme B — TF-IDF/Platt SVM → ModernBERT factorized NHSVM

| Doc | Location | Sev | Fix |
|---|---|---|---|
| `introduction.md` | SVM row "Sparse TF-IDF…Platt-scaled LinearSVC" (~L97); orthogonality para (~L99); "SVM with Vocabulary Alignment" TF-IDF (~L158) | **BLOCKER** | → ModernBERT factorized NHSVM; independence is via discounts (Denoeux 2008), not feature-space orthogonality |
| `classification.md` | Evidence Sources SVM row "Dual TF-IDF + LinearSVC (Platt)" (~L55); Evidence-Independence "sparse TF-IDF char/word n-grams" (the orthogonality argument is now **inverted** — the SVM is dense too) | **BLOCKER** | rewrite to ModernBERT NHSVM; add the 98.9%-vs-4.3% dense-failure motivation |
| `dst-evidence-independence.md` | source 6 = "TF-IDF + LinearSVC w/ LLM-mediated ICE→user alignment" (~L95) — also self-inconsistent with its own L184 | **BLOCKER** | → ModernBERT factorized NHSVM, native user codes |
| `synth.md` | entire "SVM Path (Signals)" + "SVM Training" §§ (~L69–146), key-files row | **BLOCKER** | lead with ModernBERT factorized NHSVM; frame TF-IDF as legacy/baseline; add `factorized_nhsvm.py`, `registry/nhsvm_head.py` |
| `pipeline-phases.md` | VALIDATING "synth-trained SVM translated through LLM-mediated alignment" (~L51); `svm_frontier.pkl` artifact (~L168) | MISLEADING | → registered NHSVM, native codes; artifact is `svm.pkl` |
| `ml-artifacts.md` | layout `svm_frontier.pkl` (now `svm.pkl`, `artifact_set.py:59`); "skipped if fit-to-LLM didn't fire" describes the *excised* M9 retrain; intro "runtime LLM-mediated alignment" (~L4) | MISLEADING | `svm.pkl`/`svm.classes.json`; SVM is on-by-default registered NHSVM |
| `pareto-capability-evolution.md` | "label-stable **TF-IDF view**" (~L65); SVM "C and kernel", `classify.svm.blend_ratio` (~L119,192) | MISLEADING / MINOR | → "synth-trained NHSVM view"; knob names are LinearSVC-era / non-existent |

> **`svm_frontier` also violates the reserved-term rule** (CLAUDE.md: "frontier"
> is reserved for the Pareto sense). Dropping the legacy filename removes it.

---

## 5. Theme C — `maxsim-channel.md` (the canonical new-work doc) has internal errors

This page is in the TOC and is where reviewers go to understand the headline
contribution — yet it contradicts the code (and itself) in several places:

- **BLOCKER (would break a deploy):** Configuration § documents
  `classify.cosine.late_interaction.{enabled,model,qdrant_url}` and
  `classify.discounts.cosine` — these are in `_LEGACY_MAXSIM_KEYS` and are
  **loudly rejected** by the config loader (`config.py` ~L839). Real namespace:
  `classify.maxsim.{enabled,model,union_focal_k,union_focal_alpha}` +
  `classify.discounts.maxsim = 0.20` (`base.conf` ~L603–664); `qdrant_url`
  comes from the taxonomy_registry row, not a config key.
- **BLOCKER (contradicts fail-fast):** describes a WARNING-and-degrade fallback
  with `maxsim_path: "legacy_degraded:<reason>"`. No such path — the bridge
  raises `MaxSimUnavailable` and the FSM errors; `maxsim_path` ∈
  `{late_interaction, explicit_disable, unused}`. The single-vector fallback was
  **removed** with an explicit "Do NOT reintroduce" comment (`pipeline.py` ~L2814).
- **MISLEADING:** "Late-interaction execution" describes a multi-slot,
  per-role weighted-sum query (`col_name_view`, `col_sample_*`, `w_label`,
  `w_proto_per_sample`…) that does **not** exist. Reality: one entity text
  (`to_embedding_text()`) → one ColBERT multi-vector query → single `colbert`
  field → native MaxSim. Move the multi-slot design to "Deferred work" or delete.
- **MINOR:** SHAP section claims `late_interaction_{positive,negative,view_*}`
  feature slots wired into `FEATURE_NAMES`; actual surface is a
  `maxsim_attribution` dict, no negative channel (the `negative_score` field is
  vestigial). Discount-slot framing ("starts at cosine value").

**Preserve (correct):** the naming banner, ColBERT encoder description (768→128,
per-token 128-dim, special tokens stripped), `MultiVectorConfig(MAX_SIM)` + single
`colbert` field, the `maxsim_to_mass` Haenni–Hartmann reliability-shaping math,
and the Ægir framing + ColBERT citations (Khattab 2020, Santhanam 2022).

---

## 6. Theme D — Phantom check

- **ColPali / ColVision / ColQwen / any vision-document model: ABSENT** from
  `src/` and `docs/` (independently confirmed by all five agents). ColBERT is
  consistently text-only. **The committed docs are not the source of the
  ColPali/ColVision chatter.** Recommend adding a one-line "what this is *not*"
  note in `maxsim-channel.md` (ColBERT = text late-interaction; not ColPali/
  vision-document retrieval) to inoculate against the conflation.
- **Real phantom — `CerebrasBackend`:** `agents.md` (~L84) and `classification.md`
  (~L547) name a `CerebrasBackend` class that doesn't exist. Cerebras is served
  via `OpenAICompatibleBackend` (default model `zai-glm-4.7`). Sev: MISLEADING.

---

## 7. Themes E–G — config drift, structure, misc

**E. Stale config defaults a reviewer would copy:**
- `monte-carlo.md`: `sample_fraction = 0.15` → **1.00** (shipped default, both
  `config.py` and `base.conf`); `propagation_threshold = 0.85` → **0.80**.
- `deployment-ontology-agnosticism.md`: SVM discount `0.55` → **0.22** (0.55 is
  now `catboost_base`); `gap_threshold 0.15` → **0.18**; cautious_review
  `bel_threshold 0.85` → **0.0** (off by default); `cosine 0.20` → `maxsim 0.20`.
  (`llm 0.15` is correct.)
- `data-sources.md`: universal vocab "16 PII leaves" → **29**; and it claims a
  silent fallback to the universal fixture — code now **fails fast** with a
  `RuntimeError` (anti-silent-degradation guard, `pipeline.py` ~L2496).
- `agents.md` (~L117) & `grpc.md` (~L129): `agent.model =
  "claude-sonnet-4-5-20250929"` → **`claude-opus-4-7`** (shipped default; the
  Sonnet literal is an unreachable last-resort guard).

**F. Structural / orphaned / dead links:**
- `cco-coverage.md`, `sdg-fixture-grounding.md`, `gpu-acceleration.md` exist but
  are **not linked in `SUMMARY.md`** (unreachable in the rendered book). The
  first two are accurate — **link them**. (`sdg-fixture-grounding.md` also needs
  a present→past reframe: the `GT.*`→`SDG.*` rewire it "proposes" has shipped.)
- **`gpu.md` vs `gpu-acceleration.md`:** the *orphaned* `gpu-acceleration.md` is
  the **more accurate** of the two; the *linked* `gpu.md` is partly **wrong**
  (claims posterior-sampling uncertainty "benefits from GPU" — it is **disabled**
  on GPU, `catboost_classifier.py` ~L148; cites `/api/status` for GPU — it's
  `/api/acceleration`). **Make `gpu-acceleration.md` canonical**, repoint
  `SUMMARY.md:19`, fix or retire `gpu.md`.
- Dead doc links: `pipeline.py:2749` and `base.conf` point at
  `docs/src/architecture/late-interaction-cosine.md` (**doesn't exist** → should
  be `maxsim-channel.md`); `embeddings-reviewer-guide.md` links
  `audit_2026-05-06_a.md` (**doesn't exist**).

**G. Other staleness:**
- `embeddings.md`: attributes embedding-atlas to "Apple's embedding-atlas" — we
  ship a **fork** (`rch/oss-embedding-atlas`, per `.gitmodules`) with required
  modifications; cite the fork. "SIGDG" → "SDG" (the unified surface).
- `scenarios/testing.md`: `just bdd` / `just bdd-full` **removed** → `just
  behave`; "Qdrant on :6334" → **:6333** (HTTP; 6334 is gRPC); feature inventory
  stale (lists ~35; repo has **58**).
- `scenarios/overview.md`: "155 scenarios across 35 features" → ~**338 / 58**.
- `nautilus.md`: callback cited at `gateway.py:2154` → **2738** (line drift only).
- `deployment-ontology-agnosticism.md`: "infinite recursion in `descendants`" →
  "unbounded loop" (iterative stack, no visited-set; the gap is still real).
- `pareto-capability-evolution.md`: dangling short commit hashes (history was
  filter-repo'd 2026-05-31); the claim itself is correct — drop the hashes.
- `sprint-2026-05-20.md`: dated appendix — add a "historical; superseded specifics"
  banner rather than rewriting.

**Bonus (out of mdbook scope, same root cause):** `CLAUDE.md`'s own DST list
reads "name-match, pattern, **cosine**, LLM, CatBoost, SVM" — stale; should be
`maxsim`. Worth fixing so future agents inherit the correct list.

---

## 8. What's correct & current (preserve)

- `maxsim-channel.md` naming banner, ColBERT encoder spec, MaxSim mass math, Ægir
  framing, citations (see §5).
- `gpu-acceleration.md` end-to-end (sharding, custom SAGE/SHAP kernel, cuml UMAP,
  `posterior_sampling` disabled on GPU, `/api/acceleration`, RAPIDS install).
- DST / belief-gap-convergence framing & `cautious_code(τ)` (`introduction.md`),
  Monte-Carlo three-phase architecture (`monte-carlo.md`), `ml-artifacts.md`
  extend-classification machinery, `synth.md` generators/registry + CatBoost,
  `cco-coverage.md` (10/11 + EAV-gated Units), `sotab-coverage.md`,
  `secrets.md`, `scenarios/deployment.md`, `grpc.md` RPC/REST inventory,
  `nautilus.md` design. **No phantom vision models anywhere.**

---

## 9. Prioritized remediation plan

**P0 — peer-review BLOCKERs (do first; these define what the system *is*):**
1. `introduction.md` "Six Evidence Sources" table (cosine + TF-IDF SVM rows) — the most-quoted content.
2. `classification.md` Evidence Sources table + Evidence-Independence section + module map.
3. `dst-evidence-independence.md` six-source list + "cosine reliability shaping" rebind.
4. `synth.md` SVM Path / SVM Training sections.
5. `maxsim-channel.md` Configuration block (rejected legacy keys) + fail-fast contract + multi-slot fiction.

**P1 — MISLEADING (would mislead but not mischaracterize the system):**
6. `pipeline-phases.md`, `ml-artifacts.md`, `embeddings-reviewer-guide.md`, `pareto-capability-evolution.md` SVM/cosine wording.
7. Stale config defaults: `monte-carlo.md`, `deployment-ontology-agnosticism.md`, `data-sources.md`, agent-model in `agents.md`/`grpc.md`.
8. `CerebrasBackend` phantom (`agents.md`, `classification.md`).
9. GPU doc consolidation (make `gpu-acceleration.md` canonical, fix/retire `gpu.md`).

**P2 — structural / minor:**
10. Link orphans into `SUMMARY.md` (`cco-coverage.md`, `sdg-fixture-grounding.md`, `gpu-acceleration.md`); fix dead doc links (`late-interaction-cosine.md`, `audit_2026-05-06_a.md`).
11. `embeddings.md` fork attribution + SIGDG→SDG; `testing.md`/`overview.md` recipes + counts; `nautilus.md` line anchor; misc.
12. `CLAUDE.md` DST-source list (bonus).

**Cross-cutting consistency rules for the rewrite:**
- Source name is **`maxsim`** everywhere it denotes the DST channel; keep
  **cosine** only for the MiniLM metric / MC pre-filter.
- SVM = **ModernBERT factorized fully-hierarchical NHSVM** (registered head);
  TF-IDF/Platt is **legacy/baseline**, always labeled as such.
- Honor the fail-fast / no-silent-fallback contract in every mention.
- Preserve the reserved-term discipline: "frontier" = Pareto sense only.
