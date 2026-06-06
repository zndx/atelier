# DST Evidence Independence

This note documents how Atelier's classification pipeline handles
**non-distinct evidence sources** under Dempster-Shafer fusion, and
why the discount calibration and revisit gate are structured the way
they are. It is intended to be cited by code reviewers and academic
readers.

## The pipeline as iterative refinement

Atelier's bootstrap loop is **iterative refinement on a belief-
assignment vector** B over columns: ``B_{n+1} = T(B_n)``, where T
composes the LLM sweep, ML validation (CatBoost + SVM), DST fusion,
and targeted revisit on disagreement. Cast in the language of
classical numerical analysis (Banach 1922; Saad 2003 §4.1,
*Iterative Methods for Sparse Linear Systems*), every component of
the pipeline maps onto a numerical-method primitive:

| Component | Numerical-methods primitive |
|---|---|
| Bootstrap loop | Fixed-point iteration on B |
| LLM sweep | Stochastic operator T_LLM (Robbins-Monro 1951 framing) |
| ML validation | Deterministic linearization T_ML |
| DST fusion | Combiner ⊕ producing fused state |
| Targeted revisit on disagreement | Local smoothing in multigrid (Brandt 1977) |
| Pl − Bel gap | A posteriori error estimate per column |
| Conflict K | Nonlinear residual diagnostic |
| Ontology priors | **Preconditioner** — conditions first-pass output |
| Reliability discount on derivative sources | **Damping / step-size control** |
| Hierarchical cosine mass | **Coarse-grid correction** (multigrid) |
| `cautious_promoted_code` | **Projection onto coarse grid** at level where evidence unambiguous (Smets 1993) |
| `needs_clarification` | Residual-exceeds-tolerance flag |

The diagnostic that ties the framework together is the
**residual norm** ``‖r(B)‖`` — a unified scalar measuring distance
from the fixed point — and the **contraction factor** ``ρ_n =
‖r_{n+1}‖ / ‖r_n‖``, the headline iterative-method indicator
(Saad §4.1):

* ``ρ < 1``: contractive — successive iterations reduce residual.
* ``ρ → 1``: stalled — iterations not making progress; warrants
  strategy change (different fusion rule, different preconditioner,
  agent escalation).
* ``ρ > 1``: diverging — iterations growing the residual.

`bootstrap.residual_norm` and `bootstrap.contraction_rate`
implement the diagnostic. The unified residual is an L2
combination of four normalized components: mean(gap) /
gap_threshold, frac_unclear / clarity_target, mean(K) /
k_threshold, and frac(indep-tier disagreement at meaningful mass).
A residual_norm of 1.0 means "at convergence threshold across the
board"; values <1 are converged. Both are surfaced in
`IterationMetrics` and the agent loop's `iteration_history`.

This framing is what makes the rest of the design — non-
distinctness handling, hierarchical aggregation, ontology priors,
reliability shaping — operate as a cohesive accuracy-targeting
engine rather than a collection of clever heuristics. Each
mechanism is a numerical-method primitive in service of driving
the residual to zero.



## The non-distinctness problem

Dempster's rule of combination assumes the bodies of evidence being
combined are produced by **distinct, conditionally independent**
sources (Shafer 1976, *A Mathematical Theory of Evidence*, Ch. 3 §3
and Ch. 4). Smets' Transferable Belief Model (Smets 1990; Smets &
Kennes 1994, *The Transferable Belief Model*) preserves this
assumption at the credal level. Denoeux 2008 (*Conjunctive and
Disjunctive Combination of Belief Functions Induced by Non-Distinct
Bodies of Evidence*, Artificial Intelligence) characterizes the
pathology that arises when the assumption is violated: combining two
mass functions that derive from a shared evidential atom via
Dempster's rule effectively raises the contribution of that atom to a
power. The conjunctive cautious rule, defined on commonality
functions and idempotent on identical evidence (Denoeux 2008 §4),
recovers soundness — but is non-normalising and not a drop-in
replacement for Dempster.

## The Atelier-specific violation

The classification pipeline in `src/atelier/classify/` declares six
evidence sources:

1. `name_match` — lexical column-name matching against the vocabulary.
2. `pattern` — regex/validator detection (email, IBAN, monetary, …).
3. `cosine` — semantic similarity between the curated embedding text
   and the user-vocabulary embedding.
4. `llm` — Claude Opus first-pass classification.
5. `catboost` — CatBoost classifier.
6. `svm` — synth-trained TF-IDF + LinearSVC classifier with an
   LLM-mediated ICE → user-taxonomy alignment applied at inference
   time.

The first three are genuinely independent of the LLM: their evidence
arises from the column's name, value patterns, and semantic embedding
comparison against the vocabulary. The remaining sources have a
mixed independence profile:

- **`catboost`** is trained in `fit_to_llm` mode (default true) on
  `(embedding_text, llm_code)` pairs from the current run's LLM
  sweep. See `ml_train.fit_catboost_to_llm_labels` and
  `pipeline._install_fit_to_llm_catboost`. The fitted model is, by
  construction, an explainability surface over the LLM's labels —
  not a competing classifier. **Strongly non-distinct** with the
  LLM source under Denoeux 2008 (per-column shared label
  provenance).
- **`svm`** is trained once on the synthetic corpus
  (`scripts/generate_synth_source.py` → `ml_train.train_svm`),
  with TF-IDF char-3-6gram + word-1-2gram features and labels keyed
  on the bundled-ontology ICE.* leaves from
  `synth_generators.GENERATORS`. At pipeline runtime, predictions
  are translated into the user taxonomy via subsumption-prediction
  alignment in `classify.subsumption_alignment` — sentence-transformer
  cosine similarity between ICE concept signatures and enriched
  annotation payloads from the Qdrant taxonomy collection (one
  alignment computation per (vocab, embedding_model) tuple, results
  cached on disk). **Weakly non-distinct** with the cosine source
  via shared enrichment-LLM upstream — the enriched annotations were
  generated offline by an LLM, but the alignment computation itself
  uses a structurally independent model (BERT embeddings), not the
  runtime autoregressive LLM. The prior LLM-mediated approach (one
  LLM `classify_batch` call per alignment, excised in the P7
  subsumption-alignment intervention) was weakly non-distinct with
  the runtime LLM through shared model weights — the new approach
  eliminates that correlation.
  See the `ontology_alignment.py` module docstring for the full
  independence argument.

Treating LLM and CatBoost(LLM) as fully-independent sources and
combining them via Dempster's rule double-counts the LLM atom; the
SVM evidence sits between fully independent and fully derivative.
The pre-2026-04-30 discount schedule made the legacy three-way
overlap worse: `llm=0.10`, `catboost=0.10`, `svm=0.20`, vs
`cosine=0.30`. The genuinely independent semantic source was
*more* discounted than the two derivative ones, mathematically
suppressing it whenever the LLM was loud.

A failure case observed during pipeline validation illustrated the
pathology in the abstract. A column whose values match the
``monetary_pattern`` regex was classified as a generic catch-all
code rather than a financial-domain code. Cosine top-1 distributed
mass across several financial-leaning codes in the active
vocabulary, but at softmax-spread mass on the order of a few
thousandths per code it could not overcome LLM mass (≈ 0.83) and
CatBoost mass (≈ 0.81), both concentrated on the catch-all. The
fused prediction matched the LLM; the disagreement gate at
``bootstrap._identify_disagreements`` required
``llm_code != fused_code`` and so never fired despite K ≈ 0.81 and
a unanimous independent-source pull toward financial codes.
``needs_clarification=True`` was emitted, but no LLM revisit
followed. Specific customer table names, column names, and codes
are intentionally not reproduced in this document.

## Treatment in this codebase

The pipeline uses two complementary, scope-bounded fixes:

### 1. Reliability discounting on derivative sources (Shafer §11.3)

The discount operator from Shafer 1976 §11.3 multiplies a source's
mass by reliability `α = 1 - discount`:

> m'(A) = α · m(A); m'(Θ) = α · m(Θ) + (1 - α)

When evidence sources are non-distinct, the reliability of the
derivative source *with respect to the original* is bounded above by
1 minus their information overlap. For sources trained directly on
LLM output that overlap is near-total, so a substantial discount is
the principled response under classical Dempster fusion.

The current defaults (`config/base.conf:341+`) place CatBoost and
SVM **above** the cosine discount:

| Source       | Discount | Rationale                                                                            |
|--------------|----------|--------------------------------------------------------------------------------------|
| `cosine`     | 0.20     | independent of LLM; semantic prior                                                   |
| `pattern`    | 0.25     | independent; deterministic regex evidence                                            |
| `name_match` | 0.30–0.70 | independent; lexical match against vocab                                            |
| `llm`        | 0.15     | original; first-pass label                                                           |
| `catboost`   | **0.55** | **strongly non-distinct** (`fit_to_llm`, per-column LLM labels)                      |
| `svm`        | **0.22** | **weakly non-distinct** (enrichment-mediated subsumption alignment; was 0.30 under LLM-mediated, 0.55 under M9) |
| `catboost_max` | 0.75   | variance ceiling; maintains headroom                                                 |

Operators can dial these via the Settings page when retraining
CatBoost on labels independent of the current LLM sweep (e.g.
synth-only training); the metadata in `config_overlay.SETTINGS_METADATA`
exposes the full range.  The SVM discount at 0.22 (slightly above
cosine's 0.20) reflects the subsumption-prediction alignment:
structurally independent of the runtime LLM (uses BERT embeddings,
not autoregressive inference), with weak non-distinctness only via
the shared enrichment-LLM upstream (same structural dependency the
late-interaction cosine source carries).  The 0.02 margin above
cosine accounts for subsumption prediction being a single per-ICE-code
decision (structurally more brittle than per-column cosine evidence).

### 2. Independent-tier consensus + revisit gate

For revisit decisions, the pipeline computes a parallel, isolated
fusion over the LLM-independent subset only:

```
m_indep = m_cosine ⊕ m_pattern ⊕ m_name_match    (Dempster's rule)
indep_top1 = argmax_singleton m_indep
```

Implemented in `pipeline._classify_column` via the `INDEPENDENT_TIER`
constant and `combine_multiple(strategy="dempster")`. The top-1
singleton and its mass are exposed in the result dict
(`independent_top1_code`, `independent_top1_mass`,
`independent_top1_conflict`) and stored on the `BootstrapState`.

The revisit gate at `bootstrap._identify_disagreements` then fires
when:

- `indep_top1_code ≠ llm_code` AND
- `indep_top1_mass ≥ classify.bootstrap.indep_revisit_mass_threshold`
  (default 0.45)

This restores a real cross-source disagreement test that cannot be
masked by LLM-derivative sources amplifying the LLM's vote. The
legacy high-K branch (`llm_code != fused_code AND K > k_threshold`)
is retained as a safety net and runs second in priority.

The revisit prompt context at `bootstrap._llm_revisit` now includes
the independent-tier consensus code/label/mass so the LLM has the
counter-evidence in front of it during the second pass.

## Ontology priors — substrate as semantic anchor

Patterns detect at extraction time. When a pattern fires we resolve
its canonical ICE.* metadata from `universal_vocabulary.json` (label,
description, common-name aliases, full ontological path root→leaf)
and thread that metadata through three insertion points sourced from
a single lookup (`mass_functions.lookup_pattern_ontology`):

1. **Embedding text** (`features.ColumnFeatures.to_embedding_text` —
   `ontology_priors` is a discrete `FEATURE_NAMES` entry, ablatable
   for SAGE). Cosine similarity then operates over publicly-grounded
   ontology terms an embedding model recognizes from training rather
   than the regex name alone. On the failure case that motivated this
   work, the column embedding gains the literal substring "Transaction
   Amount; The monetary value of a financial transaction.; aliases:
   amount, payment, price; ontology: Sensitive Data → Personally
   Identifiable Data → Financial Data → Payment Data → Transaction
   Amount" — orders of magnitude more semantic surface than `patterns:
   monetary_pattern` carried.

2. **First-pass LLM user prompt**
   (`llm_backend.build_batch_user_prompt`). Every batch — sweep AND
   revisit — the prompt now includes per-column "Pattern-detected
   ontology priors (from Atelier's universal taxonomy — translate to
   the closest fit in the candidate vocabulary)" with each fired
   pattern's label, description, alias list, and path. The LLM is
   explicitly instructed that the canonical ICE.* code is never a
   valid classification target; its job is ontology alignment from
   the publicly-grounded substrate to the user's frame (He et al.
   2023, *Exploring Large Language Models for Ontology Alignment*;
   Hertling & Paulheim 2023, *OLaLa: Ontology Matching with LLMs*;
   Ehrig & Sure 2004 for the classical foundation).

3. **SAGE/SHAP attribution surface**
   (`features.FEATURE_NAMES`). `ontology_priors` is now its own
   ablatable feature distinct from `pattern_signals` and
   `sample_values`. Operators can attribute classification mass to
   the publicly-grounded ontology prior independently of the raw
   embedding text — the explainability story ties each prediction
   back to the public substrate that motivated it.

Surfaced in the result dict as `ontology_priors` (list of dicts:
`pattern, code, label, description, common_names, path,
match_fraction`). The codes are universal-substrate IDs; they never
appear in user-facing classifications. The user's vocabulary
remains the authoritative result space; ICE.* is the bridge.

Architectural significance: this is the substrate→tagging bridge
the design has been pointing at. Pattern detection was always
publicly-grounded; the resolver turns ICE.* into the user's codes
when it can; when it can't, ontology priors carry the public
semantic anchor straight through to cosine + LLM + SHAP without
ever fabricating a code in the user's frame. Compatible with — and
strengthens — the indep-tier consensus + reliability-discount
mechanisms above.

## Cosine reliability shaping (Haenni-Hartmann 2006)

Static `discount=0.30` allocated 0.70 of cosine mass uniformly via
softmax across all candidate singletons. On large vocabularies (300+
leaves) this produced softmax compression — even a sharp top-1 hit
landed at ~0.004 mass per code. Cosine could see the right answer
but couldn't carry it through fusion, and the indep-tier consensus
sat permanently below the revisit threshold.

`mass_functions.cosine_to_mass` now applies dynamic source
reliability per Haenni & Hartmann 2006, *Modeling Partially
Reliable Information Sources: A General Approach Based on
Dempster-Shafer Theory* (Information Fusion 7(4), 361–379, §3):
the source-reliability factor α is an observable function of
quality indicators, with `(1 − α)` allocated to ignorance.

Two quality indicators:

* **α_abs** — sigmoid of top-1 absolute similarity around `τ_abs =
  0.40` with `σ_abs = 0.10`. Encodes "is cosine matching anything
  strongly, or just noise?"
* **α_marg** — `tanh((s₁ − s₂) / σ_marg)` with `σ_marg = 0.05`.
  Encodes "is the top-1 a decisive winner, or ambiguous among
  similar candidates?"

Weighted blend (`w_abs = 0.6, w_marg = 0.4`), clamped to
`[reliability_floor, reliability_ceiling] = [0.10, 1 −
classify_discount_maxsim]`. The ceiling preserves the legacy
maximum-mass behavior under sharp signal; the floor keeps cosine
contributing some mass even under noise.

The α-bounded evidence mass is then split via **margin-aware
allocation**:

```
m(top-1) = α · margin_weight + α · (1 − margin_weight) · softmax_top1
m(top-i, i>1) = α · (1 − margin_weight) · softmax_top_i
m(Θ) = 1 − α
```

where `margin_weight = tanh((s₁ − s₂) / σ_marg)`. When the margin
is wide, almost all evidence mass concentrates on top-1 directly
rather than diluting through softmax. When the margin is narrow,
the formula reduces to classical softmax allocation across the
full candidate set.

Behavior across regimes (BDD-locked in
`features/agent/evidence_independence.feature`, "Cosine reliability
shaping concentrates mass on a clear top-1"):

| Top-1 sim | Top-2 sim | α | margin_weight | Top-1 mass | Θ mass |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.50 | 0.700 | 1.000 | **0.700** | 0.300 |
| 0.45 | 0.20 | 0.700 | 1.000 | **0.700** | 0.300 |
| 0.45 | 0.44 | 0.452 | 0.197 | 0.091 | 0.548 |
| 0.23 | 0.23 | 0.100 | 0.002 | 0.0005 | 0.900 |

Sharp signal recovers the legacy ceiling allocation but
concentrates it on top-1 (~170× the prior compressed mass).
Ambiguous and noise regimes correctly route most mass to Θ rather
than fabricating false confidence. The indep-tier revisit gate
(threshold 0.45) is now reachable on cosine alone whenever cosine
has clear semantic signal.

Composes cleanly with the indep-tier consensus computation: when
cosine carries decisive mass on a code different from the LLM's
vote, that code becomes the indep-tier top-1 and the revisit gate
fires — which is the soundness invariant the whole evidence-
independence treatment is reaching for.

## Hierarchical mass aggregation + cross-subtree visibility

A separate structural gap surfaced after reliability shaping
landed: when cosine evidence localizes to a *subtree* (multiple
financial-leaning leaves under a common parent) but the LLM picks
a confident leaf in a *different* subtree, the predicted code falls
to the LLM's leaf and there is no surfaced signal that an
honest-but-coarser parent would apply. Three cooperating fixes
close that gap:

### 1. Cosine emits hierarchical mass

`mass_functions.cosine_to_mass` now walks up from the cosine top-1
leaf, finds the most-specific internal node whose descendants
capture ≥ 50% of the softmax probability mass
(``_significant_subtree``), and redirects the in-subtree residual
mass to that internal-node focal element rather than diluting it
across leaves. The frame already exposed every parent code as an
internal-node ``FocalElement`` (descendant leaf set); we just
weren't *emitting* mass there. Hierarchical Dempster-Shafer
treatment per Shafer 1976 §3 and Smets 1990 §6 (refinement /
coarsening): an internal-node focal element represents a
disjunction — "the answer is somewhere in this subtree" — without
committing to a specific leaf.

Walking up from top-1 (rather than requiring every top-K to share
an LCA) tolerates outliers cleanly: a small amount of probability
leaking outside the subtree doesn't void the aggregation as long
as the bulk of mass remains inside.

Sharp-signal regimes are unaffected — when the margin is wide
the residual mass `α · (1 − margin_weight)` is small, so the
hierarchical aggregation simply scales proportionally. The
top-1 leaf still wins when one is decisive.

### 2. `cautious_code` walks the full hierarchy

`HierarchicalClassification.cautious_code` previously walked only
the predicted code's ancestor chain via `belief_path` —
structurally blind to belief mass in any other subtree. It now
delegates to `cross_subtree_belief`, which iterates every
singleton AND every internal-node focal element in the frame and
returns those with ``Bel ≥ threshold``. The most-specific code
wins, regardless of subtree.

Concretely: when the LLM votes ``0.1 Internal Non-Sensitive`` but
cosine's hierarchical aggregation puts ``Bel(Financial Data) =
0.55`` on a different subtree's parent, ``cautious_code(0.5)`` can
now return ``Financial Data`` — not just ``0`` (the predicted
code's parent).

### 3. `cross_subtree_belief` surfaces the conflict

The result dict now carries a ``cross_subtree_belief`` field
listing every code (leaf or internal node, any subtree) where
``Bel ≥ 0.5``. Operators see both the LLM's leaf vote AND the
cosine-derived alternative subtree as legitimate signals,
instead of the predicted-leaf-only ``belief_path``. When evidence
sources disagree on the *subtree*, both candidates appear and the
operator can act on the disagreement directly.

This composes cleanly with the prior mechanisms: reliability
shaping ensures cosine top-1 carries enough mass to trigger
hierarchical aggregation when signal is clear; the indep-tier
gate fires when cosine's hierarchical mass disagrees with LLM at
the leaf level; and ``cross_subtree_belief`` makes the cross-
subtree disagreement explicit in the operator-facing result. The
``predicted_code`` field retains its leaf-argmax semantics for
backward compatibility — operators consume the cautious /
cross-subtree fields when ``needs_clarification = True`` or when
the cross-subtree summary surfaces a competing internal node.

## Operator-facing visibility

The fusion mechanisms above can produce mathematically correct
belief structures that are nonetheless invisible to operators
when the result-dict surface area is too narrow. Three small
changes close that gap:

### Evidence string carries per-source codes + competing summary

`HierarchicalClassification.from_combined_evidence` builds the
``evidence`` field. Previously: ``dst(cosine=0.65, llm=0.77,
catboost=0.42, svm=0.22) → Internal Non-Sensitive [Bel=0.71,
...]`` — masses only, not the codes each source voted. Now:
``dst(cosine→1.4.1.1.1(0.65), llm→0.1(0.77), ...) → Internal
Non-Sensitive [Bel=0.67, ...] [competing: Sensitive (1)
Bel=0.26]`` — leaf-level disagreement is visible at a glance,
and a "competing" trailer surfaces non-trivial belief in any
non-predicted top-level subtree.

### `cross_subtree_belief` is always informative

The 0.5 absolute threshold previously suppressed competing-
subtree alternatives whenever Dempster fusion compressed their
mass below the headline bar (the common case when one source
dominates). The default is now lower (0.20) AND a
``always_include_top_per_subtree`` rule guarantees that the
highest-belief leaf and highest-belief internal node from each
top-level subtree appears in the result regardless of
threshold (subject to a small ``min_bel`` floor so we don't
flood the result with noise). Operators always see the
structured "what does each subtree look like?" view.

### `cautious_promoted_code` (Smets least-commitment)

Per Smets 1993 (*Belief Functions: The Disjunctive Rule of
Combination and the Generalized Bayesian Theorem* and related
work on least-commitment), when a fine-grained decision is
unsupported by evidence the principled response is to commit
only at the level of granularity where evidence IS unambiguous.
This is exactly the mechanism for "the predicted leaf is not
the right answer; the parent code is more honest."

`HierarchicalClassification.cautious_promoted_code` returns
either the predicted leaf (no promotion) or the most-specific
code anywhere in the hierarchy whose belief meets the
``commit_threshold`` (default 0.55). Promotion fires only when
``needs_clarification = True`` — operators get the leaf
prediction by default; the cautious promotion is the
epistemically-honest fallback when the system itself flags the
prediction as uncertain.

The ``predicted_code`` field retains its leaf-argmax semantics
for backward compatibility with Atlas governance sync and
existing UI rendering. ``cautious_promoted_code`` lives
alongside it as a separate field operators consult when
``needs_clarification`` is True.

## Per-column residual trajectory

The corpus-wide residual norm + contraction factor establish the
headline iterative-method diagnostic, but they obscure per-column
behaviour. `BootstrapState.column_history: dict[str, list[ColumnResidualSnapshot]]`
captures the column-major view: one snapshot per labeled column per
iteration, populated in `record_iteration_metrics` after each
iteration's ML validation completes. Each snapshot records the
column's gap, belief, K, indep-tier top-1 code/mass, label, label
source, and a `revisited` flag indicating whether
`_llm_revisit` touched the column in that iteration.

`bootstrap.column_contraction(state, name)` mirrors the corpus-wide
`contraction_rate` at the column level: ρ_col = current_gap /
prev_gap (falling back to K when gap is zero), or `None` when the
column has fewer than two snapshots. ρ_col < 1 means the column is
converging; ρ_col → 1 stalled; ρ_col > 1 diverging. Per-column ρ
exposes the empirical contraction distribution that corpus
aggregates obscure — operators see *which specific columns* are
converging vs stalling.

The full trajectory is written to
`build/results/{run_id}/column_trajectories.json` at pipeline end
alongside `classifications.json`, enabling offline analysis,
operator post-mortem, and audit. The agent loop's
`iteration_history` carries a summarized view (per-column
gap/bel/K sequences plus ρ_col) so the agent can reason about
which columns are moving.

This trajectory infrastructure is the **substrate** for any
future acceleration scheme. Three plausible Phase B / Phase C
extensions all consume it:

* **Bandit-style revisit ordering** (Phase B) — extend
  `_identify_disagreements` to mix `expected_revisit_gain(name)`
  derived from history into the sort key. Revisits ordered by
  predicted marginal residual reduction. Default-off knob;
  trajectory data backs it.

* **Aitken Δ² early-stop** (Phase B) — for columns with ≥3
  snapshots and a clean linear-convergence pattern, predict the
  limit and skip further revisits when the predicted gap is
  below `cfg.gap_threshold`. Saves LLM cost on the predictable
  tail. Default-off knob; trajectory data backs it.

* **Limited per-column belief-mass Anderson** (Phase C, deferred)
  — only on columns that genuinely oscillate (per-column ρ near 1
  with sign-changing residual differences). Phase A's trajectories
  let us measure whether such a population exists before shipping
  any Anderson code.

The honest framing: classical Anderson acceleration on the full
belief-vector iteration is poorly suited to LLM-driven dynamics
(stochastic T, mostly-static state, discrete labels, targeted-not-
uniform revisit). What's value-add given the problem structure is
the per-column trajectory data itself — operators see per-column
convergence behaviour, future acceleration schemes have real
per-column data to operate on, and we can decide between bandit /
Aitken / Anderson empirically rather than rhetorically. Phase A
ships that substrate; Phase B and Phase C are gated on what the
substrate reveals.

## Cost-sensitive classification at the LLM layer (Elkan 2001)

All the prior mechanisms operate at or below the fusion layer —
they shape *how* per-source evidence is combined into a fused
belief.  But on the canonical failure case
(`loan_applications.requested_amount`), the LLM at confidence 0.88
plus its derivative cluster (CatBoost, SVM) reinforces a vote on
`0.1 Internal Non-Sensitive`, and Dempster fusion's normalization
preserves that dominance.  Algorithmic mitigations stalled at
Bel ≈ 0.74 — an honest reduction from Bel = 0.955 baseline, but
the headline classification still miscategorized financial PII.

The principled response, per **cost-sensitive classification**
(Elkan 2001, *The Foundations of Cost-Sensitive Learning*) is to
adjust the decision threshold under asymmetric cost.  In data
governance the asymmetry is severe: failing to flag truly
sensitive data (false negative, Type II) creates regulatory
liability (GDPR Art. 25 data protection by default; HIPAA Safe
Harbor; PCI DSS scope creep guidance), while over-classifying
(false positive, Type I) produces review overhead but is
recoverable.  Treating the costs as `cost(FN) ≫ cost(FP)` is the
canonical privacy-regime convention.

Atelier applies this at the **LLM layer** — upstream of fusion —
via a *Sensitivity classification perspective* section in the
system prompt (`llm_backend.build_system_prompt`). The framing is
deliberately collaborative rather than prescriptive: modern LLMs
respond better to a colleague's framing than to a compliance
checklist. Three load-bearing moves:

- **Invoke what the LLM already knows.** The preamble names BFO,
  CCO, and the privacy regimes (GDPR, HIPAA, PCI DSS) those
  ontologies overlap with — concepts the model has substantial
  training exposure to. The customer's taxonomy is framed as
  "their refinement of those publicly-grounded concepts," and
  the model is asked to pick whichever of their codes matches
  the canonical sensitivity concept it would otherwise assign
  (PII, Financial Information, Technical Identifier, Biometric,
  etc.). No re-teaching, no rule list — invocation.

- **State the asymmetry once, casually.** Cost-sensitive
  classification appears as "a practical asymmetry: in
  governance, calling sensitive data non-sensitive is a larger
  error than the reverse." The over-classification guard is
  embedded conversationally: "When signals are genuinely absent
  (operational metadata, surrogate keys, timestamps, status
  enums), non-sensitive is the correct call — don't reach for
  sensitive just because of the asymmetry." One sentence on
  confidence calibration: "Calibrate confidence to what you
  actually saw, not to this asymmetry."

- **Vocabulary-aware sensitivity map, ICE conventions only.**
  `_sensitive_subtree_summary(category_set)` activates on
  `ICE.SENSITIVE.*` / `ICE.NONSENSITIVE.*` paths and emits a
  Markdown block naming the sensitive root, catch-all, and a
  few publicly-grounded leaf abbreviations (per
  `src/atelier/classify/fixtures/PROVENANCE.md`). Returns `""`
  for every other vocabulary shape so the prompt stays silent
  where the framework can't verify the encoding is publicly
  grounded. For non-ICE vocabularies the LLM still has the full
  markdown category table, per-column ontology priors for
  pattern-bearing columns, and the perspective preamble — that
  is sufficient to navigate any taxonomy without the framework
  guessing at its sensitivity structure.

The prompt block is **default-on** for every classification run;
no config knob.  Built once per pipeline run at
`pipeline.py:577` so the helper computation is amortized and the
new content lives inside the Anthropic prompt-cache prefix —
one-time cache miss on the first batch, normal cache hits
thereafter.  Token cost is bounded (~250–300 fixed + ~80 for
the per-vocab summary).

This composes cleanly with everything below it: reliability
discounts on derivative sources still suppress double-counting,
cosine reliability shaping still concentrates mass on clear
top-1 hits, hierarchical aggregation still flows residual mass
to internal-node focal elements, the indep-tier consensus gate
still triggers revisits on cross-source disagreement, and
`cautious_promoted_code` still applies Smets least-commitment
on uncertain leaves.  The Governance Cost Model changes what
the LLM votes — biasing toward sensitive parents under
uncertainty — leaving every downstream mechanism unchanged.

The hypothesis: with a governance prior at the source, the LLM
will either (a) pick a defensible sensitive parent code on
columns like `requested_amount`, or (b) lower its confidence on
the non-sensitive choice — either of which is an improvement
over the status quo.  The exact behavior is non-deterministic
and confirmed against real LLM runs; BDD scenarios assert the
prompt structure (`features/agent/governance_cost_model.feature`),
not the LLM's vote.

## Pattern-target alias resolver

A second, narrower bug surfaced during investigation: the static
`DEFAULT_PATTERN_MAP` at `mass_functions.py` references canonical
ICE.* mnemonic strings (`monetary_pattern → ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT`)
that are absent from non-ICE vocabularies. The pre-2026-04-30
behavior silently dropped any pattern whose target wasn't in
`frame.singletons`, disabling the entire pattern source on numeric
or domain-specific vocabularies — including the run that motivated
this work.

`mass_functions.resolve_pattern_map` now resolves each ICE.* target
through three fallback layers against the active `category_set`:

1. Direct hit on `all_by_code`.
2. Match on `by_abbrev` using the leaf mnemonic (suffix after the
   final `.`).
3. Token-normalized match against `common_names` aliases.

Misses log a single `WARNING` enumerating the patterns that were
dropped. The resolver is cached on the `category_set` instance and
runs once per pipeline. The deeper BFO/Common-Core ontology mapping
this shim approximates remains future work.

## Deferred work

This treatment preserves Dempster's rule end-to-end and handles
non-distinctness through reliability discounting + per-source
reliability shaping. One future refinement remains scoped out:

- **Tiered fusion with the cautious rule (Denoeux 2008).** Combine
  the LLM-derivative cluster `{llm, catboost, svm}` via cautious
  conjunction (idempotent on identical evidence; commonality
  formulation `q1 ∧̂ q2`), the independent cluster `{cosine, pattern,
  name_match}` via Dempster, and combine the two cluster-level mass
  functions across-tier. This dissolves the non-distinctness problem
  at the math level rather than approximating it via discount.
  Trade-off: cautious is non-normalising, so derivative-tier-only
  columns will see narrower belief intervals (which is *correct*
  behaviour but a UI shift).

The `combine_multiple` infrastructure already supports adding a
`strategy="cautious"` branch alongside the existing `dempster` /
`yager` options, so the refinement is surgical when it lands.

## References

- Shafer, G. (1976). *A Mathematical Theory of Evidence.* Princeton
  University Press. Ch. 3 §3 (independence assumption); Ch. 4 §3
  (Dempster's rule); §11.3 (reliability discount).
- Smets, P. (1990). The Combination of Evidence in the Transferable
  Belief Model. *IEEE Transactions on Pattern Analysis and Machine
  Intelligence* 12(5), 447–458.
- Smets, P. & Kennes, R. (1994). The Transferable Belief Model.
  *Artificial Intelligence* 66(2), 191–234.
- Denoeux, T. (2008). Conjunctive and Disjunctive Combination of
  Belief Functions Induced by Non-Distinct Bodies of Evidence.
  *Artificial Intelligence* 172(2-3), 234–264. §1, §3.1, §4.
- Haenni, R. & Hartmann, S. (2006). Modeling Partially Reliable
  Information Sources: A General Approach Based on Dempster-Shafer
  Theory. *Information Fusion* 7(4), 361–379.

## Operational impact

Operators upgrading to this calibration should expect:

- More columns marked `needs_clarification=True` on the first run
  after upgrade. This is the *intended* outcome: derivative-source
  amplification no longer hides genuine cross-source conflict.
- A modest increase in LLM revisit volume (the gate fires on a
  wider, principled condition). Mitigated by the
  `indep_revisit_mass_threshold` floor and the existing budget caps
  at `classify.bootstrap.max_total_llm_calls` /
  `max_total_llm_attempts`.
- A pattern-source `WARNING` at startup enumerating any patterns
  whose ICE.* target failed to resolve to the active vocabulary.
  Acceptable as long as the leaf mnemonics that *do* exist in the
  vocab carry the relevant `abbrev` or `common_names` aliases —
  expected on first run with a domain-specific vocabulary.
