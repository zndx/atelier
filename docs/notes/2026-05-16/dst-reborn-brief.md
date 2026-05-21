<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# DST Reborn: Restoring Evidence Independence in Atelier's Classification Pipeline

**Brief — 2026-05-16**

---

## Abstract

Atelier's classification pipeline is a Dempster-Shafer (DST) evidence-fusion
system over a runtime-selected hierarchical category set.  Vocabulary
identity is dynamic: operators choose a `(connection, database,
annotations_table)` triple per run, and the architecture must scale
across vocabularies of unbounded size and shape.  The one structural
invariant is that every node — leaf or internal — is a first-class
tagging target.  Empirical observation,
corroborated by an audit of the in-loop training topology, shows that of six
declared evidence sources only three are structurally independent of the LLM;
the remainder are downstream of LLM labels via in-loop training.  Under
Dempster's rule of combination (Shafer 1976, Ch. 4 §3) this violates the
conditional-independence requirement and produces *false consensus*: low
conflict K, high apparent belief, no error-correcting power.  The system's
strict-match accuracy on adversarial enterprise corpora regressed from
~97 % (pre-target-space expansion, structurally independent) to ~78 %
(current, structurally collapsed) — a delta dominated by independence loss
rather than target-space difficulty.

This brief documents the architectural remediation now landed on
`feat/dst-late-interaction-cosine`: a multi-vector late-interaction
(Khattab & Zaharia 2020) cosine evidence source backed by Qdrant, an
LLM-mediated annotation enrichment pipeline paired with deterministic
verifier suite, and — most consequentially — a **channel-decomposed
Dempster combination** in which positive evidence (label / prototype /
name-hint MaxSim) and negative evidence (anti-example MaxSim) produce two
mass functions over distinct focal-element structures (singletons and their
complements, respectively) and combine via Dempster's rule.  The
construction restores DST conflict K as a first-class signal — the
fusion's native flag for "evidence sources disagree" — and re-enables the
downstream mechanisms (cross-subtree visibility, cautious promotion,
indep-tier revisit) that depend on it.

We further document the **harness principle** — that the cognitive system
performing classification is the (model + verification machinery) composite,
not the model alone — and its operational corollary, **no silent
degradation**, which forbids default-off feature flags and silent fallback
paths for DST-restoration enhancements.  Near-term work (foundation-model
integration via Ægir, in-situ SVM training on procedurally-labeled
synthetic data via Synthia, repositioning the LLM as a targeted re-evaluator
rather than a primary evidence source) is sketched.

---

## 1. Background

### 1.1 The Atelier classification problem

Atelier classifies columns of relational tables (entities) into terms drawn
from a hierarchical category set selected at runtime by the operator
(the *Atlas Lexicon* is the BFO/CCO-grounded reference used in the
current UAT; other vocabularies will be loaded across deployments and
test slices).  The current UAT snapshot carries roughly three-hundred
nodes; this is a property of today's data, not of the architecture.
Each column receives a
single hierarchical classification with belief, plausibility, and conflict.
The fusion stage combines evidence from multiple sources via Dempster-Shafer
theory; the bootstrap loop iterates fusion + targeted LLM revisitation until
convergence on a belief-gap criterion.

### 1.2 Mass functions and the algebra of evidence

Let `Θ` denote the frame of discernment — for Atelier, a finite set of
leaf-code singletons together with internal-node focal elements
representing parent categories.  A *mass function* is a map

$$m : 2^{\Theta} \to [0, 1], \quad m(\emptyset) = 0, \quad \sum_{A \subseteq \Theta} m(A) = 1.$$

A subset `A` with `m(A) > 0` is a *focal element*.  Belief and plausibility
are derived:

$$\mathrm{Bel}(A) = \sum_{B \subseteq A} m(B), \qquad \mathrm{Pl}(A) = \sum_{B \cap A \neq \emptyset} m(B).$$

Dempster's rule of combination is

$$(m_1 \oplus m_2)(C) = \frac{1}{1 - K} \sum_{A \cap B = C \neq \emptyset} m_1(A) \, m_2(B),$$

with the **conflict** $K = \sum_{A \cap B = \emptyset} m_1(A) \, m_2(B)$
and the operator undefined when `K = 1` (total conflict).

### 1.3 The categorical structure

For a fixed frame `Θ`, mass functions form a set `Mass(Θ)`; Dempster's rule
defines a partial binary operation $\oplus : \mathrm{Mass}(\Theta) \times
\mathrm{Mass}(\Theta) \to \mathrm{Mass}(\Theta)$ that is commutative and
associative *when restricted to the subcategory of conditionally
independent sources* (Shafer 1976, Ch. 3 §3).  The vacuous mass
$m_\Theta(\Theta) = 1$ is the identity.  Conditional on independence,
$(\mathrm{Mass}(\Theta), \oplus, m_\Theta)$ is a commutative monoid; this
is the algebraic structure on which the entire pipeline's evidence-fusion
arithmetic relies.

Yager's rule (Yager 1987, expounded by Smets 1990) defines an alternative
binary operation in which conflict mass is redirected to ignorance rather
than normalized away.  Yager's rule is associative and commutative
*unconditionally* — it is the categorical "honest" combination, at the cost
of losing the conflict-as-error-detection property Dempster's rule
provides.  Both operations are exposed in the codebase (`dempster_combine`,
`yager_combine`); the choice between them is consequential, as §6 will
make precise.

### 1.4 The independence axiom and its violation

Dempster's rule presupposes that the bodies of evidence are produced by
**distinct, conditionally independent** sources.  Denoeux 2008
(*Conjunctive and Disjunctive Combination of Belief Functions Induced by
Non-Distinct Bodies of Evidence*, AIJ 172) characterizes the pathology:
combining two mass functions that derive from a shared evidential atom
via Dempster's rule effectively raises the contribution of that atom to a
power, producing *false consensus* — sources agreeing because they are
echoes of one another, not because they have observed the same truth
independently.

Smets (1990, *The Combination of Evidence in the Transferable Belief
Model*) preserves the independence assumption at the credal level and
introduces the **cautious rule** — defined on commonality functions and
idempotent on identical evidence — as the principled treatment of
non-distinctness.  Cautious combination is non-normalising and not a drop-in
replacement for Dempster's rule; for Atelier, it remains documented future
work (Denoeux 2008 §4) while we address the more tractable structural cause:
*make the sources actually independent.*

---

## 2. The Current Pathology

### 2.1 Source-by-source independence audit

Atelier's pipeline (`src/atelier/classify/`) declares six evidence sources:

| Source       | Construction                                                  | Independence wrt LLM        |
|--------------|---------------------------------------------------------------|-----------------------------|
| `name_match` | Lexical matching of column name against vocabulary            | **Strong**                  |
| `pattern`    | Regex / validator detection (email, IBAN, Luhn-CC, IPv4)      | **Strong**                  |
| `cosine`     | Sentence-transformer embedding similarity column ↔ annotation | **Strong** (pre-augmentation; degraded after) |
| `llm`        | Claude Sonnet first-pass classification                       | (the reference source)      |
| `catboost`   | Trained `fit_to_llm` on `(embedding_text, llm_code)` pairs from the *current run* | **Strongly non-distinct** |
| `svm`        | Trained on synthetic corpus → ontology-aligned via LLM-mediated mapping | **Weakly non-distinct** |

The first three are genuinely independent.  The CatBoost training artifact
on disk (`catboost_fit_to_llm.cbm`) declares the dependence in its
filename: the model is a learned approximation of the LLM's classification
function over the run's embedding text.  Combined with the LLM via Dempster's
rule, CatBoost's mass is a near-identical echo of the LLM's vote — exactly
the pathology Denoeux 2008 §1 identifies.  The SVM's dependence is weaker
(vocabulary-level, via an LLM-curated alignment table) but non-zero.

### 2.2 Quantitative cost

The 920-column agent-mediated reference artifact (produced by an independent
Opus 4.7 reasoning session, paired with audit metadata) gives
an upper bound on the current pipeline's accuracy on adversarial enterprise
corpora.  Atelier-current scores 481/618 = **77.8 %** strict match (Cat A
78.9 %, Cat B 76.8 %).  Pre-expansion ablation studies on smaller target
spaces showed ~97 % strict accuracy when the four-source DST stack
(name_match, pattern, cosine, LLM) was genuinely independent.  The
~19 percentage-point gap decomposes approximately as:

- ~5 points target-space difficulty (4 → hundreds of codes; UAT
  snapshot at the time of writing)
- ~14 points structural collapse of evidence independence

A four-hour diagnostic sweep (`bel_threshold × gap_threshold` 12-cell grid,
launched 2026-05-16T14:31Z) was still in progress at brief authorship.  Its
contribution is characterization of the broken architecture's hyperparameter
sensitivity, not an operating-point recommendation: with non-independent
sources, no hyperparameter setting can restore the lost error-correcting
power.

### 2.3 Why the pathology resisted prior mitigation

The pipeline already carries substantial DST-independence machinery
(documented in `docs/src/architecture/dst-evidence-independence.md`):
reliability discounting per source (Shafer §11.3), the **independent-tier
consensus + revisit gate** that fuses {cosine, pattern, name_match} in
isolation and fires LLM revisit on cross-tier disagreement, cosine
reliability shaping per Haenni-Hartmann 2006, hierarchical mass aggregation
with cross-subtree visibility, and cost-sensitive LLM prompting per
Elkan 2001.  Each mitigation is locally correct.  None addresses the root
cause: that two of the six sources are *trained on the very labels they
purport to validate*.  The discount on CatBoost (0.55) and SVM (0.30)
attenuates the false consensus rather than dissolving it; the indep-tier
gate restores a real cross-source disagreement test, but it operates on the
three sources whose discriminative coverage shrinks dramatically as the
target space grows.  The pipeline can detect the conflict it is denied; it
cannot generate the independent evidence the rule requires.

---

## 3. Architecture: Late-Interaction Multi-Vector Cosine via Qdrant

### 3.1 The motivating gap

The legacy cosine source compresses each annotation into a single embedding
from `label + mnemonic + description` and compares it to a single column-
side embedding from `column_name + concatenated_samples`.  On adversarial
corpora — anonymized column names (e.g., `comm_val`, `period_val`,
`addr_ref` in the reference POC schema), mixed-type sample distributions,
vocab-token-as-data columns — the single-vector representation collapses
discriminative signal before it reaches the fusion layer.  Reliability
shaping correctly routes mass to ignorance under noise but cannot recover
signal that compression destroyed.

### 3.2 Multi-vector annotation profiles

Each annotation is materialized in Qdrant as a single point carrying
seven *named vector slots* — three single-vector and four multi-vector:

| Slot                | Cardinality | Source                                              |
|---------------------|:-----------:|-----------------------------------------------------|
| `label_view`        | single      | `embed(label + mnemonic + description)`             |
| `description_view`  | single      | `embed(description)`                                |
| `parent_path_view`  | single      | `embed("Root > ... > Self")`                        |
| `prototype_values`  | multi       | per-prototype-value embeddings (10–20 per tag)      |
| `value_patterns`    | multi       | per-format-hint embeddings (regex, length stats)    |
| `name_hints`        | multi       | per-likely-surface-form embeddings (incl. anonymized) |
| `anti_examples`     | multi       | per-anti-example embeddings (with confusable_tag)   |

The enrichment fields (`prototype_values`, `value_patterns`, `name_hints`,
`anti_examples`) are produced by an LLM-mediated enrichment pipeline
(`scripts/enrich_annotations.py`) paired with a procedural verifier suite
(§5).  The point's payload carries provenance metadata (augmentation
version, embedding model identity, verifier results, operator-edit log) so
the artifact satisfies the *LLM-mediated reference* discipline.

### 3.3 Column-side multi-vector representation

A column is similarly decomposed into a multi-vector query:

| Slot                | Source                                              |
|---------------------|-----------------------------------------------------|
| `col_name_view`     | `embed(column_name + " in " + table_name)`          |
| `col_sample_*`      | per-deduped-sample embeddings (top-N frequency)     |
| `col_context_view`  | `embed("table columns: " + neighbors)`              |
| `col_pattern_view`  | `embed(extracted format hints from samples)`        |

`col_pattern_view` is the structural reabsorption of regex's *original*
architectural role (feature enrichment for the column-side embedding) that
had been promoted into a standalone evidence source as a stand-in for an
H-Net-based independent learned source still in development (see §7.4).

### 3.4 Late-interaction MaxSim

For a column query $Q = \{q_1, \ldots, q_M\}$ and an annotation document
$D = \{d_1, \ldots, d_K\}$, ColBERT-style late interaction
(Khattab & Zaharia 2020, SIGIR '20) defines

$$\mathrm{score}(Q, D) = \sum_{m=1}^{M} \max_{k=1..K} \mathrm{sim}(q_m, d_k).$$

Atelier specializes this with **within-corresponding-role** restriction:
MaxSim is computed only between query and document vectors of matched
semantic roles (`col_name_view` ↔ `label_view` and `name_hints`;
`col_sample_*` ↔ `prototype_values` and `value_patterns`;
`col_context_view` ↔ `parent_path_view`; `col_pattern_view` ↔
`value_patterns`).  Cross-role comparisons (e.g., column samples against
parent_path_view) are excluded, preserving the structured-evidence
interpretation and preventing spurious cross-role hits.  Per-query-vector
normalization (sum of MaxSims divided by query-vector count) preserves
cross-column comparability — standard ColBERT practice (Khattab & Zaharia
2020 §3).

---

## 4. The Channel-Decomposed Dempster Construction

### 4.1 Positive and negative channels

For each candidate tag $x \in \Theta$, late-interaction execution produces
two scalar scores:

- $\sigma^+_x$ — *positive* score, aggregated from MaxSim of column-side
  vectors against the annotation's label, prototype, name-hint, value-
  pattern, and context slots (with per-role weights).
- $\sigma^-_x$ — *negative* score, MaxSim of column-side sample vectors
  against the annotation's `anti_examples` slot.

The positive channel answers *"does this column look like an instance of
$x$?"*; the negative channel answers *"does this column look like a known
confusable-with-$x$?"*.  These are conditionally independent questions
under the truth — the LLM-augmented annotation profile generated them from
disjoint reasoning prompts, and the procedural verifier suite confirmed
they reference disjoint exemplars (no value appears in both
`prototype_values` and `anti_examples` of the same tag).  The independence
property is preserved by construction, not by hope.

### 4.2 Two mass functions over different focal-element structures

The positive channel produces a mass function on **singletons** via the
existing reliability-shaped allocation (`cosine_to_mass`, embedding the
Haenni-Hartmann 2006 α-bounded reliability shaping):

$$m^+ : \{ \{x\} : x \in \Theta \} \cup \{\Theta\} \cup \{\text{hierarchical internal nodes}\} \to [0, 1].$$

The negative channel produces a mass function on **complement focal
elements** $\Theta \setminus \{x\}$ for the top-K tags ranked by
$\sigma^-_x$:

$$m^-(\Theta \setminus \{x\}) = \beta \cdot \frac{\sigma^-_x}{\sum_{y \in \mathrm{topK}} \sigma^-_y}, \quad m^-(\Theta) = 1 - \beta.$$

The complement focal element encodes "the truth lies somewhere other than
$x$" — the DST-natural shape of *evidence against* a tag.

### 4.3 Combination

The two channels combine via Dempster's rule (the codebase calls
`dempster_combine(m_+, m_-)`).  Because the focal-element structures are
distinct (singletons vs complements), the intersections that matter are:

- $\{x\} \cap (\Theta \setminus \{x\}) = \emptyset$ — **conflict**.  Positive
  evidence supporting $x$ and negative evidence against $x$ produce a
  product that lands in $K$.  This is the semantic signal the net-score
  approximation silently destroyed.
- $\{x\} \cap (\Theta \setminus \{y\}) = \{x\}$ for $y \neq x$ —
  **reinforcement**.  Negative evidence against $y$ combined with positive
  evidence for $x$ leaves $\{x\}$ intact (and contributes a small focal
  element $\Theta \setminus \{y\}$ to the result).
- $\Theta \cap (\Theta \setminus \{x\}) = \Theta \setminus \{x\}$ — the
  negative evidence persists as a refinement of ignorance when positive
  evidence is itself uncommitted.

Under total conflict ($K = 1$), Dempster's rule is undefined.  The
implementation falls through to Yager's rule, which redirects $K$ mass to
$\Theta$ (ignorance) — Smets' least-commitment principle (Smets 1993).
This is a DST math edge case, not a deployment-degraded fallback.

### 4.4 Empirical validation

A four-leaf taxonomy smoke ({EMAIL, PHONE, NAME, INOS}) confirms the
arithmetic:

| Case | $\sigma^+$ | $\sigma^-$ | Result (top focal elements) |
|------|------------|------------|------------------------------|
| Positive on EMAIL, no negative | EMAIL: 0.80 | — | EMAIL = 0.80, Θ = 0.20 |
| Positive EMAIL + negative PHONE | EMAIL: 0.80 | PHONE: 0.85 | EMAIL = 0.80, {EMAIL, INOS, NAME} = 0.06, Θ = 0.14 |
| Positive EMAIL + negative EMAIL (conflict) | EMAIL: 0.80 | EMAIL: 0.85 | EMAIL = 0.737, {INOS, NAME, PHONE} = 0.079, Θ = 0.184; $K \approx 0.24$ |
| Empty | — | — | Θ = 1.0 |

The conflict case exhibits the expected Dempster renormalization: EMAIL's
mass attenuated by $1/(1-K)$, complement focal element surfaces with
proportional mass, $\Theta$ enlarged.  The conflict $K \approx 0.24$ is
exposed to downstream mechanisms (cross-subtree visibility,
`cautious_promoted_code`, the indep-tier revisit gate) so the
"channels disagree" signal acts on the rest of the pipeline natively.

---

## 5. The Harness Principle

### 5.1 The cognitive system is the harness, not the model

The architectural improvements above are necessary but not sufficient.  The
deeper invariant — load-bearing for the entire programme — is that **the
cognitive system performing classification is the (model + verification
machinery) composite**, not the model alone.

Concretely: an LLM invoked through a batch API that returns one tagged
output per column is a different epistemic system than the same LLM
invoked through an Agent SDK harness that runs procedural verifications
in-loop, accumulates audit trails, cross-references prior outputs, and
falsifies generations that fail deterministic checks.  The model is the
substrate; the harness is the body and effectors.  This framing — that
agents have a *homuncular flexibility* (Lanier 1985, formalized by Won et
al. 2015 in *Homuncular Flexibility: The Human Ability to Inhabit
Nonhuman Avatars*, JCMC 20 §3) — is more than analogy: the cognitive
properties that emerge from harnessing (verifiability, reproducibility,
error correction) are properties of the system, not of the model alone, and
removing the harness removes those properties even when the model is
preserved.

A category-theoretic gloss: if $\mathcal{M}$ is the category of LLM
inference invocations and $\mathcal{S}$ is the category of agent-system
behaviors, the harness is a functor $H : \mathcal{M} \to \mathcal{S}$ that
preserves composition while enriching the morphism structure.  Two
invocations of the same model through harnesses $H_1$ and $H_2$ produce
non-isomorphic objects in $\mathcal{S}$ even when their preimages in
$\mathcal{M}$ are identical.  This is the formal content of "the model is
not the mind."

### 5.2 LLM-mediated reference artifacts

The enrichment pipeline produces *LLM-mediated reference artifacts*:
structured outputs (prototype values, value patterns, name hints,
anti-examples per annotation) generated by an LLM under a prompted
specification, *paired with* a procedural verifier suite that confirms
each generated field is well-formed, self-consistent, and consistent with
the taxonomy.  The artifact's epistemic legitimacy comes not from the
model that produced it but from the harness's ability to reproduce it
deterministically from its inputs (content-addressed cache keying:
$\mathrm{sha256}(\mathrm{taxonomy} \| \mathrm{version} \| \mathrm{row} \|
\mathrm{model})$) and falsify it against deterministic checks.

The verifier suite (six checks: `patterns_compile`,
`prototype_values_match_patterns`, `anti_example_targets_exist`,
`parent_path_consistent`, `name_hints_non_empty`,
`no_contradiction_with_anti_examples`) is the procedural component of the
harness.  Generations that fail any check are rejected and the loop
re-prompts with the failure as feedback.

### 5.3 Supervision by construction

The deeper principle the harness instantiates is **supervision by
construction**.  Standard supervised learning fits a function to
(input, output) pairs whose outputs were labeled by some external process.
When that external process is an LLM, the trained model approximates the
LLM's classification function — useful, but not independent of it.
Supervision by construction inverts the arrow: rather than labeling
inputs, we *construct* (input, output) pairs simultaneously from a
generator whose mapping is known by design.  A column produced by
`gen_email()` is labeled `EMAIL` not by an LLM's opinion but by the
constructor's contract.  This is the form Synthia (Meyer & Nagler 2021,
*Copula-based synthetic data augmentation for machine-learning
emulators*) takes when applied at scale: copula-modeled inter-column
dependence structure produces realistic synthetic relational data with
ground-truth labels by construction.

The Ægir project (Zndx 2026, `zndx.github.io/aegir`) is the foundation-
model-scale instance of the same principle: a hierarchical byte-level
sequence model (H-Net dynamic chunking + RWKV-7 time-mixing) trained
against a *deterministic four-component verifier* $R(O, I)$ with locked
aggregation weights via RLVR + GRPO.  The verifier is the procedural
component; the model never sees an LLM-labeled instance during training.
Independence-by-construction holds at the foundation-model scale.

---

## 6. Operational Discipline: No Silent Degradation

### 6.1 The cultural pathology

Architectural improvements that ship default-off and silently fall back to
the prior state on any failure are, in steady state, equivalent to not
shipping at all.  The "additive, never a hard dependency" framing creates
a cultural patch-out vector: future maintainers, seeing an optional path
that triggers rarely, can argue for its removal as maintenance overhead.
The collapsed state remains the actual production behavior; the new code
becomes ornament.

### 6.2 The discipline

Per the (newly committed) feedback memory `feedback-no-silent-dst-degradation`,
DST-restoration enhancements observe four rules:

1. **Defaults flip ON.**  The `AtelierConfig` field defaults to `True`;
   the HOCON binding in `config/base.conf` defaults `enabled = true`.
2. **Explicit disable vs. deployment-degraded are distinguished.**
   Operator-set `enabled = false` is silent (the operator knows what they
   are doing).  Anything else that prevents the new path from running —
   missing dependency, missing data, downstream service unreachable —
   produces a *loud* WARNING and a *tagged* result artifact (`cosine_path:
   "legacy_degraded:<reason>"`).
3. **Fallback paths exist for emergency rollback only.**  Their docstrings
   frame them as transitional, not as "safe defaults".  Forbidden framings:
   *optional, additive, non-fatal, graceful fallback, safe rollout, opt-in*.
   Preferred framings: *production, load-bearing, emergency rollback only,
   deployment-degraded state*.
4. **Pipeline metadata carries the degraded marker.**  Operators see at a
   glance, on per-column and per-run granularity, whether the run executed
   in the production path or in degraded mode.

### 6.3 The status taxonomy

The bridge between the pipeline and the late-interaction path
(`late_interaction_bridge.try_compute_cosine_mass`) returns a tuple
$(m, s)$ where $s \in \{$ok, explicit_disable, degraded_no_dao,
degraded_no_collection, degraded_no_qdrant_client, degraded_qdrant_connect,
degraded_load_failed, degraded_score_error, degraded_bridge_error$\}$.
Each `degraded_*` status maps to a specific operator remediation; the
pipeline emits a WARNING with the remediation hint and tags the per-column
result.  Aggregating `cosine_path` across columns in a run artifact gives
a per-run health view of the cosine evidence path.

---

## 7. Near-Term Path

### 7.1 De-circularizing CatBoost: frozen-snapshot training

CatBoost as currently constructed (`fit_to_llm` against the in-loop LLM
predictions) is structurally incapable of being an independent witness.
The near-term remediation is *workflow*, not algorithm: train CatBoost
once on a frozen offline snapshot of LLM predictions over a diverse
multi-corpus reference set, version the model artifact, and reuse it
across runs.  The CatBoost evidence then represents the LLM-family's
classification function as a stable, fast, reproducible signal — partial
independence (single model family) but freed from the in-loop feedback
that produces the false consensus.

### 7.2 SVM-on-synthetic with Synthia copulas

The SVM source is currently trained on a synthetic corpus and aligned to
the user taxonomy via LLM-mediated ICE-mapping at inference time.  The
near-term work strengthens this in two ways: (a) enhance the synthetic
generator with **copula-modeled inter-column dependence** (Synthia, Meyer
& Nagler 2021) so synthetic tables preserve realistic joint distributions
rather than column-independent marginals; (b) drive the **Agent-SDK
curation** of the synthetic corpus by *taxonomic coverage gaps* — explicit
under-represented (mnemonic, value-shape) pairs surfaced by an offline
gap-analysis script — rather than by SVM-disagreement metrics (which would
launder LLM bias one level up).  The resulting SVM is trained on
procedurally-labeled data whose joint structure approximates real-world
tables; it is independent of the LLM by construction, not by hope.

### 7.3 LLM as targeted re-evaluator

With CatBoost frozen and SVM independent, the LLM is repositioned from
*primary evidence source* to *appellate court*: invoked only on columns
where the truly independent witnesses (Aegir, SVM-on-synthetic, cosine
late-interaction) produce high DST conflict $K$ or high belief gap
$\mathrm{Pl} - \mathrm{Bel}$.  This is the active-learning idiom on the
evidence layer: query the expensive oracle at points of maximum
uncertainty.  Each LLM invocation is a *fresh, prior-informed judgment*
shown the conflicting evidence, not a pre-committed mass-function
contributor.  Cost falls to ~10–20 % of columns; the savings are
redeployed to ensemble multiple LLM calls (different prompts,
temperatures) on contested columns, producing additional independent
evidence whose internal consistency is itself a signal.

### 7.4 Ægir foundation model

Long-term, the cosine + SVM + frozen-CatBoost stack is supplemented by
**Ægir**: a hierarchical byte-level sequence model trained on public
tabular benchmarks (SOTAB v2, GitTables 1M+, WikiTables) against the
deterministic verifier $R(O, I)$.  Ægir performs three tasks the current
pipeline cannot: column-type annotation (CTA), column-property annotation
(CPA — inter-column relationships), and cross-table grouping (coherent
real-world entity recognition spanning columns and tables, e.g., a
`PaymentCard` decomposed across `card_number`, `expiry`, `cardholder`).
Ægir's outputs enter the DST stack as a fourth strongly-independent
evidence source; its CPA and cross-table outputs additionally enrich the
LLM-appellate-court prompt with structural context the LLM does not
otherwise see.

### 7.5 Channel-aware reliability shaping

The current channel-decomposed Dempster combination uses fixed positive-
channel $\alpha$ from Haenni-Hartmann reliability shaping and a fixed
negative-channel budget $\beta$.  An enhancement under consideration:
make $\beta$ a function of the column's *coverage* of the anti-examples
(if few sample values match any anti-example, $\beta$ shrinks toward 0;
if many match, $\beta$ approaches its ceiling).  The construction
preserves the channel-decomposition structure while adapting the negative
budget to evidence-availability — analogous to how Haenni-Hartmann adapts
$\alpha$ to top-1 absolute / margin signals.

---

## 8. Open Questions

1. **Cautious-rule combination of LLM-derivative cluster.**  Even after
   the near-term remediations, the LLM, frozen-CatBoost, and any LLM-as-
   appellate-court outputs share a residual non-distinctness.  Denoeux
   2008's cautious rule (idempotent on identical evidence, commonality-
   formulation $q_1 \hat{\wedge} q_2$) dissolves this at the math layer.
   The implementation cost is moderate (the existing `combine_multiple`
   infrastructure has the strategy-dispatch surface); the trade-off is
   that cautious is non-normalising, so the operator-facing belief
   intervals narrow.

2. **Anti-example over-suppression boundary.**  The negative channel
   currently allocates up to $\beta = 0.30$ of mass to complement focal
   elements.  At what corpus regime does this over-suppress legitimate
   matches?  The architecture document flags this as a calibration
   question; empirical data from the first full sweep against the new
   architecture will inform.

3. **SDG ↔ Atlas Lexicon translation.**  Ægir's outputs are in its
   learned SDG ontology; Atelier consumes Atlas Lexicon codes.  The
   translation layer is itself an evidence-injection point; whether it
   should be deterministic (table-driven, auditable) or learned (more
   flexible, reintroduces a hidden source) is unresolved.

4. **Operator-overlay reproducibility.**  Per-deployment hand-edits to
   the enriched annotation collection are valuable for ergonomics but
   break global reproducibility claims.  The proposed two-version stack
   (base augmentation version + operator overlay version) admits a
   per-deployment scoping but requires the audit machinery to be
   correspondingly two-dimensional.

5. **The cautious-rule subcategory.**  Categorically, Dempster's rule
   restricts to the subcategory of conditionally-independent sources;
   cautious combination operates on the broader category but loses
   normalisation.  Is there a natural intermediate — a *partial*
   independence framework where mass is combined conjunctively up to a
   measure of shared lineage?  This connects to recent work on
   epistemic entanglement (Cuzzolin 2014, *On the relative belief of
   singletons: a measure of total ignorance*) and remains theoretically
   open.

---

## 9. Conclusion

Atelier's DST evidence-fusion pipeline had collapsed under the weight of
in-loop training feedback: of six declared sources, only three were
structurally independent of the LLM, and Dempster's rule's independence
axiom was being violated in every fusion.  The remediation lands as a
multi-vector late-interaction cosine evidence source backed by Qdrant, a
channel-decomposed Dempster combination preserving conflict $K$ as
first-class signal, and an LLM-mediated enrichment pipeline paired with a
procedural verifier suite — supervision by construction at the
near-term scale, anticipating Ægir's foundation-model instantiation of
the same principle.

The discipline that makes this stick is not the code but the framing:
DST-restoration enhancements default ON, fallback paths are loudly tagged
as deployment-degraded states, and docstring language that signals
optionality is treated as a cultural patch-out vector to be edited out.
The cognitive system performing classification is the harness, not the
model; the legitimacy of LLM-mediated reference artifacts comes from the
procedural reproduction machinery they are paired with, not from the
model that produced them.

If the construction holds, the accuracy ceiling under the post-pivot
architecture is plausibly higher than the pre-expansion ~97 %, because
it gains capabilities (CPA, cross-table grouping via Ægir; copula-modeled
inter-column features via Synthia) the original architecture lacked.
Whether it holds is an empirical question whose first answer will come
when the post-pivot architecture is sweep-tested against the same
920-column reference the current collapsed-DST architecture was
characterized against.

---

## References

- Banach, S. (1922).  Sur les opérations dans les ensembles abstraits et leur
  application aux équations intégrales.  *Fundamenta Mathematicae* 3, 133–181.
- Cuzzolin, F. (2014).  On the relative belief of singletons: a measure of total
  ignorance.  *International Journal of Approximate Reasoning* 55(2), 512–530.
- Denoeux, T. (2008).  Conjunctive and Disjunctive Combination of Belief
  Functions Induced by Non-Distinct Bodies of Evidence.  *Artificial
  Intelligence* 172(2-3), 234–264.
- Elkan, C. (2001).  The Foundations of Cost-Sensitive Learning.  *IJCAI '01*.
- Haenni, R. & Hartmann, S. (2006).  Modeling Partially Reliable Information
  Sources: A General Approach Based on Dempster-Shafer Theory.  *Information
  Fusion* 7(4), 361–379.
- Khattab, O. & Zaharia, M. (2020).  ColBERT: Efficient and Effective Passage
  Search via Contextualized Late Interaction over BERT.  *SIGIR '20*, 39–48.
- Lanier, J. (1985).  *Conceptual frameworks for VR*.  VPL Research lab notes
  (and subsequent retrospectives in *Dawn of the New Everything*, 2017).
- Meyer, D. & Nagler, T. (2021).  Synthia: multidimensional synthetic data
  generation in Python.  *Journal of Open Source Software* 6(65), 2863.
- Saad, Y. (2003).  *Iterative Methods for Sparse Linear Systems* (2nd ed.).
  SIAM.  §4.1 fixed-point iteration framework.
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2022).
  ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
  Interaction.  *NAACL 2022*.
- Shafer, G. (1976).  *A Mathematical Theory of Evidence.*  Princeton
  University Press.  Ch. 3 §3, Ch. 4 §3, §11.3.
- Smets, P. (1990).  The Combination of Evidence in the Transferable Belief
  Model.  *IEEE TPAMI* 12(5), 447–458.
- Smets, P. (1993).  Belief Functions: The Disjunctive Rule of Combination
  and the Generalized Bayesian Theorem.  *International Journal of
  Approximate Reasoning* 9(1), 1–35.
- Smets, P. & Kennes, R. (1994).  The Transferable Belief Model.
  *Artificial Intelligence* 66(2), 191–234.
- Won, A. S., Bailenson, J., Lee, J., & Lanier, J. (2015).  Homuncular
  Flexibility in Virtual Reality.  *Journal of Computer-Mediated
  Communication* 20(3), 241–259.
- Yager, R. R. (1987).  On the Dempster-Shafer Framework and New Combination
  Rules.  *Information Sciences* 41(2), 93–137.
- Ægir project documentation: https://zndx.github.io/aegir/
- Companion architecture notes:
  [`docs/src/architecture/dst-evidence-independence.md`](../../src/architecture/dst-evidence-independence.md)
  and
  [`docs/src/architecture/late-interaction-cosine.md`](../../src/architecture/late-interaction-cosine.md).
