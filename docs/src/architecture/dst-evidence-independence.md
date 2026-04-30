# DST Evidence Independence

This note documents how Atelier's classification pipeline handles
**non-distinct evidence sources** under Dempster-Shafer fusion, and
why the discount calibration and revisit gate are structured the way
they are. It is intended to be cited by code reviewers and academic
readers.

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
6. `svm` — frontier SVM classifier.

The first three are genuinely independent of the LLM: their evidence
arises from the column's name, value patterns, and semantic embedding
comparison against the vocabulary. The last three are *not*
independent in the current architecture:

- **`catboost`** is trained in `fit_to_llm` mode (default true) on
  `(embedding_text, llm_code)` pairs from the current run's LLM
  sweep. See `ml_train.fit_catboost_to_llm_labels` and
  `pipeline._install_fit_to_llm_catboost`. The fitted model is, by
  construction, an explainability surface over the LLM's labels —
  not a competing classifier.
- **`svm` (frontier)** is hot-swapped during the bootstrap loop on
  labels filtered to `label_source in ("llm", "llm_revisit")` (see
  `ml_train` ~line 118). Its training set is exclusively LLM output.

Treating LLM, CatBoost(LLM), and SVM(LLM) as three independent
sources and combining them via Dempster's rule double- and
triple-counts the LLM atom. The pre-2026-04-30 discount schedule
made this worse: `llm=0.10`, `catboost=0.10`, `svm=0.20`, vs
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

| Source       | Discount | Rationale                                  |
|--------------|----------|--------------------------------------------|
| `cosine`     | 0.30     | independent of LLM; semantic prior         |
| `pattern`    | 0.25     | independent; deterministic regex evidence  |
| `name_match` | 0.30–0.70 | independent; lexical match against vocab  |
| `llm`        | 0.10     | original; first-pass label                 |
| `catboost`   | **0.55** | **derivative** (`fit_to_llm`)              |
| `svm`        | **0.55** | **derivative** (frontier-LLM-trained)      |
| `catboost_max` | 0.75   | variance ceiling; maintains headroom       |

Operators can dial these via the Settings page when retraining
CatBoost/SVM on labels independent of the current LLM sweep (e.g.
synth-only training); the metadata in `config_overlay.SETTINGS_METADATA`
exposes the full range.

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

This treatment is conservative: it preserves Dempster's rule
end-to-end and handles non-distinctness through reliability
discounting. Two future refinements were scoped out of the present
change:

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
- **Dynamic cosine reliability (Haenni & Hartmann 2006, *Modeling
  Partially Reliable Information Sources*).** Replace the static
  `discount=0.30` with a reliability factor `α(s₁, s₂)` shaped by
  cosine top-1 absolute similarity and top-1/top-2 margin, so sharp
  cosine signal carries weight and diffuse cosine remains
  appropriately ignorant. Naturally complements the tiered fusion.

The `combine_multiple` infrastructure already supports adding a
`strategy="cautious"` branch alongside the existing `dempster` /
`yager` options, so both refinements are surgical when they land.

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
