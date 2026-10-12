# Holdout NHSVM top-1 is not a DST singleton

The CPU fit of the `projected-holdout` head finished and is the current
registry row. Scoring it on the 491 known holdout entities fails the
pre-fusion check, so the classification pipeline was not started.

## Head

- Id `a40626cf-5720-46c6-9ce4-aeadd16787b6`, signature `4063540f9f3669eb`,
  status `current`, taxonomy `projected-holdout`, encoder
  `answerdotai/ModernBERT-base` on CPU.
- Artifact `build/cache/nhsvm/4063540f9f3669eb`.
- Fit log reports train argmax accuracy 0.990 over 5,231 rows and 1,036
  taxonomy nodes. Re-scoring the saved training texts reproduces that
  argmax: 0.990 overall, 1.000 on 4,484 synthetic rows, 0.933 on 747
  real support rows.

## Holdout decision

`conflicting_known_entities` against the head's own argmax, using the
same text the pipeline builds (`extract_features` then the adapter's
`_build_text`):

| Slice | Argmax accuracy | Median top-1 probability | Median top1−top2 logit gap |
| --- | --- | --- | --- |
| Training texts | 0.990 | 0.0080 | 1.34 |
| Real support rows | 0.933 | 0.0064 | 1.15 |
| Synthetic rows | 1.000 | 0.0081 | 1.37 |
| 491 holdout entities | 0.120 (59 correct) | 0.0029 | 0.09 |

Every holdout distribution still has all 1,036 codes above the
`svm_to_mass` cutoff of `1e-15`. A correct argmax is not a peaked
singleton. Putting that softmax into Dempster-Shafer would commit
contradictory mass on the other codes.

The training type token `object` is not the gap. Stripping it from the
training texts leaves argmax accuracy at 0.988. Inserting it into the
holdout texts leaves holdout accuracy at 0.120.

## Why the strings separate and the entities do not

Support is the lexicographic prefix of each code's columns, so 307 of
747 real training rows are a column named `id`. Their values are
surrogate keys. The synthetic rows use a different lexicon than the
holdout's stored strings.

Abuse-indicator example:

- Real support: `id | object | INDI-0001, INDI-0002, … | siblings: physical form, uncertainty, value`
- Synthetic: `abuse indicator | object | bruise_pattern_flag | siblings: manifests as ref`
- Holdout `abuse_flags.indicator_id`: `indicator id | Brute force login, DDoS flood, Insider data theft, Malware beaconing`

The head's argmax tracks the training strings. It does not assign the
holdout entity to `SDG.ICE.DESCRIPTIVE.ABUSE_INDICATOR`. The same
pattern shows up for safeguard measures (`id | MEAS-0001` and
`directive ref | SM-882` versus `policy id | WHOIS Redaction, …`) and
for roles whose support row is `id | ROLE-0001`.

Holdout column names are the same shape: 213 of 491 are `id`. The
adapter does not set `_hierarchical`, so a later pipeline run would
also post-hoc reweight this head. That reweight was not part of this
score. The score used the head's path-score argmax directly.

Fusion waits until a holdout argmax matches the assigned code and the
mass on that code is a singleton plus ignorance.
