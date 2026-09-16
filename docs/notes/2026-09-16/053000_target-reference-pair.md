# Target / reference sample pairs (NHSVM holdout)

The macbook target sample cannot DST-converge (24 SKOS terms, 77 vocab
gaps). Pair it with a **disjoint reference** whose BFO/CCO genus IRIs
cover the target; train a fresh NHSVM on the reference; classify the
target. No collection overlap.

Atelier can select a partial reference from the same sdg-corpora pin
(`just sdg-sample-reference <target-id>`). Remaining SKOS-term gaps
(`pair.skos_missing`) are Ægir's: emit collections from finepdfs +
SchemaPile that realize those IRIs without overlapping the target.

```
just classify-flow --host \
  --source-id=sdg-corpora/b24ef9f60660_macbook \
  --reference-id=sdg-corpora/b24ef9f60660_reference \
  --target=phase:nhsvm
```

DST sweep on the target is gated on NHSVM being able to label the
holdout (`perfect_possible` when `skos_missing` is empty).
