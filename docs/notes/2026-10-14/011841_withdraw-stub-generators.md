# Stub relational generators withdrawn

2026-10-14. The 523 `lambda rng` tables in
`src/atelier/classify/insitu_extensions.py` (landed as `2ad6b46`) are
removed. Each table had 2–5 columns (mean 2.59), one column carrying a
SKOS IRI, and values drawn from `rng.choice` or `rng.randint`. On
sdg-corpora `510db0b8` those lambdas covered 785 of 26,446 published
columns. None of the 26 genus labels (14,202 columns) and none of the
four parent roots (10,805 columns) had a lambda. 449 of the 523 lambdas
were for a code that is not a collection reference code. The tables do
not follow a published collection's structure or its GitTables value
distribution, so training on them attached that noise to the tag.

`GENERATORS` is now an empty dict. `install_generators` and every entry
point of `qwen_generator_harness` raise. The skill
`extend-procedural-generator` and the agent `generator-extender` say
the same. `prepare_insitu` keeps the real support columns.
`extra_generators` and `seed` are accepted and ignored.
`distribution_deficit` is unchanged and is recorded on
`plan.variants`. It is not filled. `plan.exemplars` is empty.
`plan.holdout_rows` stays empty. The amplification test still expects
`PDF_EXPORT`, `PARENTING`, and `YOUNG_RESEARCHER` at 5 when
`amplified_variants=5`.

`prepare_neighborhood_baseline` returns `synthetic_rows=[]` and lists
every label on `unfilled`. `scripts/neighborhood_nhsvm_baseline.py`
calls it with `variants_per_code=0` and scores the real rows. The
synthetic arms run only when `synthetic_rows` is non-empty. That script
was not executed. Head `a40626cf` stays. `build_nhsvm_head` still reads
the absent `corpus/cta_answer_key.csv` and was not pointed at
`align_genus_split`.

MaxSim `InSituPrototypeGenerator` still pads an empty exemplar list by
repeating the term label until length 3. That is the remaining
prototype behavior. It is not an Aegir task.

The historical `generators_v1` scripts and `qwen_synth_harness` (the
CSV trial) are a different stack. They were left in place.

Augmented collections are specified for Aegir in
`/home/rch/local/src/zndx/aegir/docs/current/src/operations/augmented-sdg-corpora.md`
(session copy `docs/scratch/2026-10-14/011841_augmented-sdg-corpora.md`).
59 order-1 neighborhood labels and 6 genus labels exist in one
collection. Twelve multi-collection children have no test collection
under the current genus split. That placement is Atelier's, once a
second collection exists only for the one-collection labels.

`uv run pytest tests/classify/test_insitu_corpus.py tests/classify/test_neighborhood_baseline.py tests/classify/test_qwen_generator_harness.py`
passed (25). The published-footprint test now expects `SDG.GENERIC`
children to stay themselves and the bucket to be absent. No head was
fit. No service was restarted.
