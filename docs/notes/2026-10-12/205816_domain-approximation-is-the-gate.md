# Domain approximation is the gate

The Qwen3.8 relational rows do not approximate the target corpus. A model
that fits them has not learned the classification. The historical bar is
a simple lexical model trained on generated rows and scored on real
columns.

## What the holdout score showed

Head `a40626cf` reaches argmax 1.000 on the 4,484 synthetic rows and
0.120 on the 491 holdout entities. The synthetic text for the
abuse-indicator code is `abuse indicator | bruise_pattern_flag`. The
holdout column is `indicator id | Brute force login, DDoS flood, …`.
Those strings do not share the tokens a lexical model would use. The
support row for the same code is the surrogate `id | INDI-0001`.

## Historical demonstration

On the 1,149-row reference, TF-IDF plus the Kronecker NHSVM reached
0.9893 fit-on-train top-1. The same embeddings under the Kronecker form
fell to 0.0426, which is why the factorized head exists. The
generalization regime that matters here is synthetic-only training
scored on real reference columns. That corpus was drawn from enrichment
prototypes and name hints: stored-string exemplars and column-name
tokens an author had taken from the domain. Template generators sampled
those prototypes. The simple model was the proof that the generated
rows occupied the same feature space as the target columns.

Qwen3.8 received the SKOS label, abbreviation, parent, and description.
The pack withheld the target columns. The lambdas it wrote are a
private lexicon. Memorizing that lexicon is not a demonstration that
we understand the domain.

## Acceptance test before another head fit

Generate relational rows whose column names, values, and siblings carry
the same salient features as the assigned target columns. Train a
simple lexical model on those rows alone. Score it on the known holdout
entities. Another ModernBERT fit waits on that result.
