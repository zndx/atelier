# Genus-neighborhood NHSVM baseline

Training a factorized NHSVM in situ on the genus neighborhood runs, on CPU, against Aegir's labeled DDL pin plus the installed Atelier generators. The same run shows that the tables do not yet carry a value signal that distinguishes a genus child. The heads were written under `build/results/neighborhood-baseline/heads/` and were not promoted. Current product head `a40626cf-5720-46c6-9ce4-aeadd16787b6` (sig `4063540f9f3669eb`) is unchanged. No fusion was run.

The script is `scripts/neighborhood_nhsvm_baseline.py`. It sets `CUDA_VISIBLE_DEVICES=""` before importing torch. The encoder log records `device=cpu`. The training pid was absent from `nvidia-smi` compute apps. Commit `359070e` is the rich-table holdout. `tests/classify/test_neighborhood_baseline.py` passed (4).

## Frame and split

Order 1 lifts the 1,238 labeled columns onto 201 codes: 500 columns on 194 children of a structural genus, and 738 columns on the 7 roots (`SDG.PROCESS`, `SDG.ROLE`, `SDG.GENERIC`, `SDG.GDC`, `SDG.INDEPENDENT_CONTINUANT`, `SDG.QUALITY`, `SDG.DISPOSITION`). The 329 leaves that hang directly on a root are those 7 root labels. The ancestor frame of the codes that remain after the stub filter is 209 nodes. The head has 160,512 parameters.

The holdout is one whole DDL table per label. A label with two rich tables holds out a rich table. The alphabetical smallest-table rule had been selecting single-label stub tables, so that rule is retired. Rows whose values are mostly `Title Words NN` placeholders are then dropped: 195 support rows and 114 holdout rows. The scored corpus is 794 support rows (261 child, 533 root), 135 holdout rows (109 child, 26 root), and 776 synthetic rows. The synthetic rows are 4 tables for each of the 194 generatable child labels. The 7 roots have no generator and stay real-only. Support tables: 404. Holdout tables: 116. The child exam covers 99 labels.

The collection answer key is still absent, and this pin emitted no cross-family foreign keys, so the holdout unit is the DDL table. `build_nhsvm_head` still refuses while `corpus/cta_answer_key.csv` is missing.

## Scores

Exact top-1 is the headline. `child` is a column whose parent is a structural genus. `root` is one of the 7 collapsed roots. `ancestor` means the prediction sits on the gold path above the gold code. That placement is a diagnostic. A genus prediction for a child column is an ancestor placement.

Always predicting the majority support label, `SDG.PROCESS`, is exact on 7 of 135 holdout rows (0.052).

| Condition | Rows | Exact | Child | Root | Train exact |
|---|---:|---:|---:|---:|---:|
| Lexical, real support | 135 | 25 (0.185) | 10/109 | 15/26 | |
| Lexical, synthetic only | 109 | 74 (0.679) | 74/109 | | |
| Lexical, real plus synthetic | 135 | 56 (0.415) | 40/109 | 16/26 | |
| Lexical, synthetic values only | 109 | 0 | 0/109 | | |
| Lexical, synthetic, name and table masked | 109 | 4 (0.037) | 4/109 | | |
| Lexical, real values only | 135 | 12 (0.089) | 0/109 | 12/26 | |
| NHSVM, real support | 135 | 4 (0.030) | 2/109 | 2/26 | 0.370 |
| NHSVM, synthetic only | 109 | 14 (0.128) | 14/109 | | 1.000 |
| NHSVM, real plus synthetic | 135 | 31 (0.230) | 19/109 | 12/26 | 0.727 |
| NHSVM, real values only | 135 | 11 (0.081) | 0/109 | 11/26 | 0.408 |
| NHSVM, synthetic values only | 109 | 2 (0.018) | 2/109 | | 0.997 |

Each NHSVM condition trained for 300 epochs with inverse-frequency sample weights. Wall time was 4.4 to 8.7 seconds per fit. The first ModernBERT encode of 1,705 full-text rows took 95 seconds on CPU. The value encode took 30 seconds once the encoder was loaded.

`nhsvm_real` placement on the 135 holdout rows: exact 4, ancestor 43, same genus 34, miss 54. `nhsvm_augmented`: exact 31, ancestor 43, same genus 16, miss 45. `nhsvm_synthetic` on the 109 child rows: exact 14, same genus 21, miss 74. Real-support train exact stopped at 0.370 while the hinge loss was still falling (1.435 at epoch 0, 0.663 at epoch 299). Synthetic train exact reached 1.000. Synthetic-values train exact reached 0.997.

## What the lexical score is using

The synthetic lexical exact of 74/109 is a cross-slot name bridge.

Ninety-nine of the 109 child holdout columns are named `id`. One hundred one of the 109 rows contain a token from the synthetic target-column name inside a sibling name or the table name. All 74 exact hits are inside those 101 rows. The other 8 rows have no such token and have 0 exact hits. Zero of the 99 child codes share a sample-value string between the synthetic tables and the holdout.

Feature ablations of the same lexical model, synthetic train against child holdout:

| Text | Exact |
|---|---:|
| Full feature text | 74/109 |
| Drop `source_table` | 75/109 |
| Drop `sample_values` | 61/109 |
| `column_name` only | 6/109 |
| `sibling_context` only | 4/109 |
| Drop `column_name` | 4/109 |
| Drop `column_name` and `source_table` | 4/109 |
| `sample_values` only | 0/109 |
| `value_description` only | 0/109 |

`column_name` alone is 6/109 because the holdout column is usually `id` while the synthetic column is the concept (`agenda_subject`). Dropping `column_name` from both sides removes the concept string from the synthetic document, and the score falls to 4/109. The bridge is the synthetic column name landing in the same TF-IDF space as the holdout sibling list and table name.

Example, gold `SDG.ICE.DESIGNATIVE.AGENDA_SUBJECT`:

- Holdout table `t_agenda_subject`, column `id`, values `SUBJ-0001` …, siblings `agenda subject`.
- Synthetic column `agenda_subject`, value `budget_ratification`, siblings `item number`, `meeting date`.

The shared tokens are `agenda` and `subject`. The stored values share nothing. Real-value child exact is also 0/109 for both the lexical model and NHSVM, so the Aegir values on these held-out columns, many of them surrogate identifiers, do not separate the 99 child labels either. The 12 real-value exact hits are all roots.

NHSVM on the full synthetic text scores 14/109 with a perfect train exact. The hierarchical head memorizes the synthetic rows. The sparse token bridge that TF-IDF uses does not transfer through the ModernBERT embedding at the same rate. Value-only NHSVM memorizes (train exact 0.997) and scores 2/109.

## Design changes

1. Keep exact child top-1 as the number that decides the next corpus. Report root exact on its own line. The 26 root rows sit on 7 classes, and a micro average lets them move the headline. `nhsvm_real` at 4/135 is below the 7/135 majority floor.

2. Mask `column_name` on the exam that claims to test stored values, and count rows whose synthetic column-name tokens appear in the holdout sibling list or table name. On this split that bridge covers 101 of 109 child rows. The 8 rows outside it are too few to score. The next corpus needs child columns that do not repeat the concept in a sibling or the table name.

3. Emit the shape the holdout already has. The labeled column is a surrogate (`id`, values `SUBJ-0001`). The concept string sits on a sibling and in the table name. A generator that names its only interesting column `agenda_subject` and stores `budget_ratification` teaches the name bridge. The useful generator emits a surrogate identifier, a sibling whose name is a role in the table, and values whose distribution is shared with held-out tables of that code.

4. Keep the rich-table rule and the stub-row filter. The alphabetical smallest-table holdout preferred single-label placeholder tables. One hundred fourteen holdout rows and 195 support rows were those placeholders.

5. Leave the 7 roots real-only. They have no relational generator. Order 2 would keep 187 siblings of the genera as their own labels (`PROCESS` 160, `GDC` 15, `INDEPENDENT_CONTINUANT` 12). `ROLE`, `GENERIC`, `QUALITY`, and `DISPOSITION` stay on the root at either order. A root score is a score on that collapse.

6. Keep the 209-node ancestor frame. The fit is small enough to repeat on CPU. More epochs will not manufacture a child-value signal: the value ablation is already empty, and the real-support fit was still moving at epoch 299 with train exact 0.370.

7. Publish `corpus/cta_answer_key.csv` before this frame is wired into `build_nhsvm_head`. Until that file exists, the collection path stays fail-closed. This baseline does not stand in for an FK-closed collection holdout.

8. Add an NHSVM cell with `column_name` masked. The lexical cell already shows the bridge. The hierarchical head should be scored on the same masked text before anyone reads a full-text synthetic accuracy as domain transfer.

9. Leave `a40626cf` as the current head. Promote a neighborhood head only after child exact top-1 on a name-masked holdout is the assigned code. Ancestor placement stays out of that gate.
