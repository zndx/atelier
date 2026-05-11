<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Fresh CAI deployment — what to verify

Everything baked into the `trunk` head as of 2026-04-18.  UAT can
deploy from scratch; this note captures the "what should we see"
checklist so the phase-gate validation is fast when the run comes
back.

## 1. Pre-flight (post-deploy, pre-pipeline)

**Credentials card** — both providers should validate.  The probe
fix (`1edbb92`) uses the right model per provider:

- `anthropic` Valid against `claude-opus-4-7` (direct API)
- `bedrock` Valid against whatever ARN the operator set in
  `ATELIER_AGENT_MODEL`

**SDK Smoke Test** should return `Atelier agent SDK operational`
in 1–3 s with a cost around $0.007.

**Atelier Terminal** — typing `chk` should now produce a full
health-check sitrep.  Prior silent-run symptom was the
`.claude/settings.json` marker missing; shipped in `382172d`.

**Ground truth CSV (optional but recommended)** — point
`ATELIER_GROUND_TRUTH_URI` at a CSV with columns
`column_name,code[,annotation]` (qualified names) OR
`table_name,column_name,code[,annotation]`.  Tailing-`/annotations`
table + `ice_t1` are excluded automatically.  When the CSV resolves,
`evaluation_report.json` gets real accuracy numbers and overwatch's
"Mispredictions vs ground truth" section activates.

## 2. Kick off the pipeline (defaults already baked)

No Settings-page overlay needed.  `config/base.conf` defaults now
reproduce our local 97.8% result:

| Setting | Value | Why |
|---|---|---|
| `mc_sample_fraction` | **1.00** | full LLM coverage |
| `classify_catboost_fit_to_llm` | **true** | CatBoost trained on LLM labels, joins fusion |
| `classify_bootstrap_clarity_target` | **0.20** | aggressive revisit on low-clarity columns |
| `classify_llm_columns_per_call` | **25** | batch-truncation guard on large vocabs |
| `classify_fusion_strategy` | **dempster** | K is a bounded diagnostic; reverted from brief Yager experiment because Yager's multi-source K exceeds 1 and confuses operators |

## 3. While running — the Status tiles to watch

Two new headline tiles in the FSM progress panel, alongside the
existing K / Confidence:

- **LLM Coverage** (green ≥95% · amber 80-95% · red <80%) — should
  reach ≥ 95% on the meta-tagging corpus at the new batch size.
  If red, LLM batches are truncating; reduce
  `classify_llm_columns_per_call` further.
- **LLM Agreement** (green ≥98% · amber 90-98% · red <90%) —
  fraction of LLM-covered columns where the fused top label equals
  the LLM's own top pick.  Under fit-to-LLM this should be 100%.
  A dip is the first-class diagnostic worth investigating: it
  means CatBoost actively disagreed with the LLM after training
  on LLM labels.  (Historically never dipped in local runs.)

**Mean K** stays visible as a diagnostic, but **stop treating it
as a correctness signal**.  With CatBoost now actually in the
fusion (prior runs silently wiped it — fixed in `7313d13`), the
LLM + CatBoost pair puts mass on the same singleton → Dempster's
K drops materially vs the `0a32f0bf` baseline of 0.756.

## 4. After convergence — artifacts to expect

`build/results/{run_id}/` should contain:

- `classifications.json` — per-column predictions, including
  `predicted_annotation` (the mnemonic), `llm_code` (the LLM's own
  top pick), `llm_confidence`, and `is_correct` as `null` when no
  ground truth (not `false`).  Evidence sources should show
  `cosine + svm + catboost + llm` as 100% presence, `name_match`
  on ~50% of columns, `pattern` absent (ICE codes not in vocab).
- `settings_snapshot.json` — the resolved config at run start.
- `focus_settings.json` — adaptive focus list for the Settings page.
- `shap_summary.json` — synchronous write (sample_fraction=1.0 puts
  MC in passthrough, so SHAP is no longer a daemon thread).
  Missing on CAI CPU hosts means something else went wrong.
- `sage_importance.json` — GPU-only, absent on CPU hosts (expected).
- `overwatch.md` — written by the prompt-refreshed agent.  Should
  lead with LLM Coverage + LLM Agreement, NOT K.  Shouldn't
  recommend `max_iterations` bumps or pattern tuning anymore.
- `evaluation_report.json` — if ground-truth CSV is plumbed,
  `columns_with_gt > 0` and `exact_accuracy` populates.  Without
  GT, these are 0 (not an error, just no reference).

## 5. Measuring against the Gopala xlsx baseline

For the CAI deployment, the Gopala xlsx IS the ground truth.
Simplest flow:

```bash
# 1. One-time export: xlsx → CSV with (column_name, code, annotation)
uv run python - <<'PY'
import csv, openpyxl
wb = openpyxl.load_workbook("Atelier_Results_Default_DB_4-16.xlsx", data_only=True)
rows = []
for sn in ('business_data','customer_data','customer_pii_pci_data',
          'Identity_data','metadata','transaction_data',
          'personal_data','system_data'):
    if sn not in wb.sheetnames: continue
    for r in list(wb[sn].iter_rows(values_only=True))[2:]:
        if not r or not r[0]: continue
        col = str(r[0])
        # Gopala's annotation mnemonic is the short uppercase token
        tag = next(
            (v for v in r if isinstance(v, str)
             and 2 <= len(v) <= 15
             and v.isupper()
             and v.replace('_','').replace('-','').isalnum()),
            None,
        )
        if tag:
            rows.append({"column_name": col, "annotation": tag})
with open("build/data/ground_truth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["column_name","annotation"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows")
PY

# 2. Set env + re-run
export ATELIER_GROUND_TRUTH_URI=build/data/ground_truth.csv
devenv up   # or equivalent on CAI
```

The ground-truth loader indexes the CSV under multiple key forms
(qualified, bare, stripped) so both the Hive-backed and
meta-tagging-source loaders resolve correctly without extra work.

## 6. Expected accuracy delta vs run `0a32f0bf`

Carrying forward all fixes (columns_per_call=25, fit_to_llm=True
actually applied, CatBoost preserved through SVM retrain,
annotations/ice_t1 excluded):

| Metric | `0a32f0bf` | Projected next run |
|---|---|---|
| CatBoost evidence presence | 0% | **100%** |
| LLM coverage | 79.1% | **≥ 95%** |
| LLM Agreement | n/a | **100%** |
| Overall match vs Gopala baseline | 63.5% | **≥ 90%** |
| identity_data match | 20% | **≥ 85%** |
| personal_data match | 58% | **≥ 92%** |
| Mean K (Dempster) | 0.756 | **< 0.40** |

The 63.5% → ~90% jump is the CatBoost-fills-LLM-gap story
finally observable.

## 7. If something isn't right

| Symptom | Likely cause | Action |
|---|---|---|
| LLM Coverage < 80% | Batch truncation | `classify_llm_columns_per_call = 15` via Settings; re-run |
| LLM Agreement < 100% | CatBoost-vs-LLM divergence | Inspect the mismatched cols; log the embedding_text and LLM's top-3 proba |
| catboost evidence = 0 | Fit-to-LLM not firing | Check settings snapshot for `classify_catboost_fit_to_llm=true`; confirm LLM sweep produced ≥ `fit_to_llm_min_labels=30` labels |
| Mean K > 1 | Yager accidentally active | Revert to `fusion_strategy=dempster` in Settings |
| `overwatch.md` missing | Anthropic key not present or Bedrock-only | overwatch requires direct Anthropic key (`cfg.has_anthropic`) |
| Shared terminal: "no assistant output" | Slash command resolved client-side | Type `clear` then re-run the command |
