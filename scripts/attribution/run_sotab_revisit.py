#!/usr/bin/env python
"""SOTAB revisit pass — iterative convergence experiment.

Loads a completed pilot run's outputs (columns.parquet, feature
attribution JSONL) and re-runs the LLM on uncertainty-flagged columns
with *enriched* prompts that include:

  * the LLM's pass-1 label,
  * CatBoost's top-3 competing candidates with probabilities,
  * per-column SHAP top features with values,
  * k-nearest-neighbor columns from the embedding space with their
    pass-1 labels (majority vote informs the enrichment).

The point: is LLM accuracy one-shot, or does structured evidence
fusion allow the LLM to converge to the published benchmark
iteratively?  This script answers by measuring improvement /
regression / unchanged on the revisit subset and on the corpus
overall.

Usage::

    uv run python scripts/attribution/run_sotab_revisit.py <pilot_run_id>

Outputs in the same run directory:

    revisit_results.parquet     per-column: pass-1, pass-2, GT, shifted
    revisit_metadata.json       uncertainty-threshold, subset size,
                                improvement/regression counts, deltas
    feature_attributions_v2.jsonl  refreshed attribution records with
                                   pass-2 labels stitched in.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path


log = logging.getLogger("sotab_revisit")

# Uncertainty threshold — cols where CatBoost's top-1 probability
# falls below this are flagged for LLM revisit with enriched context.
# NOTE: CatBoost fit-to-LLM tends to be uniformly confident (it learned
# to reproduce the LLM's decisions), so this threshold is a weak signal
# on its own.  Real independence comes from training CatBoost on synth
# data generated independently of the LLM — the Path-2 move.
CATBOOST_UNCERTAINTY_THRESHOLD = 0.7

# If set, override uncertainty filter and revisit every column.  Useful
# for measuring the envelope of iterative shift on the full corpus.
REVISIT_ALL = True

# Margin-based uncertainty: top-1 minus top-2 probability.  Much more
# informative than absolute top-1 on overfit classifiers.
CATBOOST_MARGIN_THRESHOLD = 0.2


def _rebuild_state(run_dir: Path):
    """Re-load the pilot artifacts and reconstruct enough state for revisit.

    We re-run LLM+CatBoost pipeline steps in-process so we have live
    ``ColumnFeatures``, category_set, and a trained CatBoost ready for
    per-column SHAP + top-k probability extraction.  The alternative
    (serializing these to disk during pass-1) is cleaner but heavier;
    for a pilot-scale experiment the ~30s re-build is fine.
    """
    import pandas as pd
    meta = json.loads((run_dir / "metadata.json").read_text())
    log.info("rebuilding state from %s (n=%d cols, %d classes)",
             run_dir.name, meta["total_columns"], meta["n_classes_sampled"])

    # Re-run the stratified sample + loader with identical seed/args.
    import runpy
    # Import the pilot module as a library
    pilot_mod = runpy.run_path(
        str(Path(__file__).parent / "run_sotab_pilot.py"),
        run_name="_pilot_lib",
    )

    from atelier.classify.features import FEATURE_NAMES  # noqa: F401

    gt_rows = pilot_mod["_load_gt"](pilot_mod["SOTAB_ZIP"])
    sample = pilot_mod["_stratified_sample"](
        gt_rows,
        meta["n_classes_sampled"],
        meta["per_class_target"],
        meta["seed"],
    )
    loaded = pilot_mod["_load_table_columns"](pilot_mod["SOTAB_ZIP"], sample)
    samples = pilot_mod["_build_column_samples"](sample, loaded)
    all_labels = sorted({r[2] for r in gt_rows})
    category_set = pilot_mod["_build_category_set"](all_labels)
    feats = pilot_mod["_extract_features"](samples)

    # Rehydrate pass-1 LLM labels + any saved CatBoost predictions.
    attrib_records = []
    with open(run_dir / "feature_attributions.jsonl") as f:
        for line in f:
            attrib_records.append(json.loads(line))
    by_key = {
        (r["table_id"], r["column_id"]): r for r in attrib_records
    }

    # Retain ALL samples — including rows where the LLM abstained on
    # pass 1.  Those are the cases that benefit most from a revisit
    # with CatBoost's extrapolated prediction as fresh signal.
    pass1_labels = []
    cb_top3_saved: list[list[tuple[str, float]]] = []
    pass1_present_mask: list[bool] = []
    for s in samples:
        rec = by_key.get((s.table_name, s.name))
        if rec is None:
            pass1_labels.append("")
            cb_top3_saved.append([])
            pass1_present_mask.append(False)
            continue
        pass1_labels.append(rec.get("llm_label") or "")
        cb_top3_saved.append([
            (lbl, float(p)) for lbl, p in (rec.get("catboost_top3") or [])
        ])
        pass1_present_mask.append(bool(rec.get("llm_label")))

    n_all = len(samples)
    n_labeled = sum(pass1_present_mask)
    log.info(
        "rehydrated %d samples  (pass-1 labeled: %d  · abstained: %d)",
        n_all, n_labeled, n_all - n_labeled,
    )

    # Retrain CatBoost on labeled subset if we don't have saved top-3
    # from a newer pilot run.  Older pilot outputs won't have
    # ``catboost_top3`` — detect that and retrain.
    needs_retrain = any(not t for t in cb_top3_saved)
    if needs_retrain:
        log.info("retraining CatBoost (saved top-3 absent or incomplete)")
        clf, X_train, labels = pilot_mod["_train_catboost"](
            [f for f, y in zip(feats, pass1_labels) if y],
            [y for y in pass1_labels if y],
        )
        from atelier.classify.embedding import embed_texts
        import numpy as np
        X_all = np.asarray(embed_texts([f.to_embedding_text() for f in feats]))
        proba_all = clf.predict_proba(X_all)
        cb_top3 = [
            sorted(p.items(), key=lambda kv: -kv[1])[:3] for p in proba_all
        ]
    else:
        clf = None
        X_all = None
        cb_top3 = cb_top3_saved

    return {
        "samples": samples,
        "feats": feats,
        "pass1_labels": pass1_labels,
        "pass1_present_mask": pass1_present_mask,
        "catboost_top3": cb_top3,
        "category_set": category_set,
        "clf": clf,
        "X": X_all,
        "meta": meta,
        "pilot_mod": pilot_mod,
    }


def _catboost_topk(clf, X, k: int = 3) -> list[list[tuple[str, float]]]:
    """Per-row top-k (label, probability) from the fitted CatBoost."""
    proba = clf.predict_proba(X)  # list[dict[code, prob]]
    out: list[list[tuple[str, float]]] = []
    for p in proba:
        ordered = sorted(p.items(), key=lambda kv: -kv[1])[:k]
        out.append([(code, round(prob, 4)) for code, prob in ordered])
    return out


def _run_embedding_shap_safe(feats, category_set):
    """Run the per-feature permutation SHAP (12 conceptual features)."""
    from atelier.classify.shap_explanations import run_embedding_shap
    return run_embedding_shap(feats, category_set, n_permutations=32)


def _nearest_neighbors(X, k: int = 3):
    """Row-wise top-k nearest neighbors (excluding self) by cosine sim."""
    import numpy as np
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normed = X / np.where(norms > 0, norms, 1.0)
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argsort(-sim, axis=1)[:, :k]
    return topk


def _build_revisit_context(
    samples, pass1_labels, catboost_top3,
    shap_records, neighbor_labels_per_row,
) -> dict[str, dict]:
    """Pack CatBoost top-3, SHAP top-3, neighbor labels into the
    ``revisit_context`` dict that ``build_batch_user_prompt`` already
    consumes natively (keyed by sample.name).

    The existing prompt builder surfaces:
      * ``ml_prediction`` + belief/plausibility/conflict
      * ``confusable`` (free-text) — we stuff SHAP + alternatives here
      * ``previous`` with code + confidence — we set the pass-1 label

    This keeps us on the well-tested ``classify_batch`` path rather
    than introducing a new raw-chat surface.
    """
    ctx: dict[str, dict] = {}
    for i, sample in enumerate(samples):
        cb = catboost_top3[i]
        top1_label, top1_p = cb[0]
        top2_p = cb[1][1] if len(cb) > 1 else 0.0
        # Rough DST-style interpretation: belief=top1, plausibility=top1+top2
        belief = top1_p
        plausibility = min(1.0, top1_p + top2_p)
        conflict = 1.0 - belief
        # SHAP top 3 from the permutation-over-12-features model
        sh = shap_records[i] if i < len(shap_records) else {}
        shap_str = ", ".join(
            f"{sh.get(f'shap_top{k}_name', '')}({sh.get(f'shap_top{k}_value', 0.0):+.3f})"
            for k in (1, 2, 3)
            if sh.get(f"shap_top{k}_name")
        )
        alt_str = ", ".join(f"{lbl}({p:.2f})" for lbl, p in cb[1:])
        neighbor_str = ", ".join(dict.fromkeys(neighbor_labels_per_row[i]))
        confusable_pieces = []
        if alt_str:
            confusable_pieces.append(f"CatBoost alternatives: {alt_str}")
        if shap_str:
            confusable_pieces.append(f"top SHAP features: {shap_str}")
        if neighbor_str:
            confusable_pieces.append(f"nearest-neighbor labels: {neighbor_str}")
        ctx[sample.name] = {
            "ml_prediction": top1_label,
            "belief": round(belief, 3),
            "plausibility": round(plausibility, 3),
            "conflict": round(conflict, 3),
            "confusable": "; ".join(confusable_pieces),
            "previous": {
                "code": pass1_labels[i],
                "confidence": 0.0,
            },
        }
    return ctx


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if len(sys.argv) < 2:
        print("usage: run_sotab_revisit.py <pilot_run_id>", file=sys.stderr)
        return 1

    run_dir = Path("build/sotab_pilot") / sys.argv[1]
    if not run_dir.is_dir():
        log.error("run dir not found: %s", run_dir); return 1

    state = _rebuild_state(run_dir)
    samples = state["samples"]
    feats = state["feats"]
    pass1_labels = state["pass1_labels"]
    pass1_present_mask = state["pass1_present_mask"]
    category_set = state["category_set"]
    catboost_top3 = state["catboost_top3"]
    X = state["X"]

    # If X wasn't materialized during rehydrate, embed now for neighbors.
    if X is None:
        from atelier.classify.embedding import embed_texts
        import numpy as np
        X = np.asarray(embed_texts([f.to_embedding_text() for f in feats]))

    labeled_probs = [t[0][1] for t in catboost_top3 if t]
    if labeled_probs:
        log.info(
            "CatBoost top-1 prob mean=%.3f  median=%.3f  min=%.3f",
            sum(labeled_probs) / len(labeled_probs),
            sorted(labeled_probs)[len(labeled_probs) // 2],
            min(labeled_probs),
        )

    log.info("computing per-column embedding SHAP (12 conceptual features)")
    shap = _run_embedding_shap_safe(feats, category_set)
    shap_records = shap.to_records(k=3)

    neighbors_idx = _nearest_neighbors(X, k=3)

    # ── Revisit subset ────────────────────────────────────────
    # Three-tier uncertainty classifier:
    #   tier 1  LLM pass-1 returned no label at all (empty string)
    #   tier 2  CatBoost top1-top2 margin < 0.2 (genuinely uncertain)
    #   tier 3  CatBoost top-1 absolute prob < 0.7 (rarely triggers on
    #           fit-to-LLM; kept for parity with original spec)
    # REVISIT_ALL=True overrides all three.
    reasons = []
    for i, top3 in enumerate(catboost_top3):
        top1_p = top3[0][1]
        top2_p = top3[1][1] if len(top3) > 1 else 0.0
        margin = top1_p - top2_p
        if not pass1_labels[i]:
            reasons.append("pass1_unparsed")
        elif top3[0][0] != pass1_labels[i]:
            reasons.append("catboost_disagrees")
        elif margin < CATBOOST_MARGIN_THRESHOLD:
            reasons.append("catboost_low_margin")
        elif top1_p < CATBOOST_UNCERTAINTY_THRESHOLD:
            reasons.append("catboost_low_top1")
        else:
            reasons.append("")
    if REVISIT_ALL:
        revisit_idx = list(range(len(samples)))
        log.info("revisit subset: ALL %d columns (REVISIT_ALL)", len(samples))
    else:
        revisit_idx = [i for i, r in enumerate(reasons) if r]
        log.info("revisit subset: %d of %d columns  (reasons: %s)",
                 len(revisit_idx), len(samples),
                 {r: reasons.count(r) for r in set(reasons) if r})

    # ── Prescriptive system prompt for the revisit pass ──────
    # GLM-4.7 migration doc recommends "explicit directives (MUST,
    # REQUIRED, STRICTLY)" and "early prompt positioning".  For a
    # revisit that tests whether the LLM will lean into new signals,
    # the prompt needs to be unambiguous about three things:
    #   1. New evidence is provided and MUST be consulted.
    #   2. Updating the pass-1 answer is expected when new evidence
    #      contradicts it.
    #   3. Explicit "UNCERTAIN" is a valid response when signal is
    #      genuinely weak — we want honest abstention, not silent
    #      blank output that masks uncertainty.
    def _revisit_system_prompt(category_set):
        from atelier.classify.llm_backend import build_category_table
        table = build_category_table(category_set)
        example_code = category_set.categories[0].code
        example_alt = category_set.categories[1].code if len(category_set.categories) > 1 else example_code
        # Prescriptive tone per the GLM-4.7 migration guide: rules
        # front-loaded, imperative verbs (MUST, REQUIRED, STRICTLY),
        # explicit response-format example matching the pipeline's
        # ``_parse_classifications`` schema.  This is a second-pass
        # prompt — it MUST direct the model to consult the new evidence
        # and admit honest uncertainty rather than silently abstain.
        return (
            "You are performing a SECOND-PASS data-governance "
            "classification.  Your previous (pass-1) label, an "
            "independent ML classifier's top-3 candidates, "
            "feature-attribution signals, and similar-column labels "
            "are ALL provided per column.\n"
            "\n"
            "STRICT RULES:\n"
            "1. You MUST consult the new evidence.  Sticking to the "
            "   pass-1 label without engaging with the ML prediction "
            "   and SHAP features is NOT acceptable.\n"
            "2. If the new evidence strongly contradicts pass-1, "
            "   UPDATE your answer.  Contradicting evidence is the "
            "   point of this pass.\n"
            "3. If the evidence is genuinely ambiguous, respond with "
            "   the literal token ``UNCERTAIN`` as ``category_code``. "
            "   Silent abstention (null or empty) is NOT acceptable.\n"
            "4. Respond with ONLY a JSON array (no markdown fencing), "
            "   one object per column, using the Response Format below.\n"
            "\n"
            "## Categories\n"
            "\n"
            f"{table}\n"
            "\n"
            "## Response Format\n"
            "\n"
            f'[{{"column_name": "col_7", '
            f'"category_code": "{example_code}", "confidence": 0.82, '
            f'"evidence": "updated from pass-1 based on CatBoost top-1 and '
            f'phone-pattern SHAP feature", "alternatives": '
            f'[{{"code": "{example_alt}", "confidence": 0.11}}]}}]'
        )

    # ── Build revisit context + batched re-classify ───────────
    from atelier.classify.llm_backend import (
        create_backend_from_cfg, build_system_prompt, build_category_table,
    )
    from atelier.config import load_config

    cfg = load_config()
    backend = create_backend_from_cfg(cfg)
    system_prompt = _revisit_system_prompt(category_set)

    # Pre-compute neighbor labels per row (across the full sample set).
    neighbor_labels_per_row = [
        [pass1_labels[int(j)] for j in neighbors_idx[i]]
        for i in range(len(samples))
    ]
    revisit_ctx = _build_revisit_context(
        samples, pass1_labels, catboost_top3, shap_records,
        neighbor_labels_per_row,
    )

    # Revisit the flagged subset via classify_batch with revisit_context.
    # Batch 25 at a time like the main pilot.
    BATCH = 25
    pass2_labels: dict[int, str] = {}
    reasoning_traces_p2: list[dict] = []
    t0 = time.time()
    for b in range(0, len(revisit_idx), BATCH):
        batch_idxs = revisit_idx[b : b + BATCH]
        chunk = [samples[i] for i in batch_idxs]
        # revisit_context is keyed by sample.name; all relevant keys
        # are already in revisit_ctx.
        log.info("revisit batch %d/%d (%d cols)",
                 b // BATCH + 1,
                 (len(revisit_idx) + BATCH - 1) // BATCH,
                 len(chunk))
        try:
            resp = backend.classify_batch(
                chunk, system_prompt, revisit_context=revisit_ctx,
            )
        except Exception as exc:
            log.warning("revisit batch failed: %s", exc)
            continue
        for j, c in enumerate(resp.classifications):
            global_idx = batch_idxs[j]
            pass2_labels[global_idx] = c.category_code or ""
        reasoning_traces_p2.append({
            "batch_id": b // BATCH + 1,
            "column_ids": [f"{s.table_name}:{s.name}" for s in chunk],
            "predicted_codes": [c.category_code or "" for c in resp.classifications],
            "reasoning_text": resp.reasoning_text,
            "reasoning_tokens": resp.reasoning_tokens,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "pass": "pass2_revisit",
        })

    log.info("revisit LLM calls done in %.1fs", time.time() - t0)

    # Append pass-2 reasoning traces to the run's reasoning artifact
    # (append mode — preserves pass-1 entries from the pilot).
    traces_path = run_dir / "reasoning_traces.jsonl"
    with open(traces_path, "a") as f:
        for tr in reasoning_traces_p2:
            f.write(json.dumps(tr) + "\n")
    log.info(
        "appended %d pass-2 reasoning traces (%d chars total) to %s",
        len(reasoning_traces_p2),
        sum(len(t["reasoning_text"]) for t in reasoning_traces_p2),
        traces_path,
    )

    # ── Tabulate convergence ──────────────────────────────────
    # ``UNCERTAIN`` is an explicit pass-2 response (per the revisit
    # prompt) meaning "signal genuinely too weak to commit".  We
    # surface it as its own bucket — honest abstention is a valid and
    # useful outcome, distinct from silent failure.
    UNCERTAIN = "UNCERTAIN"
    improvements = 0
    regressions = 0
    changed_still_wrong = 0
    same_decision = 0
    unparsed_p2 = 0
    p2_uncertain_from_labeled = 0
    abstention_resolved_correct = 0
    abstention_resolved_wrong = 0
    abstention_still_unresolved = 0
    abstention_resolved_uncertain = 0
    published_gt = [s.reference_code for s in samples]
    for i in revisit_idx:
        p1 = pass1_labels[i]
        p2 = pass2_labels.get(i, "")
        gt = published_gt[i]
        if not p1:
            # LLM abstained on pass 1 — a different bucket
            if p2 == UNCERTAIN:
                abstention_resolved_uncertain += 1
            elif not p2:
                abstention_still_unresolved += 1
            elif p2 == gt:
                abstention_resolved_correct += 1
            else:
                abstention_resolved_wrong += 1
            continue
        if not p2:
            unparsed_p2 += 1
            continue
        if p2 == UNCERTAIN:
            p2_uncertain_from_labeled += 1
            continue
        if p2 == p1:
            same_decision += 1
            continue
        # decision changed — did it go toward GT or away?
        if p1 != gt and p2 == gt:
            improvements += 1
        elif p1 == gt and p2 != gt:
            regressions += 1
        else:
            changed_still_wrong += 1  # changed but neither pass hit GT

    # Overall fidelity change: merge pass-2 decisions into the label set.
    final_labels = list(pass1_labels)
    for i, p2 in pass2_labels.items():
        if p2:
            final_labels[i] = p2
    pass1_fidelity = sum(
        1 for p, g in zip(pass1_labels, published_gt) if p == g
    ) / len(samples)
    final_fidelity = sum(
        1 for p, g in zip(final_labels, published_gt) if p == g
    ) / len(samples)

    summary = {
        "pilot_run_id": sys.argv[1],
        "total_columns": len(samples),
        "pass1_labeled": sum(pass1_present_mask),
        "pass1_abstained": len(samples) - sum(pass1_present_mask),
        "catboost_uncertainty_threshold": CATBOOST_UNCERTAINTY_THRESHOLD,
        "revisit_subset_size": len(revisit_idx),
        "revisit_outcomes": {
            # cols with a pass-1 label
            "labeled_rows_improvements_p1_wrong_p2_right": improvements,
            "labeled_rows_regressions_p1_right_p2_wrong": regressions,
            "labeled_rows_changed_still_wrong": changed_still_wrong,
            "labeled_rows_unchanged": same_decision,
            "labeled_rows_unparsed_p2": unparsed_p2,
            "labeled_rows_honest_uncertain_p2": p2_uncertain_from_labeled,
            # cols where pass-1 abstained — CatBoost gave them a
            # pseudo-label which went to LLM on revisit
            "abstention_resolved_correct": abstention_resolved_correct,
            "abstention_resolved_wrong": abstention_resolved_wrong,
            "abstention_resolved_honest_uncertain": abstention_resolved_uncertain,
            "abstention_still_unresolved": abstention_still_unresolved,
        },
        "fidelity_vs_published_gt": {
            "pass1": round(pass1_fidelity, 4),
            "pass2_merged": round(final_fidelity, 4),
            "delta": round(final_fidelity - pass1_fidelity, 4),
        },
    }

    (run_dir / "revisit_metadata.json").write_text(json.dumps(summary, indent=2))

    # Per-column revisit records
    rows = []
    for i, s in enumerate(samples):
        rows.append({
            "table": s.table_name,
            "column_id": s.name,
            "published_gt": s.reference_code,
            "pass1_label": pass1_labels[i],
            "in_revisit_subset": i in pass2_labels,
            "pass2_label": pass2_labels.get(i, ""),
            "final_label": final_labels[i],
            "catboost_top1": catboost_top3[i][0][0],
            "catboost_top1_prob": catboost_top3[i][0][1],
            "catboost_agrees_with_pass1": catboost_top3[i][0][0] == pass1_labels[i],
        })
    import pandas as pd
    pd.DataFrame(rows).to_parquet(run_dir / "revisit_results.parquet")

    print("\n=== revisit convergence summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\n  revisit metadata: {run_dir / 'revisit_metadata.json'}")
    print(f"  revisit results : {run_dir / 'revisit_results.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
