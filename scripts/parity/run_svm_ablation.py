#!/usr/bin/env python
"""Four-arm SVM ablation on the UAT corpus.

Tests two orthogonal hypotheses at once:

  1. Is the current ``build_svm_text`` starving SVM of discriminating
     information that's sitting in ``embedding_text``?
  2. Does a ground-up SVM on the MiniLM sentence-embedding outperform
     the TF-IDF pipeline regardless of text?

Four arms (all trained on the same synthetic corpus, evaluated on the
same real UAT columns with leak-sanitized siblings):

  A. S-short + TF-IDF + LinearSVC   (current production baseline)
  B. S-full  + TF-IDF + LinearSVC   (same classifier, richer text)
  C. E-linear                       (LinearSVC on 384-dim MiniLM embedding)
  D. E-rbf                          (RBF-kernel SVC on MiniLM embedding)

For each arm we report:
  * standalone exact + hierarchical accuracy vs ground truth
  * pairwise prediction agreement across arms + against CatBoost
  * which GT codes each arm wins/loses on the other

Reports land in build/results/svm_ablation/summary.json and
build/results/svm_ablation/per_column.parquet (one row per real column
with every arm's top-1 prediction — suitable for forensic deep-dive).
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _extract_feature_text(sample) -> str:
    """Run the same 12-feature extraction CatBoost uses on a ColumnSample."""
    from atelier.classify.features import extract_features
    feats = extract_features(
        column_name=sample.name,
        column_type=sample.column_type,
        values=sample.values,
        siblings=sample.siblings or [],
        null_count=sample.null_count,
        total_count=sample.total_count,
        source_table=sample.table_name,
        distinct_count=getattr(sample, "distinct_count", 0),
    )
    return feats.to_embedding_text()


def _load_synth_columns(synth_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load synth CSVs + ground_truth.json — mirrors ml_train._load_synth_data."""
    import csv
    import json

    ground_truth_path = synth_dir / "ground_truth.json"
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"missing {ground_truth_path}")
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    columns: dict[str, list[str]] = {}
    for csv_path in synth_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            rows = list(reader)
        for j, col_name in enumerate(header):
            vals = [r[j] for r in rows if j < len(r) and r[j]]
            columns[col_name] = vals
    return columns, ground_truth


def _build_svm_short(name: str, values: list[str]) -> str:
    """Current production ``build_svm_text`` shape."""
    from atelier.classify.svm_classifier import build_svm_text
    return build_svm_text(name, None, values[:5])


def _build_embedding_text_synth(name: str, values: list[str]) -> str:
    """Run feature extraction on a synth column (no siblings/stats beyond vals)."""
    from atelier.classify.features import extract_features
    feats = extract_features(
        column_name=name,
        column_type=None,
        values=values,
        siblings=[],
        null_count=0,
        total_count=len(values),
        source_table="synth",
        distinct_count=len(set(values)),
    )
    return feats.to_embedding_text()


# Template anchors we author ourselves in ``features.to_embedding_text()``.
# Because we own the template, we know with certainty which tokens are
# structural skeleton (non-discriminative, appear in every column) versus
# which tokens carry signal (values, pattern names, descriptions, etc.).
# Strip only the anchor strings — not the content they wrap — so TF-IDF's
# 50k-feature budget concentrates on the discriminative vocabulary rather
# than on char n-grams of ``cardinality=``, ``entropy=``, and the like.
#
# Char n-grams can't use a standard word-level ``stop_words`` parameter,
# so pre-stripping is the surgical tool.  Conservative: strip exactly the
# tokens we chose to put in; don't touch generic language in descriptions.
_TEMPLATE_ANCHORS = (
    # Exhaustively enumerated from ``features.ColumnFeatures.to_embedding_text``
    # — every ``f"<key>=..."`` prefix and every ``"<key>: "`` prefix the
    # template emits.  Kept in sync with that function; adding a new
    # structural key there means adding it here.
    "cardinality=", "null_ratio=", "entropy=",
    "avg_len=", "numeric=",
    "table=",
    "patterns: ", "siblings: ",
)


def _strip_template_anchors(text: str) -> str:
    """Remove template keywords AND the ``|`` separator.

    Arm E — the naive "stop-word" move.  Surprisingly hurts performance
    because the ``|`` was a real boundary token: without it, structured
    fields bleed into each other (``99 1.52 3.8 last name, email …``).
    """
    for anchor in _TEMPLATE_ANCHORS:
        text = text.replace(anchor, " ")
    text = text.replace("|", " ")
    return " ".join(text.split())


def _strip_anchors_keep_pipes(text: str) -> str:
    """Arm F — strip keyword anchors but preserve ``|`` as field boundary.

    Controls for the effect isolated from the separator removal.  Tests
    whether the anchors themselves carry signal (beyond just delimiting).
    """
    for anchor in _TEMPLATE_ANCHORS:
        text = text.replace(anchor, "")
    return " ".join(text.split())


# Compact type markers — replace ``cardinality=99`` with ``CARD99``, etc.
# Preserves type context in a dense form that costs fewer vocabulary slots.
# Arm G tests whether this ends up outperforming bare anchors (Arm B).
_COMPACT_MAP = {
    "cardinality=": "CARD",
    "null_ratio=": "NULLR",
    "entropy=": "ENT",
    "avg_len=": "AVGL",
    "numeric=": "NUM",
    "table=": "TBL_",
    "patterns: ": "PAT_",
    "siblings: ": "SIB_",
}


def _compact_anchors(text: str) -> str:
    """Arm G — replace anchors with short typed prefixes.

    ``cardinality=99`` → ``CARD99``.  Same type-context information in
    ~5× fewer char n-gram fragments, freeing vocabulary budget.
    """
    for anchor, short in _COMPACT_MAP.items():
        text = text.replace(anchor, short)
    return text


def _tfidf_linear_svc(*, multi_class: str = "ovr"):
    """TF-IDF (char+word) + LinearSVC, optionally Crammer-Singer.

    ``multi_class="ovr"``             — sklearn default; train K
        independent binary classifiers.  Arms A, B, E, F, G.
    ``multi_class="crammer_singer"``  — true joint multi-class SVM
        (Crammer & Singer 2001), single weight matrix optimised so
        class y's margin dominates max_{y'≠y} over sibling classes.
        Arms H, I.

    Platt calibration via ``CalibratedClassifierCV`` wraps either —
    it calibrates per-class against ``decision_function`` output
    regardless of the underlying multi_class setting.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    features = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 6),
            max_features=50_000, lowercase=True, sublinear_tf=True,
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=50_000, lowercase=True, sublinear_tf=True,
        )),
    ])
    svc_kwargs = {"C": 1.0, "max_iter": 2000}
    if multi_class == "crammer_singer":
        # Crammer-Singer requires primal solver; dual=False is incompatible
        # with hinge loss, so we use the default loss.  class_weight
        # doesn't apply to the joint formulation.
        svc_kwargs["multi_class"] = "crammer_singer"
    return Pipeline([
        ("feats", features),
        ("svc", CalibratedClassifierCV(
            LinearSVC(**svc_kwargs), method="sigmoid", cv=3,
        )),
    ])


def _linear_svc_on_embedding():
    """LinearSVC directly on the MiniLM 384-dim embedding."""
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000), method="sigmoid", cv=3,
    )


def _rbf_svc_on_embedding():
    """RBF-kernel SVC on the MiniLM embedding (probability=True for proba out)."""
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, max_iter=-1)


def _encode_embeddings(texts: list[str]):
    from atelier.classify.embedding import embed_texts
    import numpy as np
    return np.asarray(embed_texts(texts))


def _score(pred_codes: list[str], gt_codes: list[str]) -> tuple[float, float]:
    total = sum(1 for g in gt_codes if g)
    if total == 0:
        return 0.0, 0.0
    exact = sum(1 for p, g in zip(pred_codes, gt_codes) if g and p == g) / total
    hier_hits = 0
    for p, g in zip(pred_codes, gt_codes):
        if not g:
            continue
        if p == g or (p and g.startswith(p + ".")) or (g and p.startswith(g + ".")):
            hier_hits += 1
    return exact, hier_hits / total


def main() -> int:
    _setup_logging()
    log = logging.getLogger("svm_ablation")

    # ── 1.  Load real UAT samples (leak-sanitized) ──────────────
    from atelier.classify.meta_tagging_source import (
        load_meta_tagging_source,
        load_meta_tagging_vocabulary,
        resolve_meta_tagging_mount,
    )
    mount = resolve_meta_tagging_mount()
    if mount is None:
        log.error("no meta-tagging mount")
        return 1
    samples = load_meta_tagging_source(mount)
    category_set = load_meta_tagging_vocabulary(mount)
    real_cols: list = []
    for t in samples:
        for c in t.columns:
            real_cols.append((t.name, c))
    real_with_gt = [(tn, c) for tn, c in real_cols if c.ground_truth]
    log.info(
        "real corpus: %d total cols, %d with GT (leak-sanitized)",
        len(real_cols), len(real_with_gt),
    )

    # ── 2.  Generate synth training data once (reused across arms) ─
    from atelier.classify.real_data_loader import extract_value_templates
    from atelier.classify.synth import generate_synth_tables

    work = Path("build/results/svm_ablation")
    work.mkdir(parents=True, exist_ok=True)
    synth_dir = work / "synth"
    templates = extract_value_templates(mount)
    log.info("value templates: %d codes", len(templates))
    generate_synth_tables(
        category_set, synth_dir,
        value_templates=templates, variants_per_category=30, seed=42,
    )
    synth_cols, synth_gt = _load_synth_columns(synth_dir)
    log.info("synth training: %d columns", len(synth_cols))

    # Align train set in a stable order
    train_names = [n for n in synth_cols if synth_gt.get(n)]
    train_values = [synth_cols[n] for n in train_names]
    train_labels = [synth_gt[n] for n in train_names]

    # ── 3.  Build each arm's train + predict fn ─────────────────
    def _arm_result(name, train_fn, predict_fn):
        log.info("arm %s: training", name)
        train_fn()
        log.info("arm %s: predicting on %d real GT cols", name, len(real_with_gt))
        preds = predict_fn([c for _, c in real_with_gt])
        gts   = [c.ground_truth for _, c in real_with_gt]
        ex, hi = _score(preds, gts)
        log.info("arm %s: exact=%.4f hier=%.4f", name, ex, hi)
        return preds, ex, hi

    results: dict[str, dict] = {}
    arm_preds: dict[str, list[str]] = {}

    # Arm A — S-short + TF-IDF + LinearSVC
    model_a = _tfidf_linear_svc()
    def train_a():
        texts = [_build_svm_short(n, vs) for n, vs in zip(train_names, train_values)]
        model_a.fit(texts, train_labels)
    def predict_a(cols):
        texts = [_build_svm_short(c.name, c.values) for c in cols]
        return list(model_a.predict(texts))
    preds, ex, hi = _arm_result("A:S-short+tfidf", train_a, predict_a)
    arm_preds["A_short_tfidf"] = preds
    results["A_short_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm B — S-full + TF-IDF + LinearSVC
    model_b = _tfidf_linear_svc()
    def train_b():
        texts = [_build_embedding_text_synth(n, vs) for n, vs in zip(train_names, train_values)]
        model_b.fit(texts, train_labels)
    def predict_b(cols):
        texts = [_extract_feature_text(c) for c in cols]
        return list(model_b.predict(texts))
    preds, ex, hi = _arm_result("B:S-full+tfidf", train_b, predict_b)
    arm_preds["B_full_tfidf"] = preds
    results["B_full_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm C — LinearSVC on MiniLM embeddings
    import numpy as np
    train_emb = None
    model_c = _linear_svc_on_embedding()
    def train_c():
        nonlocal train_emb
        log.info("encoding %d synth texts for embedding arms", len(train_names))
        synth_texts = [_build_embedding_text_synth(n, vs) for n, vs in zip(train_names, train_values)]
        train_emb = _encode_embeddings(synth_texts)
        model_c.fit(train_emb, train_labels)
    def predict_c(cols):
        texts = [_extract_feature_text(c) for c in cols]
        emb = _encode_embeddings(texts)
        return list(model_c.predict(emb))
    preds, ex, hi = _arm_result("C:linear-on-embed", train_c, predict_c)
    arm_preds["C_linear_embed"] = preds
    results["C_linear_embed"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm E — S-full *with template anchors stripped* + TF-IDF + LinearSVC.
    # Tests whether the 19-point loss in Arm B was actually about TF-IDF
    # drowning on boilerplate tokens (``cardinality=``, ``entropy=``, etc.)
    # rather than a fundamental rich-text problem.  We author the template,
    # so we know exactly which tokens are structural skeleton.
    model_e = _tfidf_linear_svc()
    def train_e():
        texts = [
            _strip_template_anchors(_build_embedding_text_synth(n, vs))
            for n, vs in zip(train_names, train_values)
        ]
        model_e.fit(texts, train_labels)
    def predict_e(cols):
        texts = [_strip_template_anchors(_extract_feature_text(c)) for c in cols]
        return list(model_e.predict(texts))
    preds, ex, hi = _arm_result("E:S-full-stopwords+tfidf", train_e, predict_e)
    arm_preds["E_full_stopwords_tfidf"] = preds
    results["E_full_stopwords_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm F — strip keyword anchors but KEEP the ``|`` field separator.
    model_f = _tfidf_linear_svc()
    def train_f():
        texts = [
            _strip_anchors_keep_pipes(_build_embedding_text_synth(n, vs))
            for n, vs in zip(train_names, train_values)
        ]
        model_f.fit(texts, train_labels)
    def predict_f(cols):
        texts = [_strip_anchors_keep_pipes(_extract_feature_text(c)) for c in cols]
        return list(model_f.predict(texts))
    preds, ex, hi = _arm_result("F:S-full-nokw-pipes+tfidf", train_f, predict_f)
    arm_preds["F_full_nokw_pipes_tfidf"] = preds
    results["F_full_nokw_pipes_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm G — compact typed markers: ``cardinality=99`` → ``CARD99``.
    model_g = _tfidf_linear_svc()
    def train_g():
        texts = [
            _compact_anchors(_build_embedding_text_synth(n, vs))
            for n, vs in zip(train_names, train_values)
        ]
        model_g.fit(texts, train_labels)
    def predict_g(cols):
        texts = [_compact_anchors(_extract_feature_text(c)) for c in cols]
        return list(model_g.predict(texts))
    preds, ex, hi = _arm_result("G:S-full-compact+tfidf", train_g, predict_g)
    arm_preds["G_full_compact_tfidf"] = preds
    results["G_full_compact_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm H — Crammer-Singer multi-class SVM on S-short.
    # Direct comparison against Arm A: same features, same TF-IDF, same
    # corpus, only the multi-class decomposition differs.  Tests whether
    # the joint-margin formulation recovers accuracy on leaf-vs-parent
    # taxonomy errors (first_name vs Person Name) that OvR might miss.
    model_h = _tfidf_linear_svc(multi_class="crammer_singer")
    def train_h():
        texts = [_build_svm_short(n, vs) for n, vs in zip(train_names, train_values)]
        model_h.fit(texts, train_labels)
    def predict_h(cols):
        texts = [_build_svm_short(c.name, c.values) for c in cols]
        return list(model_h.predict(texts))
    preds, ex, hi = _arm_result("H:CS-short+tfidf", train_h, predict_h)
    arm_preds["H_cs_short_tfidf"] = preds
    results["H_cs_short_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm I — Crammer-Singer on the full embedding_text (no stop-wording).
    # Direct comparison against Arm B: does C-S's intrinsic sibling-class
    # awareness compensate for the vocabulary-saturation problem that sunk
    # OvR on rich text?
    model_i = _tfidf_linear_svc(multi_class="crammer_singer")
    def train_i():
        texts = [_build_embedding_text_synth(n, vs) for n, vs in zip(train_names, train_values)]
        model_i.fit(texts, train_labels)
    def predict_i(cols):
        texts = [_extract_feature_text(c) for c in cols]
        return list(model_i.predict(texts))
    preds, ex, hi = _arm_result("I:CS-full+tfidf", train_i, predict_i)
    arm_preds["I_cs_full_tfidf"] = preds
    results["I_cs_full_tfidf"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # Arm D — RBF-SVC on MiniLM embeddings
    model_d = _rbf_svc_on_embedding()
    def train_d():
        # reuse train_emb from arm C if present; otherwise recompute
        nonlocal train_emb
        if train_emb is None:
            synth_texts = [_build_embedding_text_synth(n, vs) for n, vs in zip(train_names, train_values)]
            train_emb = _encode_embeddings(synth_texts)
        model_d.fit(train_emb, train_labels)
    def predict_d(cols):
        texts = [_extract_feature_text(c) for c in cols]
        emb = _encode_embeddings(texts)
        return list(model_d.predict(emb))
    preds, ex, hi = _arm_result("D:rbf-on-embed", train_d, predict_d)
    arm_preds["D_rbf_embed"] = preds
    results["D_rbf_embed"] = {"exact": round(ex, 4), "hier": round(hi, 4)}

    # ── 4.  Pairwise agreement matrix ───────────────────────────
    arm_keys = list(arm_preds.keys())
    agreement: dict[str, dict[str, float]] = {}
    for i, a in enumerate(arm_keys):
        agreement[a] = {}
        for b in arm_keys:
            eq = sum(1 for p, q in zip(arm_preds[a], arm_preds[b]) if p == q)
            agreement[a][b] = round(eq / len(arm_preds[a]), 4) if arm_preds[a] else 0.0

    # ── 5.  Per-column parquet for forensic digging ─────────────
    import pandas as pd
    rows = []
    gts = [c.ground_truth for _, c in real_with_gt]
    for idx, (tn, c) in enumerate(real_with_gt):
        row = {
            "table": tn, "column": c.name,
            "ground_truth": c.ground_truth,
        }
        for k in arm_keys:
            row[k] = arm_preds[k][idx]
        row["consensus"] = len({row[k] for k in arm_keys})
        rows.append(row)
    per_col_df = pd.DataFrame(rows)
    per_col_path = work / "per_column.parquet"
    per_col_df.to_parquet(per_col_path)

    summary = {
        "real_gt_columns": len(real_with_gt),
        "synth_training_columns": len(train_names),
        "arm_accuracy": results,
        "pairwise_agreement": agreement,
        "per_column_parquet": str(per_col_path),
    }
    summary_path = work / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SVM ablation ===")
    print(json.dumps(summary, indent=2))
    print(f"\nsummary   : {summary_path}")
    print(f"per-column: {per_col_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
