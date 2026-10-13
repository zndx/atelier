"""Score the genus-neighborhood NHSVM baseline on the labeled DDL footprint.

The fit stays on CPU. It does not promote a head and it does not read
``corpus/cta_answer_key.csv``. Three conditions share one whole-table
holdout: real support rows, synthetic tables alone, and real support
plus the synthetic supplement.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from atelier.classify.extension_patterns import extensions_path, load_generators
from atelier.classify.neighborhood_baseline import (
    ancestor_frame,
    label_kind,
    placement,
    prepare_neighborhood_baseline,
    values_are_stub,
)
from atelier.classify.taxonomy import load_annotations_from_filesystem
from atelier.classify.features import FEATURE_NAMES
from atelier.optimize.svm.reflect import build_texts_and_labels

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "build" / "results" / "neighborhood-baseline"
_VARIANTS = 4
_EPOCHS = int(os.environ.get("NEIGHBORHOOD_EPOCHS", "300"))


def _rates(category_set, rows, predicted) -> dict:
    gold = [row.code for row in rows]
    if not gold:
        return {"n": 0}
    kinds = [label_kind(category_set, code) for code in gold]
    places = [
        placement(category_set, code, pred)
        for code, pred in zip(gold, predicted)
    ]
    by_label: dict[str, list[bool]] = {}
    for code, pred in zip(gold, predicted):
        by_label.setdefault(code, []).append(pred == code)

    def exact(mask) -> float | None:
        chosen = [i for i, keep in enumerate(mask) if keep]
        if not chosen:
            return None
        hits = sum(gold[i] == predicted[i] for i in chosen)
        return hits / len(chosen)

    return {
        "n": len(gold),
        "exact": exact([True] * len(gold)),
        "macro_exact": sum(sum(v) / len(v) for v in by_label.values()) / len(by_label),
        "n_labels": len(by_label),
        "child_exact": exact(kind == "child" for kind in kinds),
        "child_n": sum(kind == "child" for kind in kinds),
        "root_exact": exact(kind == "root" for kind in kinds),
        "root_n": sum(kind == "root" for kind in kinds),
        "placement": dict(Counter(places)),
    }


def _encode(texts: list[str], cache: Path) -> np.ndarray:
    digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
    path = cache / f"emb-{digest}.npy"
    if path.is_file():
        logging.info("reusing embeddings %s", path.name)
        return np.load(path)
    from atelier.optimize.svm.encoder import encode_modernbert

    logging.info("encoding %d texts on cpu", len(texts))
    vectors = np.asarray(encode_modernbert(texts, device="cpu"), dtype=np.float32)
    np.save(path, vectors)
    return vectors


def _fit_nhsvm(X, labels, category_set):
    import torch

    from atelier.classify.factorized_nhsvm import fit_factorized_nhsvm

    torch.set_num_threads(8)
    freq = Counter(labels)
    weights = np.asarray([1.0 / freq[label] for label in labels], dtype=np.float32)
    weights *= len(weights) / float(weights.sum())
    head, result = fit_factorized_nhsvm(
        X, labels, category_set, device="cpu", epochs=_EPOCHS,
        sample_weights=weights, verbose=True,
    )
    return head, result


def _predict(head, X) -> list[str]:
    import torch

    return head.predict_codes(torch.tensor(np.asarray(X, dtype=np.float32)))


def _lexical(train_texts, train_labels, eval_texts) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    model = LinearSVC(class_weight="balanced", max_iter=5000)
    model.fit(vectorizer.fit_transform(train_texts), train_labels)
    return model.predict(vectorizer.transform(eval_texts)).tolist()


def _write(report: dict) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main(prepare_only: bool = False) -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_OUT / "train.log", mode="w"),
        ],
    )
    vocabulary = load_annotations_from_filesystem(
        _ROOT / "external" / "sdg-corpora" / "vocabulary" / "annotations.csv",
    )
    generators = load_generators(extensions_path().read_text(encoding="utf-8"))
    plan = prepare_neighborhood_baseline(
        vocabulary, generators=generators, variants_per_code=_VARIANTS,
    )
    # Placeholder values ("Title Words 01") are not the exam. A label
    # with two rich tables already held out a rich table.
    support_stub = sum(values_are_stub(row.sample_values) for row in plan.support_rows)
    holdout_stub = sum(values_are_stub(row.sample_values) for row in plan.holdout_rows)
    plan.support_rows = [
        row for row in plan.support_rows if not values_are_stub(row.sample_values)
    ]
    plan.holdout_rows = [
        row for row in plan.holdout_rows if not values_are_stub(row.sample_values)
    ]
    kept = {row.code for row in plan.support_rows + plan.holdout_rows + plan.synthetic_rows}
    frame = ancestor_frame(vocabulary, kept)
    report = {
        "support_rows": len(plan.support_rows),
        "holdout_rows": len(plan.holdout_rows),
        "synthetic_rows": len(plan.synthetic_rows),
        "support_tables": len(plan.support_tables),
        "holdout_tables": len(plan.holdout_tables),
        "labels": len(plan.labels),
        "generatable": len(plan.generatable),
        "unfilled": plan.unfilled,
        "frame_nodes": len(frame.all_categories),
        "support_stub_rows": support_stub,
        "holdout_stub_rows": holdout_stub,
        "variants_per_code": _VARIANTS,
        "epochs": _EPOCHS,
        "conditions": {},
    }
    _write(report)
    logging.info("CORPUS %s", json.dumps({k: report[k] for k in (
        "support_rows", "holdout_rows", "synthetic_rows", "labels",
        "frame_nodes", "unfilled",
    )}))
    if not plan.holdout_rows or not plan.support_rows:
        logging.error("FAILED empty split")
        return 1

    support_text, support_y = build_texts_and_labels(plan.support_rows)
    holdout_text, _holdout_y = build_texts_and_labels(plan.holdout_rows)
    synth_text, synth_y = build_texts_and_labels(plan.synthetic_rows)
    child_holdout = [
        row for row in plan.holdout_rows if label_kind(vocabulary, row.code) == "child"
    ]
    report["holdout_child_rows"] = len(child_holdout)
    report["holdout_root_rows"] = sum(
        label_kind(vocabulary, row.code) == "root" for row in plan.holdout_rows
    )
    report["support_child_rows"] = sum(
        label_kind(vocabulary, row.code) == "child" for row in plan.support_rows
    )
    report["support_root_rows"] = sum(
        label_kind(vocabulary, row.code) == "root" for row in plan.support_rows
    )
    majority = Counter(support_y).most_common(1)[0][0]
    report["majority_label"] = majority
    report["majority_holdout_exact"] = _rates(
        vocabulary, plan.holdout_rows, [majority] * len(plan.holdout_rows),
    )["exact"]
    report["conditions"]["lexical_real"] = _rates(
        vocabulary, plan.holdout_rows,
        _lexical(support_text, support_y, holdout_text),
    )
    report["conditions"]["lexical_synthetic"] = _rates(
        vocabulary, child_holdout,
        _lexical(synth_text, synth_y, build_texts_and_labels(child_holdout)[0]),
    )
    report["conditions"]["lexical_augmented"] = _rates(
        vocabulary, plan.holdout_rows,
        _lexical(support_text + synth_text, support_y + synth_y, holdout_text),
    )
    value_mask = {name: name == "sample_values" for name in FEATURE_NAMES}
    no_name_mask = {
        name: name not in {"column_name", "source_table"} for name in FEATURE_NAMES
    }
    synth_values, _ = build_texts_and_labels(plan.synthetic_rows, value_mask)
    child_values, _ = build_texts_and_labels(child_holdout, value_mask)
    report["conditions"]["lexical_synthetic_values"] = _rates(
        vocabulary, child_holdout,
        _lexical(synth_values, synth_y, child_values),
    )
    synth_noname, _ = build_texts_and_labels(plan.synthetic_rows, no_name_mask)
    child_noname, _ = build_texts_and_labels(child_holdout, no_name_mask)
    report["conditions"]["lexical_synthetic_no_name"] = _rates(
        vocabulary, child_holdout,
        _lexical(synth_noname, synth_y, child_noname),
    )
    support_values, _ = build_texts_and_labels(plan.support_rows, value_mask)
    holdout_values, _ = build_texts_and_labels(plan.holdout_rows, value_mask)
    report["conditions"]["lexical_real_values"] = _rates(
        vocabulary, plan.holdout_rows,
        _lexical(support_values, support_y, holdout_values),
    )
    _write(report)
    logging.info("LEXICAL %s", json.dumps(report["conditions"]))
    if prepare_only:
        logging.info("DONE prepare-only")
        return 0

    packed = support_text + synth_text + holdout_text
    vectors = _encode(packed, _OUT)
    n_support = len(support_text)
    n_synth = len(synth_text)
    X_support = vectors[:n_support]
    X_synth = vectors[n_support:n_support + n_synth]
    X_holdout = vectors[n_support + n_synth:]
    child_index = [
        i for i, row in enumerate(plan.holdout_rows)
        if label_kind(vocabulary, row.code) == "child"
    ]
    X_child = X_holdout[child_index]

    def nhsvm(name: str, X, labels, eval_rows, eval_X) -> None:
        logging.info("fitting %s rows=%d", name, len(labels))
        head, result = _fit_nhsvm(X, labels, frame)
        head.save(_OUT / "heads" / name)
        predicted = _predict(head, eval_X)
        train_predicted = _predict(head, X)
        report["conditions"][name] = _rates(vocabulary, eval_rows, predicted)
        report["conditions"][name]["train_exact"] = sum(
            pred == label for pred, label in zip(train_predicted, labels)
        ) / len(labels)
        report["conditions"][name]["final_train_acc"] = result.final_train_acc
        report["conditions"][name]["elapsed_sec"] = result.elapsed_sec
        _write(report)
        logging.info("SCORED %s %s", name, json.dumps(report["conditions"][name]))

    nhsvm("nhsvm_real", X_support, support_y, plan.holdout_rows, X_holdout)
    nhsvm("nhsvm_synthetic", X_synth, synth_y, child_holdout, X_child)
    nhsvm(
        "nhsvm_augmented",
        np.vstack([X_support, X_synth]),
        support_y + synth_y,
        plan.holdout_rows,
        X_holdout,
    )
    value_packed = support_values + synth_values + holdout_values
    value_vectors = _encode(value_packed, _OUT)
    Xv_support = value_vectors[:n_support]
    Xv_synth = value_vectors[n_support:n_support + n_synth]
    Xv_holdout = value_vectors[n_support + n_synth:]
    Xv_child = Xv_holdout[child_index]
    nhsvm("nhsvm_real_values", Xv_support, support_y, plan.holdout_rows, Xv_holdout)
    nhsvm("nhsvm_synthetic_values", Xv_synth, synth_y, child_holdout, Xv_child)
    _write(report)
    logging.info("DONE %s", json.dumps(report["conditions"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(prepare_only="--prepare-only" in sys.argv))
    except Exception:
        logging.exception("FAILED")
        raise
