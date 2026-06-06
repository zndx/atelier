"""Hermetic ``--fixture`` mode for ``just optimize svm``.

Trains + promotes an NHSVM head from the committed, PUBLIC ``test-gittables``
fixture — no Hive, no agent, no LLM, no customer surface (fail-closed). The
deterministic fast path that gives the SVM critical-path test a real
registered head on demand, isolated under ``taxonomy_id='test-gittables'``.

It loads the CCO-rooted ``taxonomy.json`` so every leaf ``ReferenceCategory``
carries its referent ``cco_module`` — which means classifications produced by
the promoted head surface ``semantic_absences`` (a ``mass`` column ->
``unit: UNRESOLVED``). See docs/src/architecture/cco-coverage.md and the
fixture PROVENANCE.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger("atelier.optimize.svm.fixture")

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "classify" / "fixtures" / "test-gittables"
)
ENCODER_ID = "answerdotai/ModernBERT-base"
EMBED_DIM = 768

# Path fragments that denote a customer/UAT or live-Hive surface. The fixture
# build must never touch these; if a resolved input path contains one, fail
# closed rather than risk leakage (defence-in-depth — the code path here does
# not import the Hive/reference loaders at all).
_FORBIDDEN_FRAGMENTS = ("agent_mediated", "svm_training", "reference_corpus")


def _assert_hermetic() -> None:
    """Fail closed unless the fixture dir is the expected committed, public
    asset set and nothing routes through a customer/Hive surface."""
    if not FIXTURE_DIR.is_dir():
        raise FileNotFoundError(f"fixture dir missing: {FIXTURE_DIR}")
    required = ["taxonomy.json", "train_rows.jsonl", "enrichment_payloads.json",
                "PROVENANCE.md"]
    missing = [f for f in required if not (FIXTURE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"fixture incomplete, missing: {missing}")
    resolved = str(FIXTURE_DIR.resolve())
    if any(frag in resolved for frag in _FORBIDDEN_FRAGMENTS):
        raise RuntimeError(
            f"hermetic guard: fixture path {resolved} hits a non-public "
            f"surface; refusing to build."
        )


def load_fixture_category_set():
    """Build a HierarchicalCategorySet from the CCO-rooted taxonomy.json.

    Leaves carry ``cco_module`` + ``ice_class`` so the promoted head's
    classifications surface semantic absence.
    """
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory

    tax = json.loads((FIXTURE_DIR / "taxonomy.json").read_text())
    enrich = json.loads((FIXTURE_DIR / "enrichment_payloads.json").read_text())
    cats: list[ReferenceCategory] = []

    r = tax["root"]
    cats.append(ReferenceCategory(
        code=r["code"], label=r["label"], embedding_text=r["label"],
        parent_code=r.get("parent_code")))
    for n in tax["internal"]:
        cats.append(ReferenceCategory(
            code=n["code"], label=n["label"], embedding_text=n["label"],
            parent_code=n["parent_code"]))
    for lf in tax["leaves"]:
        e = enrich.get(lf["code"], {})
        desc = e.get("description") or lf["label"]
        cats.append(ReferenceCategory(
            code=lf["code"], label=lf["label"],
            embedding_text=f"{lf['label']}. {desc}",
            description=desc, parent_code=lf["parent_code"],
            cco_module=lf.get("cco_module"), ice_class=lf.get("ice_class"),
            cco_annotations=lf.get("cco_annotations")))
    return HierarchicalCategorySet(name="test-gittables", categories=cats)


def load_fixture_rows():
    """Load train_rows.jsonl into reflect.Row objects (Row schema exactly)."""
    from atelier.optimize.svm.reflect import Row

    rows = []
    for line in (FIXTURE_DIR / "train_rows.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(Row(**json.loads(line)))
    return rows


def _corpus_hash() -> str:
    blob = (FIXTURE_DIR / "train_rows.jsonl").read_bytes()
    return hashlib.sha1(blob).hexdigest()[:12]


def build_and_promote_fixture_head(taxonomy_id: str = "test-gittables") -> dict:
    """Encode -> fit factorized NHSVM -> promote head to ``current``.

    Returns the registry row. Requires a database (registry) and downloads
    ModernBERT once; otherwise hermetic and deterministic.
    """
    import numpy as np

    from atelier.classify.artifact_set import compute_vocab_signature
    from atelier.classify.factorized_nhsvm import (
        NHSVMHeadAdapter, fit_factorized_nhsvm,
    )
    from atelier.optimize.svm.encoder import encode_modernbert
    from atelier.optimize.svm.reflect import build_texts_and_labels
    from atelier.optimize.svm.promote import promote_head
    from atelier.registry.nhsvm_head import promote_to_current

    _assert_hermetic()
    cs = load_fixture_category_set()
    rows = load_fixture_rows()
    texts, labels = build_texts_and_labels(rows)
    log.info("fixture: %d rows, %d leaf labels, %d taxonomy nodes",
             len(rows), len(set(labels)), len(cs.all_categories))

    X = np.asarray(encode_modernbert(texts), dtype=np.float32)
    head, tr = fit_factorized_nhsvm(X, labels, cs, embed_dim=EMBED_DIM,
                                    verbose=False)
    adapter = NHSVMHeadAdapter(
        head, encoder_id=ENCODER_ID, embed_dim=EMBED_DIM,
        training_metadata={"source": "fixture", "training_mode": "fixture"})

    codes = sorted(c.code for c in cs.all_categories)
    row = promote_head(
        adapter, taxonomy_id=taxonomy_id,
        vocab_sig=compute_vocab_signature(codes),
        training_mode="fixture", corpus_hash=_corpus_hash(),
        summary="public test-gittables fixture head (hermetic --fixture build)",
        metrics={"final_train_acc": tr.final_train_acc,
                 "n_rows": len(rows), "n_nodes": len(cs.all_categories)})
    promote_to_current(row["id"])
    log.info("promoted fixture head id=%s head_sig=%s (train_acc=%.3f)",
             row["id"], row["head_sig"], tr.final_train_acc)
    return row


def build_fixture_collection(taxonomy_id: str = "test-gittables") -> str:
    """ColBERT-encode the committed enrichment payloads and upsert them into
    the test-gittables Qdrant collection, then register it as ``current`` so
    the maxsim bridge resolves it. Returns the collection name.

    Runs under the devenv stack (Qdrant + ColBERT encoder + registry DB);
    hermetic and deterministic given the committed enrichment.
    """
    from atelier.classify.colbert_encoder import get_encoder
    from atelier.config import load_config
    from atelier.db.dao import AtelierDao
    from atelier.enrichment.qdrant_writer import (
        AnnotationVectors, build_point, collection_name_for,
        compose_annotation_text, ensure_collection, taxonomy_version_hash,
        upsert_point,
    )
    from qdrant_client import QdrantClient

    _assert_hermetic()
    cfg = load_config()
    enrich = json.loads((FIXTURE_DIR / "enrichment_payloads.json").read_text())
    source_rows = list(enrich.values())  # each: code/label/mnemonic/description/...
    aug, model = "fixture", "colbert-ir/colbertv2.0"
    coll = collection_name_for(taxonomy_id, aug)

    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_http_port)
    encoder = get_encoder()
    dim = int(encoder.encode_single("probe").shape[1])
    ensure_collection(client, collection=coll, embedding_dim=dim, recreate=True)

    vh = taxonomy_version_hash(source_rows)
    for row in source_rows:
        text = compose_annotation_text(row) or row.get("label") or row["code"]
        vecs = AnnotationVectors(colbert=encoder.encode_single(text).tolist())
        point = build_point(
            source_row=row, enrichment=row, vectors=vecs, verifier_results={},
            taxonomy_id=taxonomy_id, taxonomy_version_hash_value=vh,
            taxonomy_version_label="fixture", augmentation_version=aug,
            embedding_model=model, embedding_dim=dim,
            generated_at="fixture", generated_by="fixture")
        upsert_point(client, collection=coll, point=point)

    dao = AtelierDao()
    coll_id = f"{taxonomy_id}-{aug}-{vh[:8]}"
    try:
        dao.register_taxonomy_collection(
            id=coll_id, taxonomy_id=taxonomy_id, source_table="test-gittables",
            qdrant_collection=coll, augmentation_version=aug,
            embedding_model=model, embedding_dim=dim,
            summary="public test-gittables fixture maxsim collection")
    except Exception as exc:  # already registered on a prior identical build
        log.info("collection %s already registered (%s)", coll_id, exc)
    dao.set_current_taxonomy_collection(coll_id)
    log.info("built + promoted fixture collection %s (%d points)", coll, len(source_rows))
    return coll


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Hermetic test-gittables fixture build")
    ap.add_argument("--fixture", action="store_true",
                    help="(accepted for `just optimize svm --fixture` routing)")
    ap.add_argument("--taxonomy-id", default="test-gittables")
    ap.add_argument("--head-only", action="store_true",
                    help="promote the NHSVM head only (skip the maxsim collection)")
    ap.add_argument("--collection-only", action="store_true",
                    help="build the maxsim collection only (skip the head)")
    args = ap.parse_args(argv)

    if not args.collection_only:
        row = build_and_promote_fixture_head(taxonomy_id=args.taxonomy_id)
        print(f"promoted head: id={row['id']} head_sig={row['head_sig']} "
              f"taxonomy_id={row['taxonomy_id']} status=current")
    if not args.head_only:
        coll = build_fixture_collection(taxonomy_id=args.taxonomy_id)
        print(f"promoted collection: {coll} status=current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
