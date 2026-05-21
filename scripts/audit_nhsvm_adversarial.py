#!/usr/bin/env python3
"""DST sensitivity analysis for the NHSVM evidence path.

Three variants of the per-vocabulary SVM are compared as candidates
for the production NHSVM implementation:

  A  ovr-universal      Current production: one-vs-rest LinearSVC
                        trained on label-conditional Kronecker-expanded
                        features; inference universally populates all
                        node blocks with ``sqrt(alpha_n)`` scaling.

  B  ovr-per-class      Same trained model as A; at inference, for each
                        candidate ``y`` build ``Lambda(y) ⊗ x`` with only
                        ``path(y)`` blocks active and read the model's
                        ``p_y`` under that expansion.  Normalize across y.

  C  joint-per-class    Crammer-Singer joint multi-class LinearSVC
                        trained on the same Kronecker-expanded features;
                        per-class inference like B.  Closest sklearn-
                        expressible approximation to Choi et al. (2015)
                        Structured Shared Frobenius Norm SVM.

Four DST factor sweeps characterize the parameter sensitivity of each
variant's fused-headline outcomes:

  - SVM mass discount
  - Runtime LLM confidence
  - Runtime LLM mass discount
  - Fusion strategy (dempster vs yager)

Per-tier accuracy is the primary metric.  Correctness is tier-defined:

  - easy / hard / contested / sparse / svm-was-right
        correct = fused headline equals the expected leaf
  - semantic-conflict
        correct = fused headline equals the LLM's vote code
                  (i.e., the parent held; no specific leaf is "right")

Inflection points — the parameter value at which a variant's per-tier
accuracy first reaches 0.5 — summarize each sweep compactly.  Raw
sweep data is dumped to ``build/audit/nhsvm/sensitivity_sweep.json``
for further analysis.

Run from project root::

    uv run python scripts/audit_nhsvm_adversarial.py
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.WARNING)


# ────────────────────────────────────────────────────────────────────
# 1.  Adversarial taxonomy
# ────────────────────────────────────────────────────────────────────

def build_adversarial_taxonomy():
    """Asymmetric tree: deep PII subtree + shallow OPERATIONAL subtree.

    The deep side mimics a real sensitive-data taxonomy; the shallow
    side is the classic "catch-all" that can absorb mislabels under a
    flat SVM.

    Address is intentionally a *parent* with SHIPPING / BILLING leaves
    so the production observation can be replayed: LLM cautious at
    ``ADDR`` parent ("can't reconcile origin_doc with shipping"), SVM
    committed at ``ADDR.SHIPPING`` leaf (values match shipping shape).
    """
    from atelier.classify.taxonomy import (
        HierarchicalCategorySet,
        ReferenceCategory,
    )

    cats = [
        ReferenceCategory(code="ROOT", label="Root", embedding_text="root", parent_code=None),
        ReferenceCategory(code="ROOT.SENSITIVE", label="Sensitive", embedding_text="sensitive", parent_code="ROOT"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII", label="PII", embedding_text="personally identifying", parent_code="ROOT.SENSITIVE"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.PERSON", label="Person", embedding_text="person identifier", parent_code="ROOT.SENSITIVE.PII"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.PERSON.NAME", label="Name", embedding_text="given name", parent_code="ROOT.SENSITIVE.PII.PERSON"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.PERSON.EMAIL", label="Email", embedding_text="email address", parent_code="ROOT.SENSITIVE.PII.PERSON"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.CONTACT", label="Contact", embedding_text="contact information", parent_code="ROOT.SENSITIVE.PII"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.CONTACT.PHONE", label="Phone", embedding_text="phone number", parent_code="ROOT.SENSITIVE.PII.CONTACT"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.CONTACT.ADDR", label="Address", embedding_text="full address", parent_code="ROOT.SENSITIVE.PII.CONTACT"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.CONTACT.ADDR.SHIPPING", label="Shipping Address", embedding_text="shipping destination address", parent_code="ROOT.SENSITIVE.PII.CONTACT.ADDR"),
        ReferenceCategory(code="ROOT.SENSITIVE.PII.CONTACT.ADDR.BILLING", label="Billing Address", embedding_text="billing address", parent_code="ROOT.SENSITIVE.PII.CONTACT.ADDR"),
        ReferenceCategory(code="ROOT.SENSITIVE.FIN", label="Financial", embedding_text="financial data", parent_code="ROOT.SENSITIVE"),
        ReferenceCategory(code="ROOT.SENSITIVE.FIN.CARD", label="Card", embedding_text="payment card number", parent_code="ROOT.SENSITIVE.FIN"),
        ReferenceCategory(code="ROOT.SENSITIVE.FIN.BANK", label="Bank", embedding_text="bank account number", parent_code="ROOT.SENSITIVE.FIN"),
        ReferenceCategory(code="ROOT.OPERATIONAL", label="Operational", embedding_text="operational metadata", parent_code="ROOT"),
        ReferenceCategory(code="ROOT.OPERATIONAL.ID", label="Identifier", embedding_text="generic identifier", parent_code="ROOT.OPERATIONAL"),
        ReferenceCategory(code="ROOT.OPERATIONAL.TIME", label="Timestamp", embedding_text="timestamp", parent_code="ROOT.OPERATIONAL"),
        ReferenceCategory(code="ROOT.OPERATIONAL.STATUS", label="Status", embedding_text="status flag", parent_code="ROOT.OPERATIONAL"),
    ]
    return HierarchicalCategorySet(name="adversarial", categories=cats)


# ────────────────────────────────────────────────────────────────────
# 2.  Synthetic training data
# ────────────────────────────────────────────────────────────────────

LEAF_PATTERNS = {
    "ROOT.SENSITIVE.PII.PERSON.NAME": (
        ["full_name", "first_name", "last_name", "given_name", "surname", "person_name"],
        ["Alice Smith", "Bob Jones", "Carol Lee", "David Kim", "Eve Park"],
    ),
    "ROOT.SENSITIVE.PII.PERSON.EMAIL": (
        ["email", "email_addr", "contact_email", "user_email", "primary_email"],
        ["alice@example.com", "bob.jones@corp.io", "carol@test.org", "dkim@x.com", "eve@p.io"],
    ),
    "ROOT.SENSITIVE.PII.CONTACT.PHONE": (
        ["phone", "phone_number", "mobile", "cell_phone", "tel"],
        ["555-123-4567", "+1-555-987-6543", "212-555-0100", "415-555-2200", "800-555-1212"],
    ),
    "ROOT.SENSITIVE.PII.CONTACT.ADDR.SHIPPING": (
        ["shipping_addr", "ship_to_addr", "delivery_addr", "shipping_address",
         "ship_address", "warehouse_addr", "destination_addr"],
        ["123 Main St Apt 4B, Springfield, IL 62701",
         "456 Oak Ave Unit 12, Portland, OR 97205",
         "789 Pine Rd, Suite 200, Austin, TX 78701",
         "101 Elm Blvd #305, Seattle, WA 98101",
         "202 Maple Dr, Building C, Denver, CO 80202"],
    ),
    "ROOT.SENSITIVE.PII.CONTACT.ADDR.BILLING": (
        ["billing_addr", "billing_address", "invoice_addr", "bill_to_addr",
         "billing_street", "remit_addr"],
        ["PO Box 1234, Springfield, IL 62701",
         "PO Box 5678, Portland, OR 97205",
         "PO Box 9012, Austin, TX 78701",
         "PO Box 3456, Seattle, WA 98101",
         "PO Box 7890, Denver, CO 80202"],
    ),
    "ROOT.SENSITIVE.FIN.CARD": (
        ["card_number", "credit_card", "payment_card", "card_num", "cc_num"],
        ["4111-1111-1111-1111", "5500-0000-0000-0004", "3400-0000-0000-009", "6011-0000-0000-0004", "3088-0000-0000-0009"],
    ),
    "ROOT.SENSITIVE.FIN.BANK": (
        ["account_number", "bank_account", "account_num", "checking_acct", "iban"],
        ["GB29NWBK60161331926819", "DE89370400440532013000", "1234567890", "9876543210", "5555444433"],
    ),
    "ROOT.OPERATIONAL.ID": (
        ["id", "record_id", "row_id", "uuid", "guid", "ref_id"],
        ["abc-123-def", "0001-2222-3333", "ref_99887", "xyz-001", "rid-42"],
    ),
    "ROOT.OPERATIONAL.TIME": (
        ["created_at", "updated_at", "timestamp", "event_time", "modified"],
        ["2026-05-21 10:00:00", "2026-04-15T12:30:00Z", "1716192000", "2026-05-21", "10:00 AM"],
    ),
    "ROOT.OPERATIONAL.STATUS": (
        ["status", "state", "flag", "active", "is_enabled"],
        ["active", "pending", "disabled", "1", "0"],
    ),
}


def generate_training_data(n_per_leaf: int = 30, seed: int = 0):
    """Generate (text, label) pairs using the standard build_svm_text shape."""
    from atelier.classify.svm_classifier import build_svm_text

    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[str] = []
    for code, (name_templates, value_templates) in LEAF_PATTERNS.items():
        for _ in range(n_per_leaf):
            col_name = rng.choice(name_templates)
            if rng.random() < 0.3:
                col_name = f"{col_name}_{rng.randint(1, 9)}"
            values = rng.sample(value_templates, k=min(5, len(value_templates)))
            text = build_svm_text(col_name, sample_values=values)
            texts.append(text)
            labels.append(code)
    return texts, labels


def build_adversarial_test_set():
    """Test cases that exercise the variants on six difficulty tiers.

    Each case is a 4-tuple ``(text, expected, tier, llm_vote_code)``:

      ``text``           — the input the SVM scores
      ``expected``       — the leaf the SVM should reach (or
                            ``AMBIGUOUS_ADDR`` sentinel for semantic-
                            conflict cases where no single leaf is right)
      ``tier``           — difficulty bucket (see docstring)
      ``llm_vote_code``  — the code the simulated runtime LLM commits to
                            (= ``expected`` for non-conflict tiers; the
                            ADDR parent for semantic-conflict; a parent
                            or grandparent of ``expected`` for svm-was-
                            right cases where the LLM was overcautious)
    """
    from atelier.classify.svm_classifier import build_svm_text

    ADDR_PARENT = "ROOT.SENSITIVE.PII.CONTACT.ADDR"

    cases = [
        # ── easy: PII targets with distracting names ──────────────
        ("user_id_email", "ROOT.SENSITIVE.PII.PERSON.EMAIL", "easy",
         ["alice@example.com", "bob@corp.io", "c@x.com", "d@y.org", "e@z.net"], None),
        ("contact_record_id", "ROOT.SENSITIVE.PII.PERSON.EMAIL", "easy",
         ["dkim@example.com", "evepark@test.io", "carol.lee@corp.com", "alice@x.io", "bob@y.com"], None),
        ("name_status", "ROOT.SENSITIVE.PII.PERSON.NAME", "easy",
         ["Alice Smith", "Bob Jones", "Carol Lee", "David Kim", "Eve Park"], None),
        ("record_id", "ROOT.OPERATIONAL.ID", "easy",
         ["abc-123-def", "rid-42", "0001-2222", "ref_99887", "xyz-001"], None),
        ("event_timestamp", "ROOT.OPERATIONAL.TIME", "easy",
         ["2026-05-21 10:00:00", "1716192000", "2026-04-15", "12:30:00", "Monday"], None),

        # ── hard: generic column names, value signal correct ─────
        ("data", "ROOT.SENSITIVE.PII.CONTACT.PHONE", "hard",
         ["555-123-4567", "+1-555-987-6543", "212-555-0100", "415-555-2200", "800-555-1212"], None),
        ("value", "ROOT.SENSITIVE.PII.PERSON.EMAIL", "hard",
         ["alice@x.com", "b@y.io", "c@z.org", "d@w.net", "e@v.io"], None),
        ("contact_name_or_addr", "ROOT.SENSITIVE.PII.CONTACT.ADDR.SHIPPING", "hard",
         ["123 Main St Apt 4B, Springfield, IL 62701",
          "456 Oak Ave Unit 12, Portland, OR 97205",
          "789 Pine Rd, Suite 200, Austin, TX 78701",
          "101 Elm Blvd #305, Seattle, WA 98101",
          "202 Maple Dr, Building C, Denver, CO 80202"], None),

        # ── semantic-conflict: directional qualifier contradicts leaf
        # role connotation; values fit a leaf's shape.  LLM votes at
        # the address parent; "correct" outcome = parent held.
        ("origin_doc", "AMBIGUOUS_ADDR", "semantic-conflict",
         ["123 Main St Apt 4B, Springfield, IL 62701",
          "456 Oak Ave Unit 12, Portland, OR 97205",
          "789 Pine Rd, Suite 200, Austin, TX 78701",
          "101 Elm Blvd #305, Seattle, WA 98101",
          "202 Maple Dr, Building C, Denver, CO 80202"], ADDR_PARENT),
        ("source_location", "AMBIGUOUS_ADDR", "semantic-conflict",
         ["555 Cherry Lane, Madison, WI 53703",
          "888 Walnut St, Boise, ID 83702",
          "999 Birch Way, Reno, NV 89501",
          "111 Spruce Ct, Tampa, FL 33602",
          "222 Cedar Pl, Buffalo, NY 14202"], ADDR_PARENT),
        ("destination_record", "AMBIGUOUS_ADDR", "semantic-conflict",
         ["PO Box 1234, Springfield, IL 62701",
          "PO Box 5678, Portland, OR 97205",
          "PO Box 9012, Austin, TX 78701",
          "PO Box 3456, Seattle, WA 98101",
          "PO Box 7890, Denver, CO 80202"], ADDR_PARENT),

        # ── svm-was-right: uninformative column names, unambiguous
        # value signals.  LLM hedges at a parent/grandparent because
        # the name gives no commit signal; SVM has the right leaf.
        # "Correct" outcome = headline at the expected leaf.
        ("field_42", "ROOT.SENSITIVE.FIN.CARD", "svm-was-right",
         ["4111111111111111", "5500000000000004", "3400000000000099",
          "6011000000000004", "3088000000000009"],
         "ROOT.SENSITIVE.FIN"),
        ("col_07", "ROOT.SENSITIVE.PII.PERSON.EMAIL", "svm-was-right",
         ["alice@x.com", "b.j@y.io", "carol@corp.com", "d.k@w.net", "eve@z.io"],
         "ROOT.SENSITIVE.PII"),
        ("attr_3", "ROOT.SENSITIVE.PII.CONTACT.PHONE", "svm-was-right",
         ["555-123-4567", "+1-555-987-6543", "212-555-0100",
          "415-555-2200", "800-555-1212"],
         "ROOT.SENSITIVE.PII.CONTACT"),
        ("data_blob", "ROOT.OPERATIONAL.TIME", "svm-was-right",
         ["2026-05-21 10:00:00", "1716192000", "2026-04-15",
          "12:30:00", "Monday"],
         "ROOT.OPERATIONAL"),
        ("entry_x", "ROOT.SENSITIVE.PII.PERSON.NAME", "svm-was-right",
         ["Alice Smith", "Bob Jones", "Carol Lee", "David Kim", "Eve Park"],
         "ROOT.SENSITIVE.PII.PERSON"),

        # ── contested: digits-only — ID/BANK/CARD all plausible ──
        ("ref_account", "ROOT.SENSITIVE.FIN.BANK", "contested",
         ["1234567890", "9876543210", "5555444433", "1111222233", "9999888877"], None),
        ("payment_ref", "ROOT.SENSITIVE.FIN.CARD", "contested",
         ["4111111111111111", "5500000000000004", "3400000000000099", "6011000000000004", "3088000000000009"], None),
        ("user_id", "ROOT.SENSITIVE.PII.PERSON.NAME", "contested",
         ["Alice Smith", "Bob Jones", "Carol Lee", "David Kim", "Eve Park"], None),
        ("record", "ROOT.OPERATIONAL.TIME", "contested",
         ["2026-05-21", "2026-04-15", "2025-12-31", "2026-01-01", "2026-06-30"], None),
        ("acct_handle", "ROOT.SENSITIVE.PII.PERSON.EMAIL", "contested",
         ["alice.smith.42@x.io", "b.j.27@y.io", "c.l.11@z.io", "d.k.5@w.io", "e.p.9@v.io"], None),

        # ── sparse-cue: only column name carries the signal ──────
        ("status", "ROOT.OPERATIONAL.STATUS", "sparse",
         ["abc", "xyz", "qqq", "rrr", "sss"], None),
        ("phone", "ROOT.SENSITIVE.PII.CONTACT.PHONE", "sparse",
         ["abc", "xyz", "qqq", "rrr", "sss"], None),
    ]
    return [
        (
            build_svm_text(name, sample_values=values),
            expected,
            tier,
            llm_vote_code if llm_vote_code is not None else expected,
        )
        for name, expected, tier, values, llm_vote_code in cases
    ]


# ────────────────────────────────────────────────────────────────────
# 3.  Variant B: per-class inference on Variant-A trained model
# ────────────────────────────────────────────────────────────────────

def predict_proba_per_class(svm_classifier, text: str) -> dict[str, float]:
    """Variant B: per-class inference expansion on Variant-A's trained model.

    For each candidate ``y`` the inference feature is rebuilt with only
    ``path(y)`` blocks active (matching the training-time expansion
    shape for label ``y``), the model's ``p_y`` is read under that
    expansion, and probabilities are normalized across the candidate
    set.  Same trained weights as Variant A; only inference geometry
    differs.
    """
    feat_union = svm_classifier._feature_union
    svd = svm_classifier._svd
    expander = svm_classifier._expander
    calibrated = svm_classifier._pipeline
    classes = svm_classifier._classes

    X_tfidf = feat_union.transform([text])
    X_reduced = svd.transform(X_tfidf)

    raw_p_y: dict[str, float] = {}
    class_to_index = {c: i for i, c in enumerate(classes)}
    for y in classes:
        X_y = expander.expand_with_labels(X_reduced, [y])
        proba_row = calibrated.predict_proba(X_y)[0]
        raw_p_y[y] = float(proba_row[class_to_index[y]])

    total = sum(raw_p_y.values())
    if total <= 0:
        n = len(classes)
        return {y: 1.0 / n for y in classes}
    return {y: p / total for y, p in raw_p_y.items()}


# ────────────────────────────────────────────────────────────────────
# 4.  Variant C: Crammer-Singer joint training + per-class inference
# ────────────────────────────────────────────────────────────────────

def train_variant_c(texts: list[str], labels: list[str], category_set) -> dict:
    """Variant C: Crammer-Singer joint multi-class SVM on Kronecker features.

    Same TF-IDF → SVD → Kronecker-expansion pipeline as Variant A, but
    trains with sklearn's ``multi_class="crammer_singer"`` loss instead
    of one-vs-rest.  Joint training optimizes per-class weights against
    the structured multi-class margin, which is closer to the
    structured-output objective Choi et al. (2015) Eq. 5 specifies.
    """
    from collections import Counter

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC

    from atelier.classify.svm_classifier import HierarchicalFeatureExpander

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 6),
        max_features=50_000, sublinear_tf=True,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=50_000, sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    feature_union = FeatureUnion([("char", char_tfidf), ("word", word_tfidf)])
    X_tfidf = feature_union.fit_transform(texts)

    n_components = min(200, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    alphas = category_set.compute_nhsvm_alphas()
    expander = HierarchicalFeatureExpander.from_category_set(
        category_set, alphas, n_features_in=n_components,
    )
    X_expanded = expander.expand_with_labels(X_reduced, labels)

    # Crammer-Singer requires dense data.
    X_dense = X_expanded.toarray() if hasattr(X_expanded, "toarray") else X_expanded

    min_count = min(Counter(labels).values())
    svc = LinearSVC(
        C=1.0, max_iter=20_000,
        multi_class="crammer_singer",
        loss="hinge",
        dual=True,
        class_weight="balanced",
    )
    calibrated = CalibratedClassifierCV(
        svc, cv=min(5, min_count), method="sigmoid", ensemble=False,
    )
    calibrated.fit(X_dense, labels)

    return {
        "feature_union": feature_union,
        "svd": svd,
        "expander": expander,
        "calibrated": calibrated,
        "classes": list(calibrated.classes_),
    }


def predict_proba_per_class_c(joint_model: dict, text: str) -> dict[str, float]:
    """Variant C: joint-trained Crammer-Singer model with per-class inference."""
    feat_union = joint_model["feature_union"]
    svd = joint_model["svd"]
    expander = joint_model["expander"]
    calibrated = joint_model["calibrated"]
    classes = joint_model["classes"]

    X_tfidf = feat_union.transform([text])
    X_reduced = svd.transform(X_tfidf)

    raw_p_y: dict[str, float] = {}
    class_to_index = {c: i for i, c in enumerate(classes)}
    for y in classes:
        X_y = expander.expand_with_labels(X_reduced, [y])
        X_y_dense = X_y.toarray() if hasattr(X_y, "toarray") else X_y
        proba_row = calibrated.predict_proba(X_y_dense)[0]
        raw_p_y[y] = float(proba_row[class_to_index[y]])

    total = sum(raw_p_y.values())
    if total <= 0:
        n = len(classes)
        return {y: 1.0 / n for y in classes}
    return {y: p / total for y, p in raw_p_y.items()}


# ────────────────────────────────────────────────────────────────────
# 5.  Helpers
# ────────────────────────────────────────────────────────────────────

def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence on aligned distributions."""
    import math
    codes = set(p) | set(q)
    eps = 1e-12
    p_v = {c: p.get(c, 0.0) + eps for c in codes}
    q_v = {c: q.get(c, 0.0) + eps for c in codes}
    p_sum = sum(p_v.values()); q_sum = sum(q_v.values())
    p_v = {c: v / p_sum for c, v in p_v.items()}
    q_v = {c: v / q_sum for c, v in q_v.items()}
    m = {c: 0.5 * (p_v[c] + q_v[c]) for c in codes}
    def kl(a, b):
        return sum(a[c] * math.log(a[c] / b[c]) for c in codes if a[c] > 0)
    return 0.5 * kl(p_v, m) + 0.5 * kl(q_v, m)


def subtree_of(code: str) -> str:
    parts = code.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code


def is_correct(head: str, expected: str, tier: str, llm_vote_code: str) -> bool:
    """Tier-defined correctness.

    - semantic-conflict: parent should hold → correct iff head == llm vote
    - everything else: head should land on the expected leaf
    """
    if tier == "semantic-conflict":
        return head == llm_vote_code
    return head == expected


def fuse(
    llm_code: str,
    llm_conf: float,
    llm_disc: float,
    svm_proba: dict[str, float],
    svm_disc: float,
    fusion_strategy: str,
    frame,
    category_set,
) -> tuple[str, float]:
    """Single fusion: returns (head_code, top1_margin)."""
    from atelier.classify.belief import HierarchicalClassification
    from atelier.classify.mass_functions import llm_to_mass, svm_to_mass

    llm_mass = llm_to_mass(
        llm_code, llm_conf, alternatives=[], frame=frame, discount=llm_disc,
    )
    svm_mass = svm_to_mass(svm_proba, frame, discount=svm_disc)
    hc = HierarchicalClassification.from_combined_evidence(
        {"llm": llm_mass, "svm": svm_mass}, frame, category_set,
        fusion_strategy=fusion_strategy,
    )
    head = getattr(hc.category, "code", "?") if hc.category else "?"
    try:
        margin = float(hc.top1_margin())
    except Exception:
        margin = 1.0
    return head, margin


def find_inflection(
    param_values: list[float],
    accuracies: list[float],
    threshold: float = 0.5,
) -> str:
    """Direction-aware inflection-point detector.

    Returns a string of the form ``"<dir> <value>"`` where ``<dir>`` is
    ``↑`` for monotone-increasing curves (first param where accuracy
    rises to ≥ threshold) and ``↓`` for monotone-decreasing curves
    (first param where accuracy falls below threshold).  Sentinels:

      ``"flat"``    accuracy is identically ≥ threshold throughout the
                    sweep (no inflection — the variant just works here)
      ``"flat-lo"`` accuracy is identically < threshold throughout the
                    sweep (no inflection — variant never reaches it)
      ``"—"``       non-monotone in the swept range (curve folds back)
    """
    if not accuracies or not param_values:
        return "—"
    above = [a >= threshold for a in accuracies]
    if all(above):
        return "flat"
    if not any(above):
        return "flat-lo"

    a0, an = accuracies[0], accuracies[-1]
    direction = "↑" if an > a0 else "↓" if an < a0 else "↑"

    # Check monotonicity (with small slack for ties)
    increasing = all(b >= a - 1e-9 for a, b in zip(accuracies, accuracies[1:]))
    decreasing = all(b <= a + 1e-9 for a, b in zip(accuracies, accuracies[1:]))
    if not (increasing or decreasing):
        # Non-monotone — report the first crossing in whichever direction
        # the endpoints suggest
        pass

    if direction == "↑":
        for p, a in zip(param_values, accuracies):
            if a >= threshold:
                return f"↑ {p}"
    else:  # ↓
        for p, a in zip(param_values, accuracies):
            if a < threshold:
                return f"↓ {p}"
    return "—"


# ────────────────────────────────────────────────────────────────────
# 6.  Sweep driver
# ────────────────────────────────────────────────────────────────────

def run_sweep(
    test_cases,
    proba_cache,
    frame,
    category_set,
    *,
    param_name: str,
    param_values: list,
    base_params: dict,
) -> dict:
    """Vary one parameter, hold others; return ``{param_value: {tier|variant: accuracy}}``.

    ``base_params`` must include keys: ``svm_disc``, ``llm_conf``,
    ``llm_disc``, ``fusion``.  The varied parameter is overridden per
    ``param_values``.
    """
    results: dict = {}
    for pv in param_values:
        params = dict(base_params)
        params[param_name] = pv

        tier_variant_counts: dict[tuple[str, str], list[int]] = {}
        for query_text, expected, tier, llm_vote_code in test_cases:
            pa, pb, pc = proba_cache[query_text]
            for variant, proba in [("A", pa), ("B", pb), ("C", pc)]:
                head, _ = fuse(
                    llm_vote_code, params["llm_conf"], params["llm_disc"],
                    proba, params["svm_disc"], params["fusion"],
                    frame, category_set,
                )
                correct = is_correct(head, expected, tier, llm_vote_code)
                key = (tier, variant)
                counts = tier_variant_counts.setdefault(key, [0, 0])
                counts[0] += int(correct)
                counts[1] += 1

        results[pv] = {
            f"{t}|{v}": correct / total
            for (t, v), (correct, total) in tier_variant_counts.items()
        }
    return results


# ────────────────────────────────────────────────────────────────────
# 7.  Printing helpers
# ────────────────────────────────────────────────────────────────────

TIERS = ["easy", "hard", "contested", "sparse", "semantic-conflict", "svm-was-right"]
VARIANTS = ["A", "B", "C"]
SHORT_CODE = {
    "ROOT": "ROOT",
    "ROOT.SENSITIVE": "SENS",
    "ROOT.SENSITIVE.PII": "PII",
    "ROOT.SENSITIVE.PII.PERSON": "PERSON",
    "ROOT.SENSITIVE.PII.PERSON.NAME": "NAME",
    "ROOT.SENSITIVE.PII.PERSON.EMAIL": "EMAIL",
    "ROOT.SENSITIVE.PII.CONTACT": "CONTACT",
    "ROOT.SENSITIVE.PII.CONTACT.PHONE": "PHONE",
    "ROOT.SENSITIVE.PII.CONTACT.ADDR": "ADDR",
    "ROOT.SENSITIVE.PII.CONTACT.ADDR.SHIPPING": "ADDR.SHIP",
    "ROOT.SENSITIVE.PII.CONTACT.ADDR.BILLING": "ADDR.BILL",
    "ROOT.SENSITIVE.FIN": "FIN",
    "ROOT.SENSITIVE.FIN.CARD": "CARD",
    "ROOT.SENSITIVE.FIN.BANK": "BANK",
    "ROOT.OPERATIONAL": "OPER",
    "ROOT.OPERATIONAL.ID": "ID",
    "ROOT.OPERATIONAL.TIME": "TIME",
    "ROOT.OPERATIONAL.STATUS": "STATUS",
}


def short(code: str) -> str:
    return SHORT_CODE.get(code, code.replace("ROOT.", "")[:14])


def print_alpha_table(alphas: dict[str, float]) -> None:
    print("\nLP-solved alphas (directional-constrained):")
    for code in sorted(alphas):
        depth = code.count(".")
        print(f"  {'  ' * depth}{code:<48} α = {alphas[code]:.4f}")


def print_predict_proba_table(test_cases, proba_cache) -> None:
    print(f"{'tier':<18} {'Query':<32} {'A top-1':<12} {'A_p':<5} {'B top-1':<12} {'B_p':<5} "
          f"{'C top-1':<12} {'C_p':<5} {'JS(A,B)':<8} {'JS(A,C)':<8} {'agreement'}")
    print("─" * 145)
    for query_text, _expected, tier, _ in test_cases:
        pa, pb, pc = proba_cache[query_text]
        ta = max(pa, key=lambda c: pa[c]); tb = max(pb, key=lambda c: pb[c]); tc = max(pc, key=lambda c: pc[c])
        agreement = "all" if ta == tb == tc else (
            "A=B≠C" if ta == tb else "A=C≠B" if ta == tc else "B=C≠A" if tb == tc else "all-diff"
        )
        jab = js_divergence(pa, pb)
        jac = js_divergence(pa, pc)
        print(f"{tier:<18} {query_text[:30]:<32} {short(ta):<12} {pa[ta]:.2f}  "
              f"{short(tb):<12} {pb[tb]:.2f}  {short(tc):<12} {pc[tc]:.2f}  "
              f"{jab:.3f}    {jac:.3f}    {agreement}")


def aggregate_tier_accuracy(per_case_table: dict) -> dict:
    """{(tier, variant): accuracy} from {case_idx: {(tier, variant): bool}}."""
    out: dict[tuple[str, str], list[int]] = {}
    for case_results in per_case_table.values():
        for key, correct in case_results.items():
            counts = out.setdefault(key, [0, 0])
            counts[0] += int(correct)
            counts[1] += 1
    return {k: c / t for k, (c, t) in out.items()}


def print_default_fusion_table(default_per_tier_acc: dict[tuple[str, str], float]) -> None:
    print(f"{'tier':<22} {'A acc':<10} {'B acc':<10} {'C acc':<10}")
    print("─" * 55)
    for tier in TIERS:
        a = default_per_tier_acc.get((tier, "A"), 0.0)
        b = default_per_tier_acc.get((tier, "B"), 0.0)
        c = default_per_tier_acc.get((tier, "C"), 0.0)
        print(f"{tier:<22} {a:.2f}       {b:.2f}       {c:.2f}")


def print_sweep_table(sweep: dict, param_label: str) -> None:
    """Render sweep results as variants × tiers grid across parameter values."""
    param_values = sorted(sweep.keys())
    header = f"{'tier':<22} {'var':<3} {param_label:>6} → " + " ".join(f"{p:>5}" for p in param_values)
    print(header)
    print("─" * len(header))
    for tier in TIERS:
        for variant in VARIANTS:
            key = f"{tier}|{variant}"
            cells = [sweep[pv].get(key, 0.0) for pv in param_values]
            cells_s = " ".join(f"{c:>5.2f}" for c in cells)
            print(f"{tier:<22} {variant:<3} {'':>6}   {cells_s}")
        print()


def print_inflection_table(inflections: dict) -> None:
    """One inflection-point row per (tier, variant) pair, across all sweeps."""
    sweep_names = list(inflections.keys())
    header = f"{'tier':<22} {'var':<3} " + " ".join(f"{s:>14}" for s in sweep_names)
    print(header)
    print("─" * len(header))
    for tier in TIERS:
        for variant in VARIANTS:
            cells = []
            for sweep_name in sweep_names:
                v = inflections[sweep_name].get(f"{tier}|{variant}", "—")
                cells.append(f"{v!s:>14}")
            print(f"{tier:<22} {variant:<3} {' '.join(cells)}")
        print()


def print_fusion_strategy_table(strat_results: dict) -> None:
    """Side-by-side dempster vs yager."""
    print(f"{'tier':<22} {'var':<3} {'dempster':<10} {'yager':<10}")
    print("─" * 50)
    for tier in TIERS:
        for variant in VARIANTS:
            d = strat_results["dempster"].get(f"{tier}|{variant}", 0.0)
            y = strat_results["yager"].get(f"{tier}|{variant}", 0.0)
            print(f"{tier:<22} {variant:<3} {d:.2f}       {y:.2f}")
        print()


# ────────────────────────────────────────────────────────────────────
# 8.  Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    import json

    from atelier.classify.mass_functions import FrameOfDiscernment
    from atelier.classify.svm_classifier import SVMClassifier

    # ── Setup ──
    category_set = build_adversarial_taxonomy()
    texts, labels = generate_training_data(n_per_leaf=30, seed=42)
    test_cases = build_adversarial_test_set()
    alphas = category_set.compute_nhsvm_alphas()
    frame = FrameOfDiscernment(category_set, confusable_pairs=[])

    print(f"Taxonomy: {len(category_set.categories)} nodes "
          f"(subtrees: {sorted({subtree_of(c.code) for c in category_set.categories if subtree_of(c.code) != c.code})})")
    print(f"Training: {len(texts)} samples across {len(set(labels))} classes")
    print(f"Test set: {len(test_cases)} cases across "
          f"{len({t[2] for t in test_cases})} tiers")
    print_alpha_table(alphas)

    # ── Train variants ──
    print("\n─── Training variants ───")
    model_a = SVMClassifier(category_set=category_set, hierarchical=True)
    model_a.fit(texts, labels)
    print("  Variant A (ovr-universal): trained")
    model_c = train_variant_c(texts, labels, category_set)
    print("  Variant C (joint-per-class): trained")
    print("  Variant B (ovr-per-class): inference path on A's model — no separate training")

    # ── Cache predict_probas across variants ──
    proba_cache: dict[str, tuple[dict, dict, dict]] = {}
    for query_text, *_ in test_cases:
        pa = model_a.predict_proba_single(query_text)
        pb = predict_proba_per_class(model_a, query_text)
        pc = predict_proba_per_class_c(model_c, query_text)
        proba_cache[query_text] = (pa, pb, pc)

    # ── Per-case predict_proba comparison ──
    print("\n─── predict_proba: top-1 per variant + pairwise JS divergence ───\n")
    print_predict_proba_table(test_cases, proba_cache)

    # ── Default-parameter fusion accuracy ──
    LLM_CONF_DEFAULT = 0.85
    LLM_DISC_DEFAULT = 0.15
    SVM_DISC_DEFAULT = 0.22
    FUSION_DEFAULT = "dempster"

    default_per_tier: dict[tuple[str, str], list[int]] = {}
    for query_text, expected, tier, llm_vote_code in test_cases:
        pa, pb, pc = proba_cache[query_text]
        for variant, proba in [("A", pa), ("B", pb), ("C", pc)]:
            head, _ = fuse(
                llm_vote_code, LLM_CONF_DEFAULT, LLM_DISC_DEFAULT,
                proba, SVM_DISC_DEFAULT, FUSION_DEFAULT,
                frame, category_set,
            )
            ok = is_correct(head, expected, tier, llm_vote_code)
            counts = default_per_tier.setdefault((tier, variant), [0, 0])
            counts[0] += int(ok)
            counts[1] += 1
    default_acc = {k: c / t for k, (c, t) in default_per_tier.items()}

    print(
        f"\n─── Default-parameter fusion accuracy ───"
        f"  (SVM disc={SVM_DISC_DEFAULT}, LLM conf={LLM_CONF_DEFAULT}, "
        f"LLM disc={LLM_DISC_DEFAULT}, fusion={FUSION_DEFAULT})\n"
    )
    print_default_fusion_table(default_acc)

    # ── Sweeps ──
    base = {
        "svm_disc": SVM_DISC_DEFAULT,
        "llm_conf": LLM_CONF_DEFAULT,
        "llm_disc": LLM_DISC_DEFAULT,
        "fusion": FUSION_DEFAULT,
    }
    sweep_results: dict = {}

    # Sweep 1: SVM mass discount
    svm_disc_values = [round(0.10 + 0.05 * i, 2) for i in range(9)]
    sweep_results["svm_discount"] = run_sweep(
        test_cases, proba_cache, frame, category_set,
        param_name="svm_disc", param_values=svm_disc_values, base_params=base,
    )
    print("\n─── Sweep: SVM mass discount ───\n")
    print_sweep_table(sweep_results["svm_discount"], param_label="SVM_disc")

    # Sweep 2: Runtime LLM confidence
    llm_conf_values = [round(0.40 + 0.05 * i, 2) for i in range(12)]
    sweep_results["llm_confidence"] = run_sweep(
        test_cases, proba_cache, frame, category_set,
        param_name="llm_conf", param_values=llm_conf_values, base_params=base,
    )
    print("\n─── Sweep: Runtime LLM confidence ───\n")
    print_sweep_table(sweep_results["llm_confidence"], param_label="LLM_conf")

    # Sweep 3: Runtime LLM mass discount
    llm_disc_values = [round(0.05 + 0.05 * i, 2) for i in range(6)]  # 0.05..0.30
    sweep_results["llm_discount"] = run_sweep(
        test_cases, proba_cache, frame, category_set,
        param_name="llm_disc", param_values=llm_disc_values, base_params=base,
    )
    print("\n─── Sweep: Runtime LLM mass discount ───\n")
    print_sweep_table(sweep_results["llm_discount"], param_label="LLM_disc")

    # Sweep 4: Fusion strategy
    strat_results: dict = {}
    for strat in ["dempster", "yager"]:
        tier_variant_counts: dict[tuple[str, str], list[int]] = {}
        for query_text, expected, tier, llm_vote_code in test_cases:
            pa, pb, pc = proba_cache[query_text]
            for variant, proba in [("A", pa), ("B", pb), ("C", pc)]:
                head, _ = fuse(
                    llm_vote_code, LLM_CONF_DEFAULT, LLM_DISC_DEFAULT,
                    proba, SVM_DISC_DEFAULT, strat, frame, category_set,
                )
                ok = is_correct(head, expected, tier, llm_vote_code)
                counts = tier_variant_counts.setdefault((tier, variant), [0, 0])
                counts[0] += int(ok)
                counts[1] += 1
        strat_results[strat] = {
            f"{t}|{v}": c / total
            for (t, v), (c, total) in tier_variant_counts.items()
        }
    sweep_results["fusion_strategy"] = strat_results

    print("\n─── Comparison: Fusion strategy (dempster vs yager) ───\n")
    print_fusion_strategy_table(strat_results)

    # ── Inflection-point summary ──
    print("\n─── Inflection-point summary ───")
    print("Direction-aware: ↑ <value> = increasing accuracy, first param at ≥ 0.5;")
    print("                  ↓ <value> = decreasing accuracy, first param at < 0.5.")
    print("Sentinels: 'flat' = ≥ 0.5 throughout; 'flat-lo' = < 0.5 throughout;")
    print("           '—' = non-monotone or never crosses.\n")
    inflections: dict[str, dict[str, Any]] = {}
    for sweep_name, sweep_data in sweep_results.items():
        if sweep_name == "fusion_strategy":
            continue  # not a numerical sweep
        sweep_inflections: dict[str, Any] = {}
        param_values_sorted = sorted(sweep_data.keys())
        for tier in TIERS:
            for variant in VARIANTS:
                key = f"{tier}|{variant}"
                accs = [sweep_data[pv].get(key, 0.0) for pv in param_values_sorted]
                sweep_inflections[key] = find_inflection(param_values_sorted, accs)
        inflections[sweep_name] = sweep_inflections
    print_inflection_table(inflections)

    # ── JSON dump ──
    out_path = Path("build/audit/nhsvm/sensitivity_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "alphas": {c: float(a) for c, a in alphas.items()},
        "defaults": {
            "svm_discount": SVM_DISC_DEFAULT,
            "llm_confidence": LLM_CONF_DEFAULT,
            "llm_discount": LLM_DISC_DEFAULT,
            "fusion_strategy": FUSION_DEFAULT,
        },
        "default_accuracy": {
            f"{t}|{v}": acc for (t, v), acc in default_acc.items()
        },
        "sweeps": {
            sweep_name: (
                sweep_data if sweep_name == "fusion_strategy"
                else {f"{pv:.2f}": data for pv, data in sweep_data.items()}
            )
            for sweep_name, sweep_data in sweep_results.items()
        },
        "inflections": inflections,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nRaw sweep data: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
