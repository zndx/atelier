"""Unit tests for the subsumption alignment module.

Tests the core cosine-argmax alignment logic, ICE concept signature
generation, cache behavior, and integration with the
``build_alignment`` entry point in ``ontology_alignment.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_category_set():
    """Minimal HierarchicalCategorySet with a few leaf codes.

    Both ``leaf_codes`` and ``all_by_code`` are populated so the
    hierarchical-integrity enumerator (``_enumerate_all_codes``) sees
    the same set as legacy ``leaf_codes`` consumers.  For tests that
    need the leaf-vs-internal distinction, use
    ``mock_hierarchical_category_set`` instead.
    """
    cs = MagicMock()
    leaves = {
        "ICE.SENSITIVE.PID.CONTACT.EMAIL",
        "ICE.SENSITIVE.PID.CONTACT.PHONE",
        "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN",
        "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN",
        "ICE.SENSITIVE.PID.NAME.FULLNAME",
    }
    cs.leaf_codes = leaves
    cs.all_by_code = {code: MagicMock(code=code, label=code, description="")
                      for code in leaves}
    cs.ancestors = lambda code: []
    return cs


@pytest.fixture
def mock_hierarchical_category_set():
    """Vocabulary with explicit leaves + internal nodes for hierarchical tests.

    Structure:
        FAMILY.FIN              (internal — Financial family)
          FAMILY.FIN.PAYMENT    (internal — Payment subfamily)
            FAMILY.FIN.PAYMENT.CARDNUM  (leaf — Card Number specifically)
            FAMILY.FIN.PAYMENT.CVV       (leaf — CVV specifically)
        FAMILY.PII              (internal — PII family, no leaves under it
                                 in this fixture; some ICE concepts may
                                 align to it directly when the user vocab
                                 has the family but not a specific leaf)
        FAMILY.OTHER            (leaf — standalone)

    This gives us:
      - Leaves available for tight-alignment
      - Internal nodes available as fallback-alignment targets
      - At least one parent-family node (FAMILY.PII) we expect ICE to
        align to when the user vocab lacks a leaf-level equivalent
    """
    cs = MagicMock()
    cs.leaf_codes = frozenset({
        "FAMILY.FIN.PAYMENT.CARDNUM",
        "FAMILY.FIN.PAYMENT.CVV",
        "FAMILY.OTHER",
    })
    all_codes = {
        "FAMILY.FIN",
        "FAMILY.FIN.PAYMENT",
        "FAMILY.FIN.PAYMENT.CARDNUM",
        "FAMILY.FIN.PAYMENT.CVV",
        "FAMILY.PII",
        "FAMILY.OTHER",
    }

    def _mk(code: str, label: str, description: str = "") -> MagicMock:
        m = MagicMock()
        m.code = code
        m.label = label
        m.description = description
        return m

    cs.all_by_code = {
        "FAMILY.FIN": _mk("FAMILY.FIN", "Financial Data"),
        "FAMILY.FIN.PAYMENT": _mk("FAMILY.FIN.PAYMENT", "Payment Data"),
        "FAMILY.FIN.PAYMENT.CARDNUM": _mk(
            "FAMILY.FIN.PAYMENT.CARDNUM", "Card Number",
        ),
        "FAMILY.FIN.PAYMENT.CVV": _mk(
            "FAMILY.FIN.PAYMENT.CVV", "Card Verification Value",
        ),
        "FAMILY.PII": _mk("FAMILY.PII", "Personally Identifiable Data"),
        "FAMILY.OTHER": _mk("FAMILY.OTHER", "Other Data"),
    }
    parent_chain = {
        "FAMILY.FIN.PAYMENT.CARDNUM": ["FAMILY.FIN.PAYMENT", "FAMILY.FIN"],
        "FAMILY.FIN.PAYMENT.CVV": ["FAMILY.FIN.PAYMENT", "FAMILY.FIN"],
        "FAMILY.FIN.PAYMENT": ["FAMILY.FIN"],
        "FAMILY.FIN": [],
        "FAMILY.PII": [],
        "FAMILY.OTHER": [],
    }
    cs.ancestors = lambda code: parent_chain.get(code, [])
    return cs


@pytest.fixture
def mock_hierarchical_enrichments():
    """Enriched signatures keyed by user code, covering leaves AND internals.

    Note: FAMILY.OTHER is intentionally omitted to exercise the
    out-of-vocabulary handling (ICE concepts whose best match falls
    below threshold get no alignment entry).
    """
    return {
        "FAMILY.FIN": "Financial Data | broad financial-domain values | "
                     "values: $1,234.56, USD 500, €99.99 | "
                     "names: amount, balance, total, transaction_value",
        "FAMILY.FIN.PAYMENT": "Payment Data | payment-instrument context | "
                              "values: payment_record, tx_id_42 | "
                              "names: payment_method, pay_type",
        "FAMILY.FIN.PAYMENT.CARDNUM": "Card Number | primary account number | "
                                      "values: 4111111111111111, 5500000000000004 | "
                                      "names: card_number, pan, cc_num",
        "FAMILY.FIN.PAYMENT.CVV": "Card Verification Value | 3-4 digit CVV | "
                                  "values: 123, 456, 7890 | "
                                  "names: cvv, cvv2, cvc, card_verification",
        "FAMILY.PII": "Personally Identifiable Data | broad personal info | "
                      "values: John Smith, jane.doe@example.com, 555-1234 | "
                      "names: full_name, contact, personal_data",
        "FAMILY.OTHER": "Other Data | catch-all category | "
                        "values: misc, n/a, unknown | "
                        "names: other, misc, etc",
    }


@pytest.fixture
def mock_enriched_payloads():
    """Qdrant scroll response simulating enriched annotations."""
    return [
        {
            "code": "ICE.SENSITIVE.PID.CONTACT.EMAIL",
            "label": "Email Address",
            "description": "Personal or business email address",
            "prototype_values": ["john@example.com", "jane.doe@corp.org", "user123@gmail.com"],
            "name_hints": ["email", "email_address", "contact_email", "e_mail"],
        },
        {
            "code": "ICE.SENSITIVE.PID.CONTACT.PHONE",
            "label": "Phone Number",
            "description": "Telephone number including mobile and landline",
            "prototype_values": ["+1-555-0123", "(212) 555-4567", "1-800-555-0199"],
            "name_hints": ["phone", "phone_number", "tel", "mobile"],
        },
        {
            "code": "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN",
            "label": "Social Security Number",
            "description": "US Social Security Number (9-digit identifier)",
            "prototype_values": ["123-45-6789", "987-65-4321", "555-12-3456"],
            "name_hints": ["ssn", "social_security", "ss_number", "ssn_last4"],
        },
        {
            "code": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN",
            "label": "Payment Card Number",
            "description": "Credit or debit card primary account number",
            "prototype_values": ["4111111111111111", "5500000000000004", "378282246310005"],
            "name_hints": ["card_number", "pan", "credit_card", "cc_number"],
        },
        {
            "code": "ICE.SENSITIVE.PID.NAME.FULLNAME",
            "label": "Full Name",
            "description": "Complete personal name including first and last",
            "prototype_values": ["John Smith", "Maria Garcia", "James Wilson"],
            "name_hints": ["full_name", "name", "customer_name", "person_name"],
        },
    ]


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for alignment cache."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── ICE concept signature tests ─────────────────────────────────────


class TestIceToHumanizedName:
    def test_standard_deep_code(self):
        from atelier.classify.subsumption_alignment import _ice_to_humanized_name
        result = _ice_to_humanized_name("ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN")
        assert result == "ssn (pid identity govid)"

    def test_shallow_code(self):
        from atelier.classify.subsumption_alignment import _ice_to_humanized_name
        result = _ice_to_humanized_name("ICE.SENSITIVE.EMAIL")
        assert result == "email"

    def test_minimal_code(self):
        from atelier.classify.subsumption_alignment import _ice_to_humanized_name
        # With < 3 parts, tail = all parts → leaf = last, context = preceding
        result = _ice_to_humanized_name("A.B")
        assert result == "b (a)"


class TestBuildIceConceptSignatures:
    def test_produces_signatures_for_available_generators(self):
        from atelier.classify.subsumption_alignment import _build_ice_concept_signatures
        sigs = _build_ice_concept_signatures()
        # Should have signatures for ICE codes with working generators
        assert len(sigs) > 0
        # Each signature should contain the pipe separator
        for code, sig in sigs.items():
            assert "|" in sig, f"Signature for {code} missing pipe separator"
            assert code.startswith("ICE."), f"Code {code} doesn't start with ICE."

    def test_signatures_are_deterministic(self):
        from atelier.classify.subsumption_alignment import _build_ice_concept_signatures
        sigs1 = _build_ice_concept_signatures()
        sigs2 = _build_ice_concept_signatures()
        assert sigs1 == sigs2


# ── Cosine alignment tests ──────────────────────────────────────────


class TestCosineArgmaxAlignment:
    """``_cosine_argmax_alignment`` now returns ``(alignment, score_detail)``.

    The score_detail dict carries the top-1/top-2 user codes + scores
    per ICE concept for downstream near-tie analysis in the validator.
    """

    def test_identical_texts_align_perfectly(self):
        """When ICE and user have identical text, alignment should be 1:1."""
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment

        ice_sigs = {
            "ICE.A": "email address | john@example.com, user@test.org",
            "ICE.B": "phone number | 555-0123, 212-555-4567",
        }
        user_sigs = {
            "USER.EMAIL": "email address | john@example.com, user@test.org",
            "USER.PHONE": "phone number | 555-0123, 212-555-4567",
        }

        alignment, score_detail = _cosine_argmax_alignment(ice_sigs, user_sigs, 0.5)
        assert alignment.get("ICE.A") == "USER.EMAIL"
        assert alignment.get("ICE.B") == "USER.PHONE"
        # Score detail is populated for every ICE input
        assert set(score_detail.keys()) == {"ICE.A", "ICE.B"}
        assert score_detail["ICE.A"]["top1_user"] == "USER.EMAIL"

    def test_threshold_gates_low_similarity(self):
        """ICE concepts with no good match should not appear in alignment."""
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment

        ice_sigs = {
            "ICE.OBSCURE": "xyzzy plugh nothing matches this at all",
        }
        user_sigs = {
            "USER.EMAIL": "email address electronic mail contact info",
            "USER.PHONE": "telephone mobile phone number calling",
        }

        alignment, score_detail = _cosine_argmax_alignment(ice_sigs, user_sigs, 0.8)
        # With a high threshold, the obscure ICE code shouldn't match
        assert "ICE.OBSCURE" not in alignment
        # But its score detail IS recorded (top1 user is still captured,
        # even when below threshold — the diagnostic info is preserved)
        assert "ICE.OBSCURE" in score_detail

    def test_empty_inputs(self):
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment
        assert _cosine_argmax_alignment({}, {"A": "text"}, 0.5) == ({}, {})
        assert _cosine_argmax_alignment({"A": "text"}, {}, 0.5) == ({}, {})


# ── Cache behavior tests ────────────────────────────────────────────


class TestAlignmentCache:
    def test_cache_key_stability(self):
        from atelier.classify.subsumption_alignment import _alignment_cache_key
        key1 = _alignment_cache_key(["A", "B"], ["X", "Y"], "model1", "method1")
        key2 = _alignment_cache_key(["B", "A"], ["Y", "X"], "model1", "method1")
        assert key1 == key2, "Cache key should be stable regardless of input order"

    def test_cache_key_sensitivity(self):
        from atelier.classify.subsumption_alignment import _alignment_cache_key
        key1 = _alignment_cache_key(["A", "B"], ["X", "Y"], "model1", "method1")
        key2 = _alignment_cache_key(["A", "B"], ["X", "Y"], "model2", "method1")
        key3 = _alignment_cache_key(["A", "B"], ["X", "Y"], "model1", "method2")
        assert key1 != key2, "Different models should produce different keys"
        assert key1 != key3, "Different methods should produce different keys"

    def test_cache_roundtrip(self, temp_cache_dir, mock_category_set):
        """Build alignment writes cache; second call reads from cache."""
        from atelier.classify.subsumption_alignment import (
            _alignment_cache_key,
            _METHOD_TAG,
        )

        # Pre-populate a cache file
        ice_leaves = sorted(["ICE.A", "ICE.B"])
        user_codes = sorted(mock_category_set.leaf_codes)
        key = _alignment_cache_key(
            ice_leaves, user_codes, "all-MiniLM-L6-v2", _METHOD_TAG,
        )
        cache_path = temp_cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps({
            "method": _METHOD_TAG,
            "alignment": {"ICE.A": "ICE.SENSITIVE.PID.CONTACT.EMAIL"},
            "ice_leaf_count": 2,
            "user_code_count": len(user_codes),
            "embedding_model": "all-MiniLM-L6-v2",
            "score_threshold": 0.35,
            "mapped": 1,
        }) + "\n")

        # Now try loading from cache via build_alignment_via_subsumption
        from atelier.classify.subsumption_alignment import build_alignment_via_subsumption

        # Mock the generators to only return ICE.A and ICE.B
        mock_generators = {
            "ICE.A": lambda rng: "test@example.com",
            "ICE.B": lambda rng: "555-0123",
        }
        # GENERATORS is imported function-locally from synth_generators
        # inside _build_ice_signatures, so patch it at the source module.
        with patch(
            "atelier.classify.synth_generators.GENERATORS",
            mock_generators,
        ):
            # Patch to avoid Qdrant lookup
            result = build_alignment_via_subsumption(
                mock_category_set,
                cache_dir=temp_cache_dir,
            )

        assert result == {"ICE.A": "ICE.SENSITIVE.PID.CONTACT.EMAIL"}


# ── Integration with ontology_alignment.build_alignment ─────────────


class TestBuildAlignmentDelegation:
    def test_build_alignment_delegates_to_subsumption(self, temp_cache_dir):
        """The top-level build_alignment should delegate to subsumption."""
        from atelier.classify.ontology_alignment import build_alignment

        cs = MagicMock()
        cs.leaf_codes = {"CODE.A", "CODE.B"}

        with patch(
            "atelier.classify.subsumption_alignment.build_alignment_via_subsumption",
            return_value={"ICE.X": "CODE.A"},
        ) as mock_sub:
            result = build_alignment(
                cs,
                llm_backend=MagicMock(),  # should be ignored
                system_prompt="ignored",
                model_name="ignored",
                cache_dir=temp_cache_dir,
            )

        assert result == {"ICE.X": "CODE.A"}
        mock_sub.assert_called_once()

    def test_build_alignment_passes_cfg(self, temp_cache_dir):
        """cfg parameter is forwarded to subsumption alignment."""
        from atelier.classify.ontology_alignment import build_alignment

        cs = MagicMock()
        cs.leaf_codes = {"CODE.A"}
        cfg = MagicMock()
        cfg.classify_subsumption_score_threshold = 0.45
        cfg.classify_embedding_model = "custom-model"

        with patch(
            "atelier.classify.subsumption_alignment.build_alignment_via_subsumption",
            return_value={},
        ) as mock_sub:
            build_alignment(cs, cfg=cfg, cache_dir=temp_cache_dir)

        call_kwargs = mock_sub.call_args
        assert call_kwargs.kwargs["score_threshold"] == 0.45
        assert call_kwargs.kwargs["embedding_model"] == "custom-model"


# ── Validation harness tests ────────────────────────────────────────


# ── Hierarchical integrity (P7-R2 / R3) ─────────────────────────────


class TestHierarchicalIntegrity:
    """All-nodes alignment, many-to-one signal, out-of-vocab dropout."""

    def test_enumerate_all_codes_returns_leaves_and_internals(
        self, mock_hierarchical_category_set,
    ):
        from atelier.classify.subsumption_alignment import _enumerate_all_codes
        codes = _enumerate_all_codes(mock_hierarchical_category_set)
        # All 6 nodes — leaves AND internals — must be present
        assert codes == frozenset({
            "FAMILY.FIN", "FAMILY.FIN.PAYMENT",
            "FAMILY.FIN.PAYMENT.CARDNUM", "FAMILY.FIN.PAYMENT.CVV",
            "FAMILY.PII", "FAMILY.OTHER",
        })
        # Specifically, the fix vs the legacy leaf_codes regression
        assert codes != mock_hierarchical_category_set.leaf_codes
        assert "FAMILY.PII" in codes  # internal node IS a target
        assert "FAMILY.FIN" in codes  # internal node IS a target

    def test_enumerate_all_codes_falls_back_to_leaves_only(self):
        """A plain CategorySet without all_by_code falls back to .categories."""
        from atelier.classify.subsumption_alignment import _enumerate_all_codes
        cs = MagicMock(spec=[])  # no all_by_code attribute
        cs.categories = [
            MagicMock(code="A"),
            MagicMock(code="B"),
        ]
        assert _enumerate_all_codes(cs) == frozenset({"A", "B"})

    def test_classify_node_kind_distinguishes_leaf_vs_internal(
        self, mock_hierarchical_category_set,
    ):
        from atelier.classify.subsumption_alignment import _classify_node_kind
        # Leaves
        assert _classify_node_kind(
            mock_hierarchical_category_set, "FAMILY.FIN.PAYMENT.CARDNUM",
        ) == "leaf"
        assert _classify_node_kind(
            mock_hierarchical_category_set, "FAMILY.OTHER",
        ) == "leaf"
        # Internals
        assert _classify_node_kind(
            mock_hierarchical_category_set, "FAMILY.FIN",
        ) == "internal"
        assert _classify_node_kind(
            mock_hierarchical_category_set, "FAMILY.PII",
        ) == "internal"

    def test_alignment_can_target_internal_nodes(
        self, mock_hierarchical_category_set, mock_hierarchical_enrichments,
    ):
        """ICE concepts must be allowed to align to user internal nodes.

        This is the load-bearing hierarchical-integrity assertion: the
        matcher's argmax over the full user code set may legitimately
        land on an internal node when an ICE leaf's best match is a
        user family-level concept rather than a leaf-level equivalent.
        """
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment
        # Two ICE concepts:
        #   "personally identifiable contact ..." → should align to FAMILY.PII (internal)
        #   "card number for payment" → should align to FAMILY.FIN.PAYMENT.CARDNUM (leaf)
        ice_sigs = {
            "ICE.CONTACT.GENERIC": "personally identifiable contact info "
                                   "personal details broad PII",
            "ICE.CARD.NUMBER": "primary account number card_number pan "
                               "credit card number 16 digits",
        }
        alignment = _cosine_argmax_alignment(
            ice_sigs, mock_hierarchical_enrichments, 0.10,
        )
        if isinstance(alignment, tuple):
            # signature: (alignment, score_detail)
            alignment, _ = alignment

        # At least one alignment should land on an internal node
        from atelier.classify.subsumption_alignment import _classify_node_kind
        internal_targets = [
            u for u in alignment.values()
            if _classify_node_kind(mock_hierarchical_category_set, u) == "internal"
        ]
        assert internal_targets, (
            f"Expected at least one alignment to target an internal node, "
            f"got alignment={alignment}"
        )

    def test_alignment_values_are_subset_of_all_user_codes(
        self, mock_hierarchical_category_set, mock_hierarchical_enrichments,
    ):
        """Architectural invariant: SVM must never emit non-user codes.

        Every value in the alignment dict must be a member of the user
        vocabulary's all-nodes code set.  Violation = the matcher is
        emitting alignments to codes that don't exist in the user
        frame, which would silently corrupt downstream training.
        """
        from atelier.classify.subsumption_alignment import (
            _cosine_argmax_alignment,
            _enumerate_all_codes,
        )
        ice_sigs = {
            "ICE.A": "Card Number primary account number 16 digit",
            "ICE.B": "Card Verification Value CVV 3-4 digit",
            "ICE.C": "Personally identifiable data person info",
        }
        result = _cosine_argmax_alignment(
            ice_sigs, mock_hierarchical_enrichments, 0.10,
        )
        if isinstance(result, tuple):
            alignment, _ = result
        else:
            alignment = result

        all_user_codes = _enumerate_all_codes(mock_hierarchical_category_set)
        # The mock_hierarchical_enrichments fixture covers all user codes,
        # so alignment values must be drawn from that set
        for ice_code, user_code in alignment.items():
            assert user_code in all_user_codes, (
                f"Alignment {ice_code} -> {user_code} emits a code NOT in "
                f"the user vocabulary's all-nodes set. Architectural "
                f"invariant violation: SVM must never emit non-user codes."
            )

    def test_many_to_one_signal_aggregation(
        self, mock_hierarchical_category_set, mock_hierarchical_enrichments,
    ):
        """Multiple ICE concepts may align to a single user code.

        This is signal aggregation, not noise: multiple ICE generators
        contributing training rows to the same user code strengthens
        that user code's decision boundary.  The alignment dict must
        permit many-to-one (ICE → user) without dedupe.
        """
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment
        # Three ICE concepts all card-payment-flavored
        ice_sigs = {
            "ICE.CARD.PAN":   "primary account number PAN 16 digit credit card",
            "ICE.CARD.LAST4": "last 4 digits of card number masked PAN",
            "ICE.CARD.EXP":   "card expiration date MM/YY card_exp",
        }
        # Only put one user-vocab card-related target (CARDNUM) — all
        # three ICE concepts must land on the same user code
        result = _cosine_argmax_alignment(
            ice_sigs,
            {"FAMILY.FIN.PAYMENT.CARDNUM": mock_hierarchical_enrichments[
                "FAMILY.FIN.PAYMENT.CARDNUM"]},
            0.10,
        )
        if isinstance(result, tuple):
            alignment, _ = result
        else:
            alignment = result

        # All three ICE codes should map to the single user code
        assert len(alignment) == 3
        assert all(u == "FAMILY.FIN.PAYMENT.CARDNUM" for u in alignment.values())

    def test_out_of_vocabulary_ice_concepts_drop_silently(
        self, mock_hierarchical_category_set, mock_hierarchical_enrichments,
    ):
        """ICE concepts with no good user-code match are absent from alignment.

        The ICE corpus will outgrow user vocabularies over time. ICE
        concepts whose semantics don't appear in the user vocabulary
        must simply not enter the alignment — they contribute no
        training rows to the per-vocab SVM, which is correct.
        """
        from atelier.classify.subsumption_alignment import _cosine_argmax_alignment
        ice_sigs = {
            "ICE.HEIGHT.ALTITUDE": "barometric altitude flight ceiling "
                                   "atmospheric pressure aircraft mountain",
        }
        # Use a high threshold so this clearly-unrelated ICE concept
        # has no alignment in the financial-and-PII enrichments
        result = _cosine_argmax_alignment(
            ice_sigs, mock_hierarchical_enrichments, 0.85,
        )
        if isinstance(result, tuple):
            alignment, _ = result
        else:
            alignment = result

        assert "ICE.HEIGHT.ALTITUDE" not in alignment, (
            "ICE concept with no semantic overlap to user vocab should NOT "
            "appear in alignment"
        )


# ── Coverage metrics (P7-R3) ─────────────────────────────────────────


class TestCoverageMetrics:
    """Verify the four coverage metrics surfaced for operator review."""

    def test_coverage_user_code_perspective(
        self, mock_hierarchical_category_set,
    ):
        """user_code_coverage = fraction of user codes with ≥1 ICE alignment."""
        from atelier.classify.alignment_validator import _compute_coverage
        # 3 ICE codes aligning to 2 distinct user codes (one user code
        # gets 2 ICE concepts — the many-to-one case)
        alignment = {
            "ICE.A": "FAMILY.FIN.PAYMENT.CARDNUM",
            "ICE.B": "FAMILY.FIN.PAYMENT.CARDNUM",  # shares user code with A
            "ICE.C": "FAMILY.PII",                    # internal-node target
        }
        cov = _compute_coverage(
            alignment,
            category_set=mock_hierarchical_category_set,
            ice_total=5,  # 2 ICE concepts had no user-code match
        )
        assert cov.ice_total == 5
        assert cov.ice_mapped == 3
        assert cov.ice_coverage_pct == 60.0
        # 6 user codes in the hierarchy; 2 of them are covered
        assert cov.user_codes_total == 6
        assert cov.user_codes_covered == 2
        # Target kind distribution: 2 leaf alignments, 1 internal
        assert cov.target_kind_distribution["leaf"] == 2
        assert cov.target_kind_distribution["internal"] == 1
        # Many-to-one histogram: 1 user code has 2 ICE, 1 user code
        # has 1 ICE, 4 user codes have 0 ICE
        assert cov.many_to_one_histogram["0"] == 4
        assert cov.many_to_one_histogram["1"] == 1
        assert cov.many_to_one_histogram["2"] == 1

    def test_coverage_near_tie_rate(
        self, mock_hierarchical_category_set,
    ):
        """near_tie_rate computes from score_detail top1 - top2 differences."""
        from atelier.classify.alignment_validator import _compute_coverage
        alignment = {"ICE.A": "FAMILY.FIN", "ICE.B": "FAMILY.FIN.PAYMENT"}
        score_detail = {
            # ICE.A: clear winner — top1 0.80, top2 0.40 (diff 0.40 > 0.05)
            "ICE.A": {"top1_user": "FAMILY.FIN", "top1_score": 0.80,
                      "top2_user": "FAMILY.PII", "top2_score": 0.40},
            # ICE.B: near tie — top1 0.72, top2 0.71 (diff 0.01 < 0.05)
            "ICE.B": {"top1_user": "FAMILY.FIN.PAYMENT", "top1_score": 0.72,
                      "top2_user": "FAMILY.FIN", "top2_score": 0.71},
        }
        cov = _compute_coverage(
            alignment,
            category_set=mock_hierarchical_category_set,
            ice_total=2,
            score_detail=score_detail,
        )
        # 1 of 2 ICE concepts had near-tie → 50%
        assert cov.near_tie_rate == 0.5

    def test_coverage_legacy_method_no_near_tie(
        self, mock_hierarchical_category_set,
    ):
        """Without score_detail, near_tie_rate is None (legacy LLM method)."""
        from atelier.classify.alignment_validator import _compute_coverage
        cov = _compute_coverage(
            {"ICE.A": "FAMILY.OTHER"},
            category_set=mock_hierarchical_category_set,
            ice_total=1,
            score_detail=None,
        )
        assert cov.near_tie_rate is None


# ── Validation harness tests ────────────────────────────────────────


class TestAlignmentValidator:
    def test_should_validate_returns_true_for_new_combo(self, temp_cache_dir):
        from atelier.classify.alignment_validator import should_validate
        assert should_validate(temp_cache_dir, "abc123", "subsumption_st_v1")

    def test_should_validate_returns_false_after_report(self, temp_cache_dir):
        from atelier.classify.alignment_validator import should_validate

        # Create a fake report file
        report_file = temp_cache_dir / "abc123-subsumption_st_v1-all-MiniLM-L6-v2-v1-2026-05-18.json"
        report_file.write_text("{}")

        assert not should_validate(temp_cache_dir, "abc123", "subsumption_st_v1")

    def test_run_ab_validation_without_llm(self, temp_cache_dir, mock_category_set):
        """Validation should succeed with just the new method (no LLM)."""
        from atelier.classify.alignment_validator import run_ab_validation

        with patch(
            "atelier.classify.subsumption_alignment.build_alignment_via_subsumption",
            return_value={"ICE.A": "ICE.SENSITIVE.PID.CONTACT.EMAIL"},
        ):
            report = run_ab_validation(
                mock_category_set,
                cache_dir=temp_cache_dir,
                force=True,
            )

        assert report is not None
        assert report.method_new == "subsumption_st_v1"
        assert len(report.alignment_new) == 1
        assert report.alignment_legacy == {}
        # Report file should exist
        report_files = list(temp_cache_dir.glob("*.json"))
        assert len(report_files) == 1
