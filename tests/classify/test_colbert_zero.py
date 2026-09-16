"""ColBERT-Zero is the MaxSim encoder; ModernBERT stays NHSVM-only."""

from __future__ import annotations

import pytest

from atelier.classify.colbert_encoder import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    GURU_RETIRED,
    refuse_retired_embedding_model,
    set_model_name,
)
from atelier.config import AtelierConfig
from atelier.optimize.precondition import COLBERT_MODEL, ENCODER_ID
from atelier.optimize.svm.encoder import EMB_DIM as MODERNBERT_DIM
from atelier.optimize.svm.encoder import MODEL_ID as MODERNBERT_ID


def test_maxsim_default_is_colbert_zero() -> None:
    assert DEFAULT_MODEL == "lightonai/ColBERT-Zero"
    assert COLBERT_MODEL == DEFAULT_MODEL
    assert AtelierConfig().classify_colbert_model == DEFAULT_MODEL
    assert EMBEDDING_DIM == 128


def test_nhsvm_encoder_is_modernbert_not_colbert_zero() -> None:
    assert ENCODER_ID == "answerdotai/ModernBERT-base"
    assert MODERNBERT_ID == ENCODER_ID
    assert MODERNBERT_ID != DEFAULT_MODEL
    assert MODERNBERT_DIM == 768
    assert MODERNBERT_DIM != EMBEDDING_DIM


def test_colbert_v2_is_retired_for_encode() -> None:
    with pytest.raises(RuntimeError, match=GURU_RETIRED):
        refuse_retired_embedding_model("colbert-ir/colbertv2.0")
    with pytest.raises(RuntimeError, match=GURU_RETIRED):
        set_model_name("colbert-ir/colbertv2.0")


def test_colpali_family_retired() -> None:
    with pytest.raises(RuntimeError, match=GURU_RETIRED):
        refuse_retired_embedding_model("vidore/colpali")


def test_empty_name_means_colbert_zero() -> None:
    refuse_retired_embedding_model(None)
    refuse_retired_embedding_model("")
    refuse_retired_embedding_model(DEFAULT_MODEL)
