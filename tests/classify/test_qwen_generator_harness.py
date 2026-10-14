"""The stub relational-generator harness is withdrawn."""

from atelier.classify.extension_patterns import ExtensionError, install_generators
from atelier.classify.qwen_generator_harness import (
    WithdrawnGeneratorError,
    incorporate,
    main,
    write_generator_pack,
)


def test_authoring_a_stub_generator_is_refused(tmp_path) -> None:
    try:
        write_generator_pack(tmp_path, tmp_path / "annotations.csv", ["SDG.ROLE"])
    except WithdrawnGeneratorError as exc:
        assert "withdrawn" in str(exc)
    else:
        raise AssertionError("write_generator_pack must refuse")
    try:
        incorporate(tmp_path, tmp_path / "annotations.csv")
    except WithdrawnGeneratorError as exc:
        assert "sdg-corpora" in str(exc)
    else:
        raise AssertionError("incorporate must refuse")
    try:
        main([])
    except WithdrawnGeneratorError as exc:
        assert "withdrawn" in str(exc)
    else:
        raise AssertionError("main must refuse")
    try:
        install_generators(["stub"], tmp_path / "insitu_extensions.py")
    except ExtensionError as exc:
        assert "withdrawn" in str(exc)
    else:
        raise AssertionError("install_generators must refuse")
