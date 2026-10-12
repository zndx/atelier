"""Workpack isolation and post-hoc scoring. No live model calls."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from atelier.classify.qwen_synth_harness import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    QwenSynthError,
    grok_argv,
    grok_env,
    pack_inputs,
    run_generator,
    score_workdir,
    write_workpack,
)
from tests.classify.explicit_sdg import (
    CLINICAL,
    PARENTING,
    PDF_EXPORT,
    PROCESS,
    ROLE,
    YOUNG_RESEARCHER,
    write_annotations,
)

_SLICE = (PROCESS, ROLE, PDF_EXPORT, PARENTING)


def _pack(tmp_path: Path, codes: list[str] | None = None) -> Path:
    write_annotations(tmp_path / "annotations.csv", _SLICE, sentinel=True)
    (tmp_path / "naming_map.parquet").write_bytes(b"slot_ref INV-1")
    dest = tmp_path / "work"
    write_workpack(dest, tmp_path / "annotations.csv", codes or [PDF_EXPORT])
    return dest


def test_workpack_is_an_ontology_slice(tmp_path: Path) -> None:
    dest = _pack(tmp_path)
    ontology = (dest / "ontology.csv").read_text(encoding="utf-8")
    assert ontology.splitlines()[0] == "code,label,abbrev,parent_code,description"
    assert PDF_EXPORT in ontology
    assert any(line.startswith(f"{PROCESS},") for line in ontology.splitlines())
    assert PARENTING not in ontology
    assert f"{ROLE}," not in ontology
    assert "example_values" not in ontology
    assert "INV-SECRET" not in ontology
    assert "slot_ref" not in ontology
    inputs = "\n".join(path.read_text(encoding="utf-8") for path in pack_inputs(dest))
    assert "naming_map" not in inputs
    assert "INV-SECRET" not in inputs
    assert list(dest.rglob("*.parquet")) == []
    request = json.loads((dest / "request.json").read_text(encoding="utf-8"))
    assert request["codes"] == [PDF_EXPORT]
    assert (dest / ".grok" / "skills" / "synthetic-relational-corpus" / "SKILL.md").is_file()
    assert (dest / ".grok" / "agents" / "relational-synth.md").is_file()


def test_code_outside_the_explicit_slice_fails_before_a_partial_pack(tmp_path: Path) -> None:
    write_annotations(tmp_path / "annotations.csv", _SLICE)
    dest = tmp_path / "work"
    with pytest.raises(QwenSynthError, match=CLINICAL):
        write_workpack(dest, tmp_path / "annotations.csv", [CLINICAL])
    assert not dest.exists()


def test_default_argv_is_local_qwen(tmp_path: Path) -> None:
    dest = _pack(tmp_path)
    argv = grok_argv(dest)
    assert DEFAULT_PROVIDER == "local"
    assert argv[argv.index("-m") + 1] == PROVIDERS["local"]
    assert "qwen-thinking-cerebras" not in argv
    assert "api.cerebras.ai" not in argv
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "strict"
    assert "--tools" in argv
    assert "run_terminal_cmd" in argv[argv.index("--disallowed-tools") + 1]
    denies = [argv[i + 1] for i, item in enumerate(argv) if item == "--deny"]
    assert any(rule.startswith("Read(") and rule.endswith("/**)") for rule in denies)
    env = grok_env({"PATH": "/usr/bin"})
    if list(Path("/nix/store").glob("*-bubblewrap-*/bin/bwrap")):
        assert shutil.which("bwrap", path=env["PATH"])
    hosted = grok_argv(dest, provider="cerebras")
    assert hosted[hosted.index("-m") + 1] == "qwen-thinking-cerebras"
    env = grok_env({})
    assert env["GROK_CLAUDE_SKILLS_ENABLED"] == "false"
    assert env["GROK_CURSOR_SKILLS_ENABLED"] == "false"


def test_run_generator_records_the_local_invocation(tmp_path: Path) -> None:
    dest = _pack(tmp_path)
    seen: dict[str, object] = {}

    def runner(argv, env, timeout_s):
        seen["argv"] = argv
        seen["env"] = env
        seen["timeout"] = timeout_s
        return subprocess.CompletedProcess(argv, 0, stdout='{"ok":true}\n', stderr="")

    completed = run_generator(dest, runner=runner, timeout_s=15)
    assert completed.returncode == 0
    assert seen["argv"][seen["argv"].index("-m") + 1] == "qwen-thinking"  # type: ignore[index]
    assert seen["timeout"] == 15
    timing = json.loads((dest / "harness" / "timing.json").read_text(encoding="utf-8"))
    assert timing["provider"] == "local"
    assert timing["model"] == "qwen-thinking"
    assert "qwen-thinking-cerebras" not in (dest / "harness" / "argv.json").read_text(encoding="utf-8")


def test_score_jaccard_uses_only_gold_passed_after_generation(tmp_path: Path) -> None:
    dest = _pack(tmp_path, [PDF_EXPORT, PARENTING])
    tables = dest / "out" / "tables"
    (tables / "billing.csv").write_text(
        "export_number,note\nINV-19,hello\nINV-37,there\n",
        encoding="utf-8",
    )
    (dest / "out" / "manifest.json").write_text(
        json.dumps([
            {"table": "billing", "column": "export_number", "code": PDF_EXPORT},
            {"table": "billing", "column": "note", "code": ""},
        ]),
        encoding="utf-8",
    )
    gold = [SimpleNamespace(
        table="billing", column="export_id", code=PDF_EXPORT,
        values=["INV-100", "INV-200"],
    )]
    score = score_workdir(dest, gold)
    assert score.parse_ok
    assert score.coverage == 0.5
    assert score.emitted_codes == (PDF_EXPORT,)
    assert score.name_jaccard == pytest.approx(1 / 3)
    assert score.value_jaccard == pytest.approx(1 / 5)
    assert score_workdir(dest, None).name_jaccard is None
    assert score_workdir(dest, None).value_jaccard is None


def test_code_outside_the_slice_is_not_a_successful_parse(tmp_path: Path) -> None:
    dest = _pack(tmp_path)
    (dest / "out" / "tables" / "billing.csv").write_text(
        "doc_ref\nINV-9\n",
        encoding="utf-8",
    )
    (dest / "out" / "manifest.json").write_text(
        json.dumps([{"table": "billing", "column": "doc_ref", "code": YOUNG_RESEARCHER}]),
        encoding="utf-8",
    )
    score = score_workdir(dest, None)
    assert not score.parse_ok
    assert score.coverage == 0.0
    assert any(YOUNG_RESEARCHER in problem for problem in score.problems)
