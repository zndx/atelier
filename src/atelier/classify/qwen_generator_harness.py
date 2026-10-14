"""Withdrawn. This harness wrote stub relational generators.

Augmented collections are an Aegir projection into sdg-corpora. Calling
any entry point here is an error.
"""

from __future__ import annotations


class WithdrawnGeneratorError(RuntimeError):
    """Stub relational-generator authoring is withdrawn."""


_MESSAGE = (
    "stub relational generators are withdrawn; "
    "augmented tables are an sdg-corpora projection"
)


def _withdrawn(*_args, **_kwargs):
    raise WithdrawnGeneratorError(_MESSAGE)


write_generator_pack = _withdrawn
select_uncovered = _withdrawn
generator_argv = _withdrawn
incorporate = _withdrawn
eligible_codes = _withdrawn
read_code_list = _withdrawn


def main(argv: list[str] | None = None) -> int:
    """Refuse to author or install stub generators."""
    del argv
    raise WithdrawnGeneratorError(_MESSAGE)
