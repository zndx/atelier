# Qwen pack lists installed IRIs

Batch 91 of the coverage run returned after 12 turns with no `out/extend_generators.py`. The model had spent the turns reading `installed.py`. That file was a copy of `insitu_extensions.py`, about 235 KB and 360 generator bodies, and the call reported 305,939 input tokens.

The pack now writes `INSTALLED`, a tuple of the accepted IRI strings. The harness still reads the full module to decide which codes are uncovered. The skill tells Qwen that `installed.py` is that tuple.

`uv run pytest tests/classify/test_qwen_generator_harness.py`: 9 passed. The coverage process was restarted onto this pack. The generators already accepted stay in the working tree.
