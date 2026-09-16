"""CLI entry: apply discovered platform Metaflow env, then ClassificationFlow.

Metaflow reads datastore/metadata at import. Set env before importing the
FlowSpec so we never fall through to local / metaflow-artifacts.
"""

from __future__ import annotations

import os


def main() -> None:
    from atelier.flows.platform_metaflow import metaflow_child_env, require_signals_metaflow

    env = metaflow_child_env()
    os.environ.update(env)
    require_signals_metaflow(os.environ)
    from atelier.flows.classify import ClassificationFlow

    ClassificationFlow()


if __name__ == "__main__":
    main()
