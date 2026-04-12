"""Atelier classification pipeline — Dempster-Shafer evidence fusion.

Classifies database column metadata against a controlled vocabulary using
multiple independent evidence sources (cosine similarity, pattern detection,
name matching) fused via Dempster's rule of combination.

Ported from the signals project methodology, adapted for agent-mediated
execution on Cloudera AI with hive data connections.

Public API::

    from atelier.classify import run_pipeline, get_fsm_status
    result = run_pipeline(cfg, use_mock=True)
"""

from atelier.classify.belief import HierarchicalClassification
from atelier.classify.fsm import AgentFSM, FSMRun, FSMState
from atelier.classify.pipeline import run_classification_pipeline

# Module-level FSM singleton (initialized lazily by gateway/service)
_fsm: AgentFSM | None = None


def get_fsm(dao=None) -> AgentFSM:
    """Return the module-level FSM singleton."""
    global _fsm
    if _fsm is None:
        _fsm = AgentFSM(dao=dao)
    return _fsm


def run_pipeline(cfg, *, use_mock: bool = False, **kwargs) -> dict:
    """Convenience wrapper for the full classification pipeline."""
    fsm = get_fsm()
    return run_classification_pipeline(cfg, fsm, use_mock=use_mock, **kwargs)


def get_fsm_status(run_id: str | None = None) -> dict | None:
    """Get current FSM status as a dict."""
    fsm = get_fsm()
    run = fsm.get_status(run_id)
    return run.to_dict() if run else None
