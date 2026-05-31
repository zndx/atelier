"""atelier.optimize.maxsim — GEPA-shaped semantic optimization of the
ColBERT/Qdrant maxsim channel (ColBERT late-interaction scored via Qdrant MaxSim).

Modules:
    semantic  — LLM critic loop that proposes enrichment-payload edits
                and measures rescue@K against the agent-mediated reference

The CLI surface (`scripts/semantic_optimize.py`) imports from here.
"""
