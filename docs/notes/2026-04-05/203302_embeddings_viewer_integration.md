<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Embeddings Viewer Integration

## What Changed

Integrated Apple's embedding-atlas library into the Atelier React frontend,
renamed all "Atlas Viewer" references to "Embeddings Viewer" to avoid confusion
with Apache Atlas (Cloudera metadata governance), and wired the backend to serve
real parquet datasets from a GitTables benchmark preparation pipeline.

## Naming Audit

Renamed "Atlas Viewer" to "Embeddings Viewer" across all user-facing surfaces:

- `ui/src/pages/Landing.tsx` - card title and description
- `docs/src/introduction.md`, `docs/src/architecture/overview.md`, `docs/src/architecture/agents.md`
- `README.md`, `catalog-entry.yaml`, `CLAUDE.md`
- `docs/src/scenarios/overview.md`

Attribution ("powered by embedding-atlas") kept in developer docs only.
Disambiguation note added to `docs/src/architecture/embeddings.md`.

## New Files

| File | Purpose |
|------|---------|
| `scripts/prepare_gittables_sample.py` | Reads signals eval parquet, computes sentence-transformer embeddings + UMAP 2D projection, outputs visualization-ready parquet |
| `ui/src/pages/Embeddings.tsx` | React page: DuckDB WASM + Mosaic coordinator + EmbeddingAtlas component (renamed from EmbeddingsViewer.tsx) |
| `features/deployment/embeddings.feature` | 4 tier-0 scenarios: npm dep, page component, React Router, preparation script (renamed from embeddings_viewer.feature) |
| `features/deployment/naming_audit.feature` | 2 tier-0 scenarios: no Apache Atlas confusion in UI/docs |
| `features/deployment/step_defs/naming_steps.py` | Step defs for file search and content matching |
| `docs/src/architecture/embeddings.md` | Architecture page with d2 diagram, parquet schema, GitTables description |
| `data/.gitkeep` | Data directory (parquet files gitignored) |

## Modified Files

| File | Change |
|------|--------|
| `ui/src/App.tsx` | Added React Router with `/` and `/embeddings/:datasetId` routes |
| `ui/src/pages/Landing.tsx` | Real API stats, dataset links, "Embeddings Viewer" naming |
| `ui/package.json` | Added embedding-atlas, @uwdata/mosaic-core, @uwdata/mosaic-sql, react-router-dom |
| `src/atelier/service.py` | ListDatasets wired to AtelierDao (no longer stub) |
| `src/atelier/gateway.py` | Added `/api/datasets/{id}/data` parquet serving endpoint |
| `src/atelier/db/dao.py` | Added list_datasets(), get_dataset(), upsert_dataset() |
| `justfile` | Added `seed` and `prepare-gittables` recipes |
| `pyproject.toml` | Added sentence-transformers, umap-learn to viz optional deps |
| `.gitignore` | Added data/*.parquet, data/*.json |
| `features/steps/__init__.py` | Re-export naming_steps |

## BDD Status

24 tier-0 scenarios pass (up from 18), 6 correctly skipped (tier-1/cai).
30 total scenarios across 10 features, 2 domains.

## Remaining (requires devenv shell)

1. `pnpm install` in ui/ to install declared npm dependencies
2. `npm run build` to produce ui/dist/ with new pages
3. `just prepare-gittables ~/local/src/cldr/signals/build/gittables_eval.parquet`
4. `just seed` to register the GitTables dataset in the database
5. Manual browser verification of the Embeddings Viewer

## Next Session

E2E testing with GitTables using signals DST/CatBoost methodology.
Treat GitTables' 122 DBpedia instance labels as controlled vocabulary
to be grounded in the SIGDG ontology.
