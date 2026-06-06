# Embeddings

The Embeddings page provides interactive visualization of classification results. It renders 2D projections of embedding vectors, allowing users to explore clusters, search data points, and cross-filter by metadata columns.

## Architecture

```d2
direction: right

browser: Browser {
  react: React App
  duckdb: DuckDB WASM {
    tooltip: "In-browser SQL engine\nLoads parquet directly"
  }
  viewer: EmbeddingAtlas Component {
    tooltip: "WebGPU scatter plot\nDensity contours\nSearch + filters"
  }
}

gateway: FastAPI Gateway {
  api: "/api/datasets/{id}/data"
}

react -> duckdb: fetch parquet
duckdb -> viewer: SQL queries
react -> gateway: GET parquet
```

The viewer runs entirely in the browser. DuckDB WASM loads parquet data locally and the EmbeddingAtlas component renders the visualization using WebGPU with WebGL 2 fallback. The component comes from Atelier's fork of Apple's embedding-atlas, [`rch/oss-embedding-atlas`](https://github.com/rch/oss-embedding-atlas), which carries required modifications and ships a committed pre-built `dist/` so CAI deployments don't need the full build toolchain.

## Data Flow

1. **Backend** serves the parquet file via `/api/datasets/{id}/data`
2. **React** fetches the parquet and loads it into DuckDB WASM via a Mosaic coordinator
3. **EmbeddingAtlas** queries the DuckDB table for rendering: x/y coordinates, categories, text for tooltips
4. All filtering, search, and aggregation happens client-side — no round-trips to the server

## Parquet Schema

The Embeddings page expects parquet files with these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | yes | Unique row identifier |
| `x` | float32 | yes | 2D projection x-coordinate (UMAP) |
| `y` | float32 | yes | 2D projection y-coordinate (UMAP) |
| `text` | string | recommended | Tooltip and search text |
| `category` | string | recommended | Color-coding category |

Additional columns (e.g., `source_table`, `belief`, `plausibility`) are automatically available as cross-filter charts.

## GitTables Dataset

The initial dataset is derived from the [GitTables](https://zenodo.org/records/5706316) CTA benchmark — 2,517 columns extracted from real tables, annotated with 122 DBpedia property types. These instance labels serve as the controlled vocabulary to be grounded in the SDG ontology.

To prepare the visualization parquet:

```bash
# From signals evaluation output (recommended)
just prepare-gittables ~/local/src/cldr/signals/build/gittables_eval.parquet

# Then seed the database
just seed
```

The preparation script computes sentence-transformer embeddings and UMAP 2D projections. The resulting parquet includes DST evidence fusion columns (belief, plausibility, uncertainty gap) when derived from the signals evaluation output.

## Naming: Embeddings vs Apache Atlas

The Embeddings page is powered by the `embedding-atlas` library (Atelier's [fork](https://github.com/rch/oss-embedding-atlas) of Apple's project). This is **unrelated to Apache Atlas**, the Cloudera metadata governance catalog used by the [signals](https://github.com/rch/signals) pipeline.

- **Embeddings** (Atelier) — Interactive scatter plot of classification embeddings
- **Apache Atlas** (Cloudera/signals) — Metadata governance catalog on port 21000

To avoid confusion, all user-facing surfaces use "Embeddings". The `embedding-atlas` library name appears only in developer documentation and `package.json`.
