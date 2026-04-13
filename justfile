# Atelier task recipes
#
# HOCON (config/base.conf) is the single source of truth for all config.
# Environment variables are captured by HOCON, not read directly by code.

# Workflow:
#   1. cp .env.example .env && edit .env
#   2. devenv shell             (loads .env, provides toolchain)
#   3. just resolve-config      (hydrates HOCON from current env)
#   4. just preflight           (validates materialized config)
#   5. just up / just test / just proto ...

# ── Config management ─────────────────────────────────────────────

# Hydrate HOCON config from current environment into build artifacts
resolve-config:
    uv run python bin/resolve-config.py

# Validate materialized config (structured deny/warn checks)
preflight:
    @uv run python -c "\
    from atelier.preflight import run_preflight; \
    from atelier.config import load_config; \
    r = run_preflight(load_config()); \
    [print(f'  [pass] {c.name}: {c.message}') for c in r.checks if c.status == 'pass']; \
    [print(f'  [WARN] {c.name}: {c.message}') for c in r.warnings]; \
    [print(f'  [DENY] {c.name}: {c.message}') for c in r.denies]; \
    [print(f'         -> {c.remediation}') for c in r.denies if c.remediation]; \
    exit(0 if r.ok else 1); \
    "

# Run conftest policy checks against materialized config JSON
policy:
    conftest test build/config/atelier.json --policy policy/environment/ --all-namespaces

# Show resolved config
show-config:
    @if [ -f build/config/atelier.env ]; then cat build/config/atelier.env; else echo "Run 'just resolve-config' first"; fi

# ── Development ───────────────────────────────────────────────────

# Start all services locally (via devenv process manager)
up:
    devenv up

# Start like CAI does (gRPC + gateway, no devenv required)
start port="8090":
    bash bin/start-app.sh {{port}}

# Run gRPC server only (from materialized config)
grpc:
    env -i $(cat build/config/atelier.env 2>/dev/null | xargs) PATH="$$PATH" \
        uv run python -m atelier.server

# Run React dev server only
ui:
    cd ui && pnpm dev

# Run HTTP gateway (serves built React + proxies to gRPC)
gateway:
    env -i $(cat build/config/atelier.env 2>/dev/null | xargs) PATH="$$PATH" \
        uv run uvicorn atelier.gateway:app --reload --host 0.0.0.0 --port 8090

# ── Database ─────────────────────────────────────────────────────

# Run dbmate migrations against the configured database
migrate:
    dbmate --url "$(uv run python -c 'from atelier.config import load_config; print(load_config().db_url.replace("+psycopg", ""))')" --migrations-dir db/migrations up

# Rollback last migration
migrate-down:
    dbmate --url "$(uv run python -c 'from atelier.config import load_config; print(load_config().db_url.replace("+psycopg", ""))')" --migrations-dir db/migrations down

# Show migration status
migrate-status:
    dbmate --url "$(uv run python -c 'from atelier.config import load_config; print(load_config().db_url.replace("+psycopg", ""))')" --migrations-dir db/migrations status

# Seed database with sample datasets
seed:
    uv run python -c "from atelier.db.dao import AtelierDao; dao = AtelierDao(); dao.upsert_dataset('gittables-sample', 'GitTables CTA Benchmark', 'data/gittables_sample.parquet', '2517 columns from GitTables with 122 DBpedia instance labels as controlled vocabulary', 2517); print('Seeded gittables-sample dataset')"

# Prepare GitTables visualization parquet from signals eval output
prepare-gittables input:
    uv run python scripts/prepare_gittables_sample.py --input {{input}}

# ── Build ─────────────────────────────────────────────────────────

# Build embedding-atlas from submodule fork
build-embedding-atlas:
    cd external/embedding-atlas && npm install && \
    npm run package -w @embedding-atlas/utils && \
    npm run package -w @embedding-atlas/component && \
    npm run package -w @embedding-atlas/viewer && \
    npm run package -w embedding-atlas

# Install all dependencies
install:
    uv sync && just build-embedding-atlas && cd ui && pnpm install

# Build React frontend
build-ui:
    cd ui && pnpm build

# Generate proto stubs
proto:
    bash bin/generate-proto.sh

# ── Tests ─────────────────────────────────────────────────────────

# Run all tests (includes preflight config check)
test:
    uv run pytest

# Run BDD scenarios (tier-0, fast only — excludes @slow ML training tests)
bdd *ARGS:
    ATELIER_BDD_TIER=0 uv run behave features/ --tags="@tier-0" --tags="~@slow" {{ARGS}}

# Run BDD including slow ML training tests (tier-0 only)
bdd-slow *ARGS:
    ATELIER_BDD_TIER=0 uv run behave features/ --tags="@tier-0" {{ARGS}}

# Run BDD with full stack (tier-0 + tier-1, requires devenv services)
bdd-full *ARGS:
    ATELIER_BDD_TIER=1 uv run behave features/ {{ARGS}}

# Run only deployment runtime profile
bdd-runtime:
    ATELIER_BDD_TIER=0 uv run behave features/deployment/runtime_profile.feature

# ── Documentation ─────────────────────────────────────────────────

# Build mdbook docs
docs-build:
    mdbook build docs/

# Serve mdbook docs with live reload
docs-serve:
    mdbook serve docs/
