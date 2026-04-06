# Atelier task recipes
#
# HOCON (config/base.conf) is the single source of truth for all config.
# Environment variables are captured by HOCON, not read directly by code.
#
# Workflow:
#   1. cp .env.example .env && edit .env
#   2. just resolve-config    (materializes build/config/atelier.env)
#   3. just preflight          (validates all required keys)
#   4. just up / just test / just proto ...

# ── Config management ─────────────────────────────────────────────

# Resolve HOCON config + env vars to build/config/atelier.env
resolve-config:
    uv run python -c "from atelier.config import load_config, materialize_config; materialize_config(load_config(), 'build/config/atelier.env')"
    @echo "Resolved config -> build/config/atelier.env"

# Validate materialized config has all required keys
preflight:
    uv run python -c "from atelier.config import validate_materialized_config; errs = validate_materialized_config(); [print(f'  ERROR: {e}') for e in errs]; exit(1) if errs else print('Preflight OK')"

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

# ── Build ─────────────────────────────────────────────────────────

# Install all dependencies
install:
    uv sync && cd ui && pnpm install

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

# ── Documentation ─────────────────────────────────────────────────

# Build mdbook docs
docs-build:
    mdbook build docs/

# Serve mdbook docs with live reload
docs-serve:
    mdbook serve docs/
