# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
        uv run uvicorn atelier.gateway:app --reload --host 0.0.0.0 --port ${CDSW_APP_PORT:-8090}

# ── Database ─────────────────────────────────────────────────────

# Helper: build dbmate-compatible URL (strip +psycopg, add sslmode=disable for local)
_db_url := "$(uv run python -c 'from atelier.config import load_config; u=load_config().db_url.replace(\"+psycopg\",\"\"); print(u+(\"?\" if \"?\" not in u else \"&\")+\"sslmode=disable\")')"

# Run migrations + seed keystone agents (what devenv up does automatically)
bootstrap:
    uv run python -m atelier.db.bootstrap

# Run dbmate migrations against the configured database
migrate:
    dbmate --url "{{_db_url}}" --migrations-dir db/migrations up

# Rollback last migration
migrate-down:
    dbmate --url "{{_db_url}}" --migrations-dir db/migrations down

# Show migration status
migrate-status:
    dbmate --url "{{_db_url}}" --migrations-dir db/migrations status

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

# Run BDD — full stack (tier-0 + tier-1), excludes @slow.  Auto-starts
# devenv if needed.  Canonical BDD entry point.
behave *ARGS:
    ATELIER_BDD_TIER=1 uv run behave features/ --tags="~@slow" {{ARGS}}

# Run BDD with full stack including @slow scenarios (pipeline convergence,
# ML training).  Heavy synth-scale validation (9782 cols × 512 perms)
# belongs to UI-driven pipeline runs, not BDD.
behave-slow *ARGS:
    ATELIER_BDD_TIER=1 uv run behave features/ {{ARGS}}

# Run only deployment runtime profile
bdd-runtime:
    ATELIER_BDD_TIER=0 uv run behave features/deployment/runtime_profile.feature

# ── Secrets ───────────────────────────────────────────────────────

# Materialize all SOPS-encrypted artifacts into their runtime paths.
# Called automatically by bin/start-app.sh (CAI) and devenv enterShell
# (local); also safe to run manually from any checkout.
bootstrap-secrets:
    bash bin/bootstrap-secrets.sh

# Decrypt CAI env defaults (requires age private key).
# Decrypts as JSON and shell-quotes each value so values containing
# spaces or special chars survive `source .env.cai` cleanly — sops's
# native dotenv output leaves values unquoted.
decrypt-secrets:
    sops --decrypt --output-type json .env.cai.enc \
      | python3 -c 'import json,shlex,sys; d=json.load(sys.stdin); [print(f"{k}={shlex.quote(str(v))}") for k,v in d.items() if not k.startswith("sops")]' \
      > .env.cai
    @echo "Decrypted .env.cai ($(wc -l < .env.cai) lines)"

# Encrypt CAI env defaults (after editing .env.cai)
encrypt-secrets:
    sops --encrypt --input-type dotenv --output-type json .env.cai > .env.cai.enc
    @echo "Encrypted .env.cai.enc"

# Decrypt the curated-reference CSV from BDD fixtures into build/data/
# for local inspection. Safe to re-run; plaintext is gitignored.
decrypt-reference:
    mkdir -p build/data
    sops --decrypt features/fixtures/curated_reference.csv.enc > build/data/curated_reference.csv
    @echo "Decrypted build/data/curated_reference.csv ($(wc -l < build/data/curated_reference.csv) lines)"

# Encrypt the curated-reference CSV at build/data/curated_reference.csv
# back into features/fixtures/ for commit. Maintainer runs this after
# updating the answer key. --filename-override lets SOPS resolve
# creation_rules against the intended destination path without
# staging plaintext under features/fixtures/.
encrypt-reference:
    mkdir -p features/fixtures
    sops --encrypt --input-type binary --output-type binary \
        --filename-override features/fixtures/curated_reference.csv \
        build/data/curated_reference.csv > features/fixtures/curated_reference.csv.enc
    @echo "Encrypted features/fixtures/curated_reference.csv.enc"

# ── Governance ────────────────────────────────────────────────────

# Push default.annotations → Atlas as classification typedefs.
# Idempotent: skips typedefs that already exist (no drift correction).
# Atlas creds come from ATELIER_ATLAS_URL/USER/PASSWORD via materialized HOCON.
sync-taxonomy *ARGS:
    env -i $(cat build/config/atelier.env 2>/dev/null | xargs) PATH="$$PATH" \
        uv run python -m atelier.governance taxonomy {{ARGS}}

# ── Versioning ────────────────────────────────────────────────────

# Bump version: just bump-version --minor (or --patch, --major, X.Y.Z)
bump-version *ARGS:
    scripts/bump-version.sh {{ARGS}}

# ── Documentation ─────────────────────────────────────────────────

# Build mdbook docs
docs-build:
    mdbook build docs/

# Serve mdbook docs with live reload
docs-serve:
    mdbook serve docs/

# ── Release tooling ───────────────────────────────────────────────

# Stamp Cloudera proprietary header on every shippable source + doc.
# RUN ON A RELEASE BRANCH ONLY — trunk stays unmarked in the dev tree.
stamp-headers:
    uv run python scripts/apply_cloudera_header.py

# Preview what would be stamped without writing anything.
stamp-headers-dry:
    uv run python scripts/apply_cloudera_header.py --dry-run --verbose

# CI gate — exit 1 if any tracked file is missing the Cloudera header.
stamp-headers-check:
    uv run python scripts/apply_cloudera_header.py --check

# Build a self-contained source archive (atelier-{version}.tar.gz) for
# offline CAI deployments — main repo + embedding-atlas submodule, no
# hermes-agent.  Run on a release branch so the archive carries the
# stamped headers and the bumped version.
build-archive:
    bash scripts/build_source_archive.sh
