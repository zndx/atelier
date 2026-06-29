# Atelier — Docker test stack

A self-contained Docker Compose stack for testing the application (the
new `ui-next` frontend + gRPC core + gateway) against containerized
Postgres and Qdrant. No devenv, Nix, or local toolchain required.

## What's in it

| Service    | Image                      | Ports (host) | Role                                            |
|------------|----------------------------|--------------|-------------------------------------------------|
| `postgres` | `pgvector/pgvector:pg16`   | `5532`       | State persistence (pgvector for embeddings)     |
| `qdrant`   | `qdrant/qdrant:v1.12.4`    | `6333/6334`  | Vector store (late-interaction maxsim)          |
| `app`      | built from `docker/Dockerfile` | `8090`   | gRPC servicer + FastAPI gateway serving the UI  |

The `app` image is multi-stage: stage 1 builds `ui-next` into `ui/dist`
with Node; stage 2 is a Python runtime that installs the backend, bakes
the `all-MiniLM-L6-v2` embedding model into the image, and serves
everything on one port.

Setting `ATELIER_DB_URL` in compose makes the app use the external
Postgres and **skip its bundled PGlite**; `QDRANT_HOST` points it at the
Qdrant container. So `docker/entrypoint.sh` only has to: resolve config →
wait for Postgres → migrate → start gRPC → start the gateway.

## Run it

```bash
docker compose up --build
# then open the new UI (served same-origin by the gateway):
open http://localhost:8090
```

First build is large and slow — the backend pulls the full ML stack
(torch, sentence-transformers, catboost, docling, …) and bakes the
embedding model. Subsequent builds are cached.

Tear down (and wipe the DB/vector volumes):

```bash
docker compose down -v
```

## Credentials (optional)

The stack **boots without any API keys** — health, navigation, Status,
Settings, dataset registries, and the embedding atlas all work. You only
need credentials to **start a classification run** (`POST /api/fsm/start`)
or use the **agent terminal**.

```bash
cp docker/.env.example docker/.env
# edit docker/.env — e.g. ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

`docker/.env` is gitignored and loaded into the `app` container.

> The agent terminal additionally needs the `claude` CLI inside the
> container; this test image does not install it. The rest of the UI and
> the classification pipeline do not require it.

## Verifying it's up

```bash
curl -s http://localhost:8090/api/health        # gRPC health via gateway
curl -s http://localhost:8090/api/status        # aggregated service health
curl -s http://localhost:6333/readyz            # qdrant
docker compose exec postgres psql -U postgres -d atelier -c '\dt'  # migrated tables
```

The gateway's `/api/status` should report `grpc.ok`, `postgres.ok`, and
`qdrant.ok` all true once the stack settles.

## Notes & troubleshooting

- **Architecture / missing wheels.** On Apple Silicon the image builds
  `arm64` natively. If a Python wheel is unavailable for arm64, build for
  amd64 (emulated, slower): `docker compose build --build-arg … ` or add
  `platform: linux/amd64` to the `app` service.
- **Ports.** Postgres is published on host `5532` (not 5432/5533) to avoid
  clashing with a local Postgres or the devenv one. The app is on `8090`.
- **Data reset.** `docker compose down -v` drops the `pgdata` and
  `qdrant_storage` volumes for a clean slate; omit `-v` to keep them.
- **This is a test harness, not the CAI deployment.** Production uses
  `bin/start-app.sh` (PGlite supervisor, Qdrant binary, SOPS secrets).
  This compose deliberately swaps those for plain service containers.
