# Runtime Profile

The CAI Runtime Profile is a set of tier-0 scenarios that validate deployment readiness without requiring a live CAI session. Run it before every push to catch the class of errors that only manifest when CAI tries to start the application.

```bash
just bdd-runtime
```

## Why This Exists

CAI deployment failures are expensive to debug. The install job runs in a container with a 30-minute timeout. If it fails, the only feedback is a log dump. If it succeeds but the application crashes at startup, the only feedback is a "Application failed to start" banner with a link to logs that may or may not contain the root cause.

The runtime profile catches failures that would otherwise require a deploy-debug-redeploy cycle:

| Check | Failure mode it prevents |
|-------|--------------------------|
| Core package importable | Missing `__init__.py`, circular imports, broken package structure |
| Entry points importable | New dependency not declared in `pyproject.toml` |
| Proto stubs importable | Forgot to run `just proto` after editing `.proto` |
| Scripts exist and are executable | Missing `chmod +x`, file not committed |
| HOCON config resolves | Undefined substitution variable, syntax error in `.conf` |
| Migrations parseable | Malformed `-- migrate:up` block, missing SQL terminator |

## The Scenarios

### Import chain validation

The most common CAI deployment failure is an import error. A module works in devenv because all dev dependencies are installed, but fails in CAI because the install script only installs production dependencies.

```gherkin
Scenario: Core package is importable
  When I import "atelier"
  Then no ImportError is raised
  And atelier.__version__ is defined

Scenario: All entry points are importable
  When I import "atelier.server"
  And I import "atelier.gateway"
  And I import "atelier.config"
  And I import "atelier.db.bootstrap"
  Then no ImportError is raised

Scenario: Proto stubs are generated and importable
  When I import "atelier.proto.atelier_pb2"
  And I import "atelier.proto.atelier_pb2_grpc"
  Then no ImportError is raised
```

These scenarios exercise the full import graph. If `atelier.gateway` imports `fastapi` which imports `pydantic` which imports `annotated_types`, and `annotated_types` isn't in the dependency chain — this catches it.

### Script executability

CAI runs scripts via `#!/usr/bin/env python3` or `#!/usr/bin/env bash`. If the shebang is wrong or the execute bit isn't set, the deploy fails with a cryptic "Permission denied" error.

```gherkin
Scenario: Required scripts exist and are executable
  Then the file "scripts/install_deps.py" exists
  And the file "scripts/startup_app.py" exists
  And the file "scripts/install_node.sh" is executable
  And the file "scripts/install_qdrant.sh" is executable
  And the file "bin/start-app.sh" is executable
```

### Configuration resolution

HOCON configs use `${?VAR}` substitution for environment variables. A typo in a variable name or an unresolvable reference won't fail until `load_config()` is called at startup. The runtime profile forces resolution at test time:

```gherkin
Scenario: HOCON config resolves without errors
  When I load the config with no overrides
  Then no exception is raised
  And the config has grpc_port > 0
  And the config has gateway_port > 0
```

### Migration parsing

Atelier uses a [dbmate](https://github.com/amacneil/dbmate)-compatible migration runner (`atelier.db.bootstrap`) that parses `-- migrate:up` / `-- migrate:down` blocks from SQL files. If a migration is missing its UP block, the bootstrap silently skips it — which means the schema diverges from what the code expects.

```gherkin
Scenario: Database migrations are parseable
  Given migration files exist in "db/migrations/"
  When I parse each migration for UP/DOWN blocks
  Then every migration has a valid UP block
```

## When to Extend the Profile

Add a new runtime profile scenario whenever you:

- Add a new Python entry point or importable module
- Add a new script that CAI executes directly
- Add a new HOCON config key that downstream code depends on
- Add a new migration file

The rule of thumb: if it can break a CAI deploy and you can verify it without services running, it belongs in the runtime profile.
