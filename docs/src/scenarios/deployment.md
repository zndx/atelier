# Deployment Modalities

Cloudera AI offers four ways to run code. Each has different constraints on networking, filesystem layout, process lifecycle, and dependency management. Atelier's BDD scenarios encode these constraints as executable specifications.

```d2
direction: right

project: Project {
  tooltip: "Git-backed workspace\nBase for all modalities"
}

amp: AMP {
  tooltip: "One-click provisioning\n.project-metadata.yaml"
}

app: Application {
  tooltip: "Long-running service\nCDSW_APP_PORT binding"
}

studio: Studio {
  tooltip: "Pre-built Docker image\nIS_COMPOSABLE=true"
}

project -> amp: install + start tasks
project -> app: bind to reverse proxy
project -> studio: embedded service {style.stroke-dash: 3}
```

## Project

Every CAI deployment starts as a Project — a Git-backed workspace cloned into `/home/cdsw`. The Project modality is implicit: it provides the filesystem layout, environment variables, and Python runtime that all other modalities build on.

No dedicated feature file. Project constraints are tested indirectly through every other deployment scenario.

## AMP (Automated Machine Learning Prototype)

An AMP is a one-click provisioning workflow defined in `.project-metadata.yaml`. It runs a sequence of tasks — typically `create_job` to install dependencies, then `start_application` to launch the service.

**Why BDD captures this well:** AMP metadata is YAML that CAI interprets at deploy time. A malformed task definition doesn't fail until someone clicks "Deploy" in the CAI UI. Our tier-0 scenarios catch structural problems immediately.

### What the scenarios validate

**AMP metadata structure** (`amp_lifecycle.feature`):

```gherkin
Scenario: AMP metadata file is valid
  Given the file ".project-metadata.yaml" exists
  When I parse the AMP metadata
  Then it has a "name" field
  And it has a "runtimes" section
  And it has a "tasks" section
```

**Task ordering pattern** — CAI requires `create_job` before `run_job` for the same entity label. Getting this wrong means the install job never runs:

```gherkin
Scenario: AMP tasks follow create_job/run_job pattern
  Given the AMP metadata is loaded
  Then a "create_job" task with entity_label "install_deps" exists
  And a "run_job" task with entity_label "install_deps" exists
  And a "start_application" task exists
```

**Install script validity** — `scripts/install_deps.py` runs in a bare Python environment without uv or devenv. A syntax error here means the entire deployment fails:

```gherkin
Scenario: Install script is valid Python
  When I compile "scripts/install_deps.py" with py_compile
  Then no SyntaxError is raised
```

**Tier-CAI scenarios** document what a successful AMP deploy looks like. These are skipped locally but serve as a regression checklist when debugging deployment failures:

```gherkin
@tier-cai
Scenario: AMP install job completes successfully
  Given I am in a CAI project session
  When I run the install dependencies job
  Then the job exits with code 0
  And "atelier" is importable in system Python
  And "node --version" succeeds
  And the directory "ui/dist" exists
```

## Application

An Application is a long-running web service. CAI assigns a port via `CDSW_APP_PORT` and routes subdomain traffic through a reverse proxy that handles authentication.

**The key constraint:** When `CDSW_APP_PORT` is set, the service must bind to `127.0.0.1`, not `0.0.0.0`. The reverse proxy connects over localhost; binding to all interfaces bypasses CAI's auth layer.

For local development (no `CDSW_APP_PORT`), binding to `0.0.0.0` is correct — it lets you access the service from a browser.

```gherkin
Scenario: start-app.sh binds to 127.0.0.1 when CDSW_APP_PORT is set
  Given CDSW_APP_PORT is set to "8090"
  When I parse bin/start-app.sh for the HOST variable
  Then HOST is "127.0.0.1"

Scenario: start-app.sh binds to 0.0.0.0 for local dev
  Given CDSW_APP_PORT is not set
  When I parse bin/start-app.sh for the HOST variable
  Then HOST is "0.0.0.0"
```

The tier-1 scenario verifies the full stack actually starts and serves traffic:

```gherkin
@tier-1
Scenario: Full application stack starts locally
  When I run bin/start-app.sh in the background
  Then the HTTP gateway responds on port 8090 within 30 seconds
  And the gRPC server responds on port 50051
```

## Studio (future)

A Studio is a pre-built Docker image where `IS_COMPOSABLE=true`. Instead of being the root application, Atelier runs as an embedded service within a larger container.

**The key constraint:** When `IS_COMPOSABLE` is set, the install script must use `/home/cdsw/atelier` as the root directory (the project subdirectory) instead of `/home/cdsw` (the container root). Getting this wrong means dependencies install into the wrong location and imports fail at startup.

```gherkin
Scenario: install_deps.py handles IS_COMPOSABLE root path
  When I set IS_COMPOSABLE to "true"
  And I parse scripts/install_deps.py for root_dir
  Then root_dir is "/home/cdsw/atelier"

Scenario: install_deps.py uses default root without IS_COMPOSABLE
  When IS_COMPOSABLE is not set
  And I parse scripts/install_deps.py for root_dir
  Then root_dir is "/home/cdsw"
```

Studio support is currently speculative — these scenarios document the expected behavior so the contract is established before implementation begins.
