<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# CAI (Cloudera AI) Deployment Modalities: Comprehensive Research

## Executive Summary

Cloudera AI (formerly CML / Cloudera Machine Learning) supports four distinct deployment modalities, each serving a different purpose in the ML lifecycle. They exist in a hierarchy: a **Project** is the container; **Applications**, **Models**, **Jobs**, and **Sessions** are workload types within a Project; an **AMP** is a one-click template that creates a Project with pre-configured workloads; and a **Studio** is a higher-level, composable, runtime-image-based deployment that bundles multiple services into a single managed experience.

---

## 1. Project

### What Is It?

A **Project** is the fundamental organizational unit in Cloudera AI. It is a Git-backed workspace that contains code, data files, libraries, and configuration. All other workload types (Sessions, Jobs, Models, Applications) exist *within* a Project.

### How Is It Deployed?

- Created manually via the CML Workbench UI or API
- Backed by a Git repository (internal CML git server or external like GitHub)
- All project files reside on an NFS-mounted POSIX filesystem (`/home/cdsw/`)
- Metadata (sessions, jobs, models, applications) stored in an internal PostgreSQL database

### Runtime Constraints

- Runs on Kubernetes pods with assigned ML Runtime images
- Supports Python 3.x, R, Scala kernels
- Runtimes are Docker images (e.g., `ml-runtime-pbj-workbench-python3.10-standard`)
- Persistent storage via NFS; ephemeral compute via K8s pods
- Resource profiles: CPU cores, memory (GB), GPU count, shared memory

### Key Characteristics

| Property | Value |
|----------|-------|
| Isolation | K8s namespace per workspace |
| File system | NFS-backed, persistent across sessions |
| Git integration | Built-in or external |
| Multi-user | Yes, with RBAC |
| Workload types | Sessions, Jobs, Models, Applications, Experiments |

### Concrete Example (Dask on CML)

From `/home/rch/local/src/cldr/CML_AMP_Dask_on_CML/.project-metadata.yaml`:

```yaml
name: Distributed XGBoost with Dask on CML
runtimes:
  - editor: JupyterLab
    kernel: Python 3.9
    edition: Standard
tasks:
  - type: create_job
    name: Install Dependencies
    entity_label: install_deps
    script: scripts/install_dependencies.py
    cpu: 1
    memory: 2
  - type: run_job
    entity_label: install_deps
```

This is the simplest modality: a plain Project with a job to install deps. No application, no model deployment. The user interacts via Sessions (JupyterLab).

---

## 2. Application (Analytical Application)

### What Is It?

An **Application** is a long-running web service deployed within a CML Project. It provides an interactive UI (dashboards, model frontends, data tools) accessible via a subdomain URL.

### How Is It Deployed?

**Entry point:** A Python script that starts a web server bound to `CDSW_APP_PORT` or `CDSW_READONLY_PORT` on `127.0.0.1`.

**Configuration (via UI or API):**
- Name, subdomain (e.g., `ragstudio` -> `ragstudio.<workspace-domain>`)
- Script path (entry point)
- Kernel, CPU, memory, GPU
- Environment variables (override project-level)
- `bypass_authentication`: true/false

**Via `.project-metadata.yaml`:**
```yaml
- type: start_application
  name: RagStudio
  subdomain: ragstudio
  bypass_authentication: false
  script: scripts/startup_app.py
  cpu: 2
  memory: 4
  environment_variables:
    TASK_TYPE: START_APPLICATION
```

**Via CML API v2:**
```python
cml.create_application(
    cmlapi.CreateApplicationRequest(
        name="My App",
        subdomain="my-app",
        script="entry.py",
        cpu=2, memory=4,
        bypass_authentication=False,
    ),
    project_id=project_id,
)
```

### Runtime Constraints

| Constraint | Detail |
|------------|--------|
| Port | Must bind to `CDSW_APP_PORT` (read-write) or `CDSW_READONLY_PORT` (read-only) |
| Host | Must bind to `127.0.0.1`, never `0.0.0.0` |
| Timeout | **None** -- applications do not auto-timeout; must be stopped manually |
| Max concurrent web UIs | 3 per engine (APP_PORT, READONLY_PORT, deprecated PUBLIC_PORT) |
| Subdomain | DNS-safe: `[a-z0-9-]` |
| Authentication | Platform SSO by default; can be bypassed |
| Persistence | NFS filesystem persists; compute is ephemeral K8s pods |

### Key Property: `is_embedded`

Some applications use the `is_embedded: true` flag in `.project-metadata.yaml`:

```yaml
- type: start_application
  name: Agent Studio
  subdomain: cai-agent-studio
  script: startup_scripts/run-app.py
  is_embedded: true          # <-- marks as embedded in the platform UI
  cpu: 4
  memory: 16
```

This flag appears to integrate the application's UI into the CML workbench chrome rather than launching as a standalone browser tab. Observed in both Agent Studio and Fine Tuning Studio.

### Concrete Example (RAG Studio)

From `/home/rch/local/src/cldr/CML_AMP_RAG_Studio/scripts/startup_app.py`:

```python
import subprocess, os, cmlapi

client = cmlapi.default_client()
applications = client.list_applications(project_id=os.environ['CDSW_PROJECT_ID'])
# ... find metadata app URL ...

while True:
    subprocess.run(["bash scripts/startup_app.sh"], shell=True, env=env)
    print("Application Restarting")
```

Key pattern: the startup script runs in an infinite loop with auto-restart, uses `cmlapi` to discover sibling applications, and delegates to a shell script for the actual web server.

---

## 3. AMP (Applied ML Prototype / Accelerator for ML Projects)

### What Is It?

An **AMP** is a **complete, pre-built, one-click-deployable ML project template**. It is NOT a separate runtime -- it is a Project provisioning mechanism. When you "deploy an AMP," the platform:

1. Creates a new CML Project
2. Clones the AMP's git repository
3. Reads `.project-metadata.yaml`
4. Executes tasks sequentially (install deps, run jobs, deploy models, start applications)

AMPs were originally called "Applied ML Prototypes" (created by Cloudera's Fast Forward Labs). They are now branded as "Accelerators for ML Projects."

### How Is It Deployed?

**From the AMP Catalog:** CML Workbench left nav > "AMPs" > select > configure > deploy.

**From a custom catalog:** Admin can register external Git repos as custom AMP sources.

**From a Git URL:** Point directly at a repo containing `.project-metadata.yaml`.

**The deployment is entirely driven by `.project-metadata.yaml`:**

```yaml
name: RAG Studio
author: Cloudera
specification_version: 1.0
prototype_version: 1.0

environment_variables:
  UV_HTTP_TIMEOUT:
    default: "60000"

runtimes:
  - editor: JupyterLab
    kernel: Python 3.10
    edition: Standard

tasks:
  - type: create_job
    name: Download/Install Project Dependencies
    entity_label: install_dependencies
    script: scripts/01_install_base.py
    kernel: python3
    cpu: 2
    memory: 4

  - type: run_job
    entity_label: install_dependencies

  - type: start_application
    name: RagStudio
    subdomain: ragstudio
    script: scripts/startup_app.py
    cpu: 2
    memory: 4
```

### Task Types Available in `.project-metadata.yaml`

| Task Type | Purpose |
|-----------|---------|
| `run_session` | Execute a script in an interactive session |
| `create_job` | Define a job (does not run it) |
| `run_job` | Run a previously-created job by `entity_label` |
| `create_model` | Register a model endpoint |
| `build_model` | Build a model image |
| `deploy_model` | Deploy a model for serving |
| `start_application` | Create and start a long-running web application |
| `run_experiment` | Run an MLflow experiment |

### Runtime Constraints

- Same as the underlying Project + Application constraints
- Tasks run sequentially; a failure halts subsequent tasks
- The `runtimes` block specifies which ML Runtime image to use
- Environment variables can be prompted from the user at deploy time (`required: true`)
- Resource allocation (CPU, memory, GPU) is per-task

### Deploy Mode: `AGENT_STUDIO_DEPLOY_MODE=amp`

When an AMP is deployed in the traditional way (from source code via the AMP catalog), the deploy mode is `amp`. The code is cloned into `/home/cdsw/`, dependencies are installed at runtime, and the application runs from source.

### Concrete Example (Synthetic Data Studio)

From `/home/rch/local/src/cldr/CAI_AMP_Synthetic_Data_Studio/.project-metadata.yaml`:

```yaml
name: Synthetic Data Studio

environment_variables:
  AWS_ACCESS_KEY_ID:
    default: null
  OPENAI_API_KEY:
    default: null

runtimes:
  - editor: JupyterLab
    kernel: Python 3.10
    edition: Standard

tasks:
  - type: create_job
    name: Synthetic_data_base_job
    entity_label: synthetic_data_job_template
    script: app/text_examples.py
    cpu: 1
    memory: 2

  - type: create_job
    name: Build Client Application
    entity_label: build_client_app
    script: build/build_client.py
    cpu: 2
    memory: 4

  - type: run_job
    entity_label: build_client_app

  - type: start_application
    name: Synthetic Data Studio
    subdomain: synthetic-data-generator
    script: build/start_application.py
    cpu: 2
    memory: 8
```

Pattern: Creates template jobs (that serve as "base" for spawning future jobs programmatically), builds a client app, then starts the application.

---

## 4. Studio (AI Studio / Composable Runtime)

### What Is It?

A **Studio** is the newest and most sophisticated deployment modality. It is a **pre-built ML Runtime Docker image** that ships with all code, dependencies, and frontend already compiled. Instead of cloning source and installing at deploy time (like an AMP), the Studio runs from a containerized image.

There are currently four Studios in Cloudera AI (all in Technical Preview as of mid-2025):
1. **RAG Studio** -- Build RAG applications
2. **Fine Tuning Studio** -- Fine-tune LLMs with PEFT
3. **Synthetic Data Studio** -- Generate synthetic datasets
4. **Agent Studio** -- Design and deploy agentic workflows

### How Is It Deployed?

**As a pre-built ML Runtime image:**

The Docker image is built in CI and registered as a custom ML Runtime. The key differentiators from an AMP deployment are:

```dockerfile
# From CAI_STUDIO_AGENT/Dockerfile
FROM docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5

ENV AGENT_STUDIO_DEPLOY_MODE=runtime    # <-- NOT "amp"
ENV IS_COMPOSABLE=true                   # <-- Composable mode
ENV APP_DIR=/studio_app                  # <-- Code lives at /studio_app, not /home/cdsw
ENV APP_DATA_DIR=/home/cdsw/agent-studio # <-- User data lives here
ENV PATH="/studio_app/.venv/bin:$PATH"

LABEL com.cloudera.ml.runtime.edition="Agent Studio"
LABEL com.cloudera.ml.runtime.kernel="Agent Studio"
```

**Key environment variables that distinguish Studio mode:**

| Variable | AMP Mode | Studio/Runtime Mode |
|----------|----------|---------------------|
| `AGENT_STUDIO_DEPLOY_MODE` | `amp` | `runtime` |
| `IS_COMPOSABLE` | `false` (or unset) | `true` |
| `APP_DIR` | `/home/cdsw` | `/studio_app` (read-only, in image) |
| `APP_DATA_DIR` | `/home/cdsw` | `/home/cdsw/<studio-name>` |

### The IS_COMPOSABLE Pattern

When `IS_COMPOSABLE=true`, the Studio operates in a split-directory model:

- **`APP_DIR`** (`/studio_app`): Read-only application code baked into the Docker image. Contains source, compiled frontend, venvs.
- **`APP_DATA_DIR`** (`/home/cdsw/agent-studio`): Writable user data directory on the NFS filesystem. Contains databases, uploaded files, project defaults.

This pattern is pervasive across all startup scripts. Example from Agent Studio's `run-app.py`:

```python
is_composable = os.getenv("IS_COMPOSABLE", "false").lower() == "true"

if is_studio:
    app_data_dir = "/home/cdsw/agent-studio" if is_composable else "/home/cdsw"

app_dir = os.getenv("APP_DIR", "/home/cdsw/agent-studio") if is_composable else "/home/cdsw"
```

### Runtime Constraints

| Constraint | Detail |
|------------|--------|
| Base image | `ml-runtime-pbj-workbench-python3.10-standard` |
| Python | 3.10 (baked into image) |
| Node.js | v22 via nvm (baked into image) |
| Package manager | uv (baked into image) |
| Code location | `/studio_app` (read-only) |
| Data location | `/home/cdsw/<studio-name>` (writable, NFS) |
| Dependencies | Pre-installed in `.venv` within the image |
| Frontend | Pre-built Next.js in `.next/` within the image |
| Port | `CDSW_APP_PORT` (standard CML application port) |
| Multi-process | gRPC server + workflow runners + ops server + Next.js frontend |

### Studio Internal Architecture

Agent Studio, for example, runs multiple concurrent processes within a single CML Application pod:

1. **Next.js frontend** (serves UI on `CDSW_APP_PORT`)
2. **gRPC backend server** (port 50051, application logic)
3. **N workflow runner processes** (ports 51000+, configurable via `AGENT_STUDIO_NUM_WORKFLOW_RUNNERS`)
4. **Phoenix ops server** (port 50052, observability/tracing)

### RENDER_MODE: Studio vs Workflow

Agent Studio introduces a `AGENT_STUDIO_RENDER_MODE` variable:

- `studio` (default): Shows the full Studio UI for designing workflows
- `workflow`: Shows a deployed workflow's dedicated UI

When deploying a workflow from Agent Studio, it creates a *new* CML Application with `RENDER_MODE=workflow`, reusing the same Docker image/code but rendering a different view.

### Concrete Example (Agent Studio Dockerfile + Metadata)

The `.project-metadata.yaml` still exists for the AMP deployment path:

```yaml
name: Agent Studio
runtimes:
  - editor: PBJ Workbench
    kernel: Python 3.10
    edition: Standard

tasks:
  - type: run_session
    name: Pre installation check
    script: startup_scripts/pre_install_check.py

  - type: run_session
    name: Ensure uv Package Manager
    script: startup_scripts/ensure-uv-package-manager.py

  - type: run_session
    name: Install Dependencies
    script: startup_scripts/install-dependencies.py

  - type: run_session
    name: Initialize Project Defaults
    script: startup_scripts/uv_initialize-project-defaults.py

  - type: start_application
    name: "Agent Studio"
    subdomain: cai-agent-studio
    script: startup_scripts/run-app.py
    bypass_authentication: true
    is_embedded: true
    cpu: 4
    memory: 16
```

But the Dockerfile sets `AGENT_STUDIO_DEPLOY_MODE=runtime`, meaning when deployed as a Runtime image, the AMP setup tasks are skipped and the pre-built image handles everything.

---

## Comparison Matrix

| Dimension | Project | Application | AMP | Studio |
|-----------|---------|-------------|-----|--------|
| **What** | Git-backed workspace | Long-running web service | One-click project template | Pre-built runtime image |
| **Contains** | Code, data, config | Single web server | Full ML pipeline | Multi-service application |
| **Deployed via** | Manual creation | UI/API within a project | Catalog one-click | Runtime image registration |
| **Entry point** | N/A (interactive) | Python script -> web server | `.project-metadata.yaml` | Dockerfile + YAML |
| **Code location** | `/home/cdsw/` | `/home/cdsw/` | `/home/cdsw/` | `/studio_app/` (read-only) |
| **Dependency install** | Manual | Manual | Automated via tasks | Pre-baked in image |
| **Lifecycle** | Persistent | Start/Stop/Restart | Create-once | Start/Stop/Restart |
| **Timeout** | Sessions timeout | Never (manual stop) | Tasks complete | Never (manual stop) |
| **Port binding** | N/A | `CDSW_APP_PORT` | `CDSW_APP_PORT` (for apps) | `CDSW_APP_PORT` |
| **GPU support** | Via runtime edition | Yes, per-app config | Yes, per-task config | Yes, per-task config |
| **is_embedded** | N/A | Optional | Optional | Typically true |
| **IS_COMPOSABLE** | false | false | false | true |
| **Multi-process** | Single session | Single web server | Varies | gRPC + workers + frontend |

---

## Deployment Evolution Path

```
Project (manual, interactive)
  |
  v
Application (persistent web service within a project)
  |
  v
AMP (automated project creation + application deployment from source)
  |
  v
Studio (pre-built Docker image, composable, multi-service)
```

The Studios represent the latest evolution: from "here's source code, install and run it" (AMP) to "here's a fully-built container, just launch it" (Studio/Runtime).

Notably, Agent Studio supports **dual-mode deployment**: the same codebase can be deployed as an AMP (from source, `DEPLOY_MODE=amp`) or as a Studio (from pre-built image, `DEPLOY_MODE=runtime`). The code paths diverge at runtime based on these environment variables.

---

## BDD Scenarios for Validating Deployment Readiness

### Scenario 1: Project Readiness

```gherkin
Feature: CML Project Deployment Readiness

  Scenario: Project has valid metadata for AMP deployment
    Given a Git repository with a ".project-metadata.yaml" in the root
    And the YAML contains required fields: name, description, author, specification_version
    And the "runtimes" block specifies a valid ML Runtime
    When the repository is registered as an AMP source
    Then the AMP catalog should list the project
    And one-click deployment should succeed

  Scenario: Project files are accessible in the workspace
    Given a CML Project has been created
    When a session is started with the configured runtime
    Then "/home/cdsw/" should contain the project files
    And the Python kernel should match the runtime specification
```

### Scenario 2: Application Readiness

```gherkin
Feature: CML Application Deployment Readiness

  Scenario: Application binds to the correct port
    Given an application entry script exists at the configured path
    When the script is executed in a CML Application context
    Then a web server should bind to "127.0.0.1" on port "$CDSW_APP_PORT"
    And the server should respond to HTTP GET "/" with status 200

  Scenario: Application subdomain is accessible
    Given an application is configured with subdomain "my-app"
    And the application status is "Running"
    When a user navigates to "https://my-app.<workspace-domain>"
    Then the application UI should load successfully
    And authentication should be enforced if bypass_authentication is false

  Scenario: Application survives restart
    Given a running CML Application
    When the application process exits unexpectedly
    Then the startup script should restart the process automatically
    And the application should return to "Running" status
```

### Scenario 3: AMP Readiness

```gherkin
Feature: AMP Deployment Readiness

  Scenario: AMP tasks execute sequentially
    Given a .project-metadata.yaml with tasks:
      | type             | entity_label      |
      | create_job       | install_deps      |
      | run_job          | install_deps      |
      | start_application| (inline)          |
    When the AMP is deployed via one-click
    Then the "create_job" task should complete before "run_job"
    And the "run_job" task should complete before "start_application"
    And each task should use the specified CPU/memory resources

  Scenario: AMP environment variables are prompted
    Given a .project-metadata.yaml with:
      """yaml
      environment_variables:
        API_KEY:
          required: true
          description: "Your API key"
      """
    When a user initiates AMP deployment
    Then the UI should prompt for "API_KEY" before proceeding
    And deployment should not start until all required variables are provided

  Scenario: AMP creates functional application
    Given an AMP with a start_application task
    When AMP deployment completes all tasks
    Then a CML Application should exist with the configured subdomain
    And the application should be in "Running" status
    And the application URL should serve the expected UI
```

### Scenario 4: Studio (Composable Runtime) Readiness

```gherkin
Feature: Studio Runtime Deployment Readiness

  Scenario: Studio Docker image contains all required components
    Given a Studio Docker image built from the Dockerfile
    Then "/studio_app/.venv/" should contain installed Python packages
    And "/studio_app/.next/" should contain the built frontend
    And "/studio_app/.nvm/" should contain Node.js v22
    And "uv" should be available on the PATH
    And environment variable "AGENT_STUDIO_DEPLOY_MODE" should equal "runtime"
    And environment variable "IS_COMPOSABLE" should equal "true"

  Scenario: Studio operates in composable split-directory mode
    Given IS_COMPOSABLE is "true"
    And APP_DIR is "/studio_app"
    And APP_DATA_DIR is "/home/cdsw/agent-studio"
    When the studio application starts
    Then application code should be read from "/studio_app"
    And user data should be written to "/home/cdsw/agent-studio"
    And the database should be created at "/home/cdsw/agent-studio/studio-data/"

  Scenario: Studio multi-process startup
    Given a Studio application with AGENT_STUDIO_RENDER_MODE="studio"
    When the start-app-script.sh executes
    Then a gRPC server should start on port 50051
    And 5 workflow runners should start on ports 51000-51004
    And a Phoenix ops server should start on port 50052
    And the Next.js frontend should start on CDSW_APP_PORT
    And all processes should be managed as a process group

  Scenario: Studio supports dual-mode deployment
    Given the same codebase with both .project-metadata.yaml and Dockerfile
    When deployed as an AMP (DEPLOY_MODE=amp)
    Then code should clone to /home/cdsw/ and install dependencies at runtime
    When deployed as a Runtime image (DEPLOY_MODE=runtime)
    Then code should exist at /studio_app/ pre-installed
    And IS_COMPOSABLE should be true
    And APP_DATA_DIR should be separate from APP_DIR

  Scenario: Studio deploys sub-applications for workflows
    Given Agent Studio is running in "studio" render mode
    When a user deploys a workflow
    Then a new CML Application should be created with RENDER_MODE="workflow"
    And a new CML Model should be created for the workflow endpoint
    And the workflow application should reuse the Studio's runtime image
    And the workflow application subdomain should follow "workflow-<id>" pattern
```

---

## Reference Implementation Files

| File | Purpose |
|------|---------|
| `/home/rch/local/src/cldr/CML_AMP_RAG_Studio/.project-metadata.yaml` | RAG Studio AMP definition |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/.project-metadata.yaml` | Agent Studio AMP definition |
| `/home/rch/local/src/cldr/CML_AMP_LLM_Fine_Tuning_Studio/.project-metadata.yaml` | Fine Tuning Studio AMP definition |
| `/home/rch/local/src/cldr/CML_AMP_Dask_on_CML/.project-metadata.yaml` | Simple AMP (no app, jobs only) |
| `/home/rch/local/src/cldr/CAI_AMP_Synthetic_Data_Studio/.project-metadata.yaml` | Synthetic Data Studio AMP |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/Dockerfile` | Agent Studio runtime image definition |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/bin/start-app-script.sh` | Agent Studio multi-process startup |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/startup_scripts/run-app.py` | Dual-mode (AMP vs Runtime) entry point |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/startup_scripts/startup_utils.py` | IS_COMPOSABLE path resolution |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/studio/deployments/applications.py` | Workflow application creation via cmlapi |
| `/home/rch/local/src/cldr/CAI_STUDIO_AGENT/studio/deployments/targets/workbench.py` | Model + App deployment orchestration |
| `/home/rch/local/src/cldr/CML_AMP_RAG_Studio/scripts/startup_app.py` | RAG Studio entry point (auto-restart pattern) |
| `/home/rch/local/src/cldr/CML_AMP_LLM_Fine_Tuning_Studio/bin/run-app.py` | Fine Tuning Studio IS_COMPOSABLE branching |

---

## Sources

- [Cloudera AI Studios Overview (Technical Preview)](https://docs.cloudera.com/machine-learning/1.5.5/use-ai-studios/topics/ml-ai-studios-overview.html)
- [AMP Project Specification](https://docs.cloudera.com/machine-learning/1.5.4/applied-ml-prototypes/topics/ml-amp-project-spec.html)
- [Cloudera Accelerators for ML Projects Overview](https://docs.cloudera.com/machine-learning/cloud/applied-ml-prototypes/topics/ml-amps-overview.html)
- [Analytical Applications](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-applications-c.html)
- [Web Applications Embedded in Sessions](https://docs.cloudera.com/machine-learning/cloud/projects/topics/ml-embedded-web-apps.html)
- [Cloudera AI Architecture Overview](https://docs.cloudera.com/machine-learning/cloud/architecture-overview/topics/ml-architecture-overview-cml.html)
- [Cloudera AI Product Overview](https://docs.cloudera.com/machine-learning/cloud/product/topics/ml-product-overview.html)
- [Cloudera AI Studios Product Page](https://www.cloudera.com/products/machine-learning/ai-studios.html)
- [Cloudera AMP Catalog](https://cloudera.github.io/Applied-ML-Prototypes/)
- [GitHub: CAI_STUDIO_AGENT](https://github.com/cloudera/CAI_STUDIO_AGENT)
- [CML ML Runtimes Repository](https://github.com/cloudera/ml-runtimes)
- [Accelerating AI Success with Cloudera AMPs](https://www.cio.com/article/4070287/accelerating-ai-success-with-cloudera-amps.html)
