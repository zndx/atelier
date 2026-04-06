# Cloudera AI (CML) Application Deployment Research

## 1. Application Entry Point Configuration

### How It Works

CML "Analytical Applications" are long-running web services hosted within a CML project.
The platform exposes two special environment variables that your entry point script must bind to:

| Variable | Purpose |
|----------|---------|
| `CDSW_APP_PORT` | For applications with write/control capabilities |
| `CDSW_READONLY_PORT` | For read-only applications |

Both ports enforce authentication by default. An admin can optionally allow unauthenticated (public) access.

### Creating an Application

Via the UI: Project > Applications > New Application, then configure:
- **Name** and **Subdomain** (DNS-safe: a-z, 0-9, hyphens)
- **Script**: the entry point file (e.g., `entry.py`)
- **Engine Kernel**: python3, R, etc.
- **Resources**: CPU, memory, GPU
- **Environment Variables**: key-value pairs (app-level overrides project-level)

Via APIv2:
```python
application_request = cmlapi.CreateApplicationRequest(
    name="application_name",
    description="application_description",
    project_id=project_id,
    subdomain="application-subdomain",
    kernel="python3",
    script="entry.py",
    environment={"KEY": "VAL"}
)
app = client.create_application(project_id=project_id, body=application_request)
```

### Entry Point Script Examples

**Flask:**
```python
from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    PORT = os.getenv("CDSW_READONLY_PORT", "8090")
    app.run(host="127.0.0.1", port=int(PORT))
```

**Streamlit** (launcher.py):
```python
!streamlit run app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
```

**Gradio:**
```python
import gradio as gr
import os

# ... build interface ...
demo.launch(
    server_name="127.0.0.1",
    server_port=int(os.getenv("CDSW_APP_PORT"))
)
```

**Generic subprocess pattern** (TensorBoard, Hiplot, etc.):
```python
import os, subprocess

subprocess.call([
    "tensorboard",
    "--logdir=logs/fit",
    "--host", "127.0.0.1",
    "--port", os.environ["CDSW_APP_PORT"]
])
```

Key rules:
- Always bind to `127.0.0.1` (localhost), not `0.0.0.0`
- Always read the port from the environment variable; never hardcode it
- CML natively sets `CDSW_APP_PORT`; do NOT set it yourself in .env when running in CML

---

## 2. Application vs AMP Deployment

### Analytical Application
- A **single long-running web service** deployed within a CML project
- You write the code, configure the entry script, and deploy it manually
- Runs in its own isolated compute instance (no session timeout)
- Suitable for dashboards, model frontends, data visualization tools
- Lifecycle: create, start, stop, restart, delete via UI or API

### AMP (Applied ML Prototype)
- A **complete, pre-built reference project** (end-to-end ML use case)
- One-click deployment from a catalog (Cloudera-provided or custom)
- Includes everything: data pipelines, model training, model deployment, AND applications
- Driven by `.project-metadata.yaml` which defines automated setup tasks
- Tasks run sequentially on import: install deps, run jobs, build/deploy models, start applications
- Created by Cloudera's Fast Forward Labs or community contributors

### When to Use Which

| Scenario | Use |
|----------|-----|
| Custom app serving a model you built | Application |
| Quick-start reference implementation | AMP |
| Sharing a reproducible ML workflow | AMP |
| Single interactive dashboard | Application |
| Full pipeline: ingest -> train -> deploy -> serve | AMP |

---

## 3. .project-metadata.yaml Specification

### Required Top-Level Fields

```yaml
name: "My AMP"                    # string(200) - project name
description: "What it does"       # string(2048) - overview
author: "Your Name"               # string(64) - creator
specification_version: "1.0"      # string(16) - schema version
prototype_version: "1.0"          # string(16) - project version
```

### Optional Top-Level Fields

```yaml
date: "2024-09-10"               # YYYY-MM-DD
shared_memory_limit: 0.5          # GB, default 0.0625

environment_variables:
  MY_VAR:
    default: "some_value"
    description: "Explanation of this variable"
    required: true

feature_dependencies:
  - model_metrics

runtimes:
  - editor: JupyterLab           # or Workbench, PBJ Workbench
    kernel: Python 3.10
    edition: Standard             # or Nvidia GPU
    version: "2021.03"           # optional
    addons:                       # optional
      - Spark 3.2
```

### Task Types

Tasks run sequentially on AMP import.

#### run_session
```yaml
- type: run_session
  name: Install Dependencies
  script: cml/install_dependencies.py
  memory: 5
  cpu: 2
```

#### create_job / run_job
```yaml
- type: create_job
  name: Install Dependencies
  entity_label: install_deps
  script: scripts/01_install_base.py
  cpu: 2
  memory: 4
  kernel: python3

- type: run_job
  entity_label: install_deps
```

#### create_model / build_model / deploy_model
```yaml
- type: create_model
  name: My Model
  entity_label: my_model
  # ... resource config ...

- type: build_model
  entity_label: my_model
  # file_path, function_name, examples ...

- type: deploy_model
  entity_label: my_model
```

#### start_application
```yaml
- type: start_application
  name: Launch Streamlit Application
  subdomain: streamlit              # required, DNS-safe
  script: cml/launch_app.py
  kernel: python3
  cpu: 2
  memory: 4
  gpu: 0
  bypass_authentication: false      # true for public access
  static_subdomain: true            # prevent randomization
  environment_variables:
    TASK_TYPE: START_APPLICATION
```

#### run_experiment
```yaml
- type: run_experiment
  name: My Experiment
  script: experiment.py
```

### Complete Real-World Example

From Cloudera's Object Detection AMP:
```yaml
name: Object Detection Inference Visualized
description: A blog-style application to visualize object detection workflow
author: Cloudera Inc.
specification_version: 1.0
prototype_version: 2.0
date: "2022-04-04"

environment_variables:
  STREAMLIT_SERVER_FILE_WATCHER_TYPE:
    default: poll
    description: >-
      Instruct Streamlit to use polling rather than watchdog file watching.

runtimes:
  - editor: PBJ Workbench
    kernel: Python 3.9
    edition: Standard

tasks:
  - type: run_session
    name: Install Dependencies
    script: cml/install_dependencies.py
    memory: 5
    cpu: 2

  - type: start_application
    name: Launch Streamlit Application
    subdomain: streamlit
    script: cml/launch_app.py
    short_summary: Starting Streamlit application
    cpu: 2
    memory: 4
    environment_variables:
      TASK_TYPE: START_APPLICATION
```

### AMP Folder Convention

```
/
├── 0_session-install-dependencies/
├── 1_job-run-python-job/
├── 2_model-deploy-model/
├── 3_app-run-python-script/
├── .project-metadata.yaml
├── cdsw-build.sh                    # only for CML-native models
└── README.md
```

Pattern: `/<step_number>_<CML_Component>-<task_description>/`

---

## Sources

- [Cloudera: Analytical Applications](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-applications-c.html)
- [Cloudera: AMP Project Specification](https://docs.cloudera.com/machine-learning/1.5.4/applied-ml-prototypes/topics/ml-amp-project-spec.html)
- [Cloudera: Securing Applications](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-securing-applications.html)
- [Cloudera: AMPs Overview](https://docs.cloudera.com/machine-learning/cloud/applied-ml-prototypes/topics/ml-amps-overview.html)
- [GitHub: CML AMP Community Template](https://github.com/cloudera/CML_Community_AMP_Template)
- [GitHub: Streamlit on CML](https://github.com/cloudera/CML_AMP_Streamlit_on_CML)
- [GitHub: Object Detection AMP](https://github.com/cloudera/CML_AMP_Object_Detection_Inference)
- [GitHub: RAG Studio AMP](https://github.com/cloudera/CML_AMP_RAG_Studio)
- [GitHub: Running Custom Applications](https://github.com/fastforwardlabs/running-custom-applications)
- [GitHub: Minimal Streamlit CML](https://github.com/fastforwardlabs/cml_streamlit)
- [Cloudera: Flask Example](https://docs.cloudera.com/machine-learning/1.5.5/projects/topics/ml-projects-embedded-web-applications-example-flask-application.html)
- [Cloudera Community: Running Custom Applications](https://community.cloudera.com/t5/Community-Articles/Running-custom-applications-in-CDSW-CML/ta-p/292388)
