"""Install Atelier dependencies in CAI environment.

Installs directly into system Python (no virtualenv) so that the
Application session can find packages without venv activation.
This matches the pattern used by RAG Studio and Fine Tuning Studio.

Designed to run as a CML Job (create_job/run_job in .project-metadata.yaml).
The job persists in the project and can be re-run from the Jobs tab.
"""

import os
import subprocess
import sys

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

pip = [sys.executable, "-m", "pip"]

print(f"Python: {sys.executable} ({sys.version})")
print(f"Working directory: {os.getcwd()}")

# Install Python package + dependencies into system Python
print("\n--- Installing Python dependencies ---")
subprocess.run([*pip, "install", "-e", ".[viz]"], check=True)
print("Python dependencies installed")

# Verify atelier is importable
subprocess.run(
    [sys.executable, "-c", "import atelier; print(f'atelier {atelier.__version__}')"],
    check=True,
)

# Install Node.js via nvm (RAG Studio pattern)
print("\n--- Installing Node.js ---")
subprocess.run(["bash", "scripts/install_node.sh"], check=True)
print("Node.js installed")

# Source nvm so npm is on PATH for subsequent commands.
# load_nvm.sh deliberately suppresses errors from .bashrc — verify node is available.
nvm_prefix = "source scripts/load_nvm.sh && "
print("\n--- Verifying Node.js is on PATH ---")
subprocess.run(
    ["bash", "-c", nvm_prefix + "node --version && npm --version"],
    check=True,
)

# Ensure git submodules are initialized (CAI clones may skip submodules)
print("\n--- Initializing git submodules ---")
subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
print("Submodules initialized")

# Verify embedding-atlas fork has pre-built dist/
# The fork (external/embedding-atlas) carries committed dist/ artifacts so
# we don't need to run the full workspace build chain on CAI (which requires
# uv, Emscripten, Rust/wasm-bindgen — none available on bare runtimes).
ea_dist = os.path.join("external", "embedding-atlas", "packages", "embedding-atlas", "dist", "react.js")
if os.path.exists(ea_dist):
    print("\n--- embedding-atlas fork: pre-built dist/ present ---")
else:
    print("\n--- WARNING: embedding-atlas dist/ not found; UI build may fail ---")
    print(f"    Expected: {ea_dist}")
    print("    Run the full build locally and commit dist/ to the fork.")

# Install Claude Code CLI (required by claude-agent-sdk at runtime)
# Pin to the version the SDK was built against for compatibility.
print("\n--- Installing Claude Code CLI ---")
subprocess.run(
    ["bash", "-c", nvm_prefix + "npm install -g @anthropic-ai/claude-code@2.1.92"],
    check=True,
)
subprocess.run(
    ["bash", "-c", nvm_prefix + "claude --version"],
    check=True,
)
print("Claude Code CLI installed")

# Install PGlite server dependencies
print("\n--- Installing PGlite ---")
subprocess.run(
    ["bash", "-c", nvm_prefix + "cd scripts && npm install"],
    check=True,
)
print("PGlite dependencies installed")

# Install Node.js dependencies and build React frontend
print("\n--- Building React UI ---")
subprocess.run(
    ["bash", "-c", nvm_prefix + "cd ui && npm install && npm run build"],
    check=True,
)
print("Node.js dependencies installed and UI built")

# Install Qdrant binary
print("\n--- Installing Qdrant ---")
subprocess.run(["bash", "scripts/install_qdrant.sh"], check=True)
print("Qdrant installed")

# Prepare GitTables visualization parquet if source is available
gittables_source = os.environ.get("GITTABLES_EVAL_PARQUET", "")
if gittables_source and os.path.exists(gittables_source):
    print("\n--- Preparing GitTables visualization parquet ---")
    subprocess.run(
        [sys.executable, "scripts/prepare_gittables_sample.py",
         "--input", gittables_source],
        check=True,
    )
    print("GitTables parquet prepared")
elif os.path.exists("data/gittables_sample.parquet"):
    print("\n--- GitTables parquet already exists, skipping preparation ---")
else:
    print("\n--- Skipping GitTables preparation (set GITTABLES_EVAL_PARQUET to enable) ---")

print("\n=== All dependencies installed successfully ===")
