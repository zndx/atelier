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
subprocess.run([*pip, "install", "-e", "."], check=True)
print("Python dependencies installed")

# Verify atelier is importable
subprocess.run(
    [sys.executable, "-c", "import atelier; print(f'atelier {atelier.__version__}')"],
    check=True,
)

# Install PGlite server dependencies
print("\n--- Installing PGlite ---")
subprocess.run(
    ["bash", "-c", "cd scripts && npm install"],
    check=True,
)
print("PGlite dependencies installed")

# Install Node.js dependencies and build React frontend
print("\n--- Building React UI ---")
subprocess.run(
    ["bash", "-c", "cd ui && npm install && npm run build"],
    check=True,
)
print("Node.js dependencies installed and UI built")

# Install Qdrant binary
print("\n--- Installing Qdrant ---")
subprocess.run(["bash", "scripts/install_qdrant.sh"], check=True)
print("Qdrant installed")

print("\n=== All dependencies installed successfully ===")
