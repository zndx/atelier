"""Install Atelier dependencies in CAI environment.

Installs directly into system Python (no virtualenv) so that the
Application session can find packages without venv activation.
This matches the pattern used by RAG Studio and Fine Tuning Studio.
"""

import os
import subprocess
import sys

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

pip = [sys.executable, "-m", "pip"]

# Install Python package + dependencies into system Python
subprocess.run([*pip, "install", "-e", "."], check=True)
print("Python dependencies installed")

# Install Node.js dependencies and build React frontend
subprocess.run(
    ["bash", "-c", "cd ui && npm install && npm run build"],
    check=True,
)
print("Node.js dependencies installed and UI built")

# Install Qdrant binary
subprocess.run(["bash", "scripts/install_qdrant.sh"], check=True)
print("Qdrant installed")

print("All dependencies installed successfully.")
