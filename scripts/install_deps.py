"""Install Atelier dependencies in CAI environment."""

import os
import subprocess
import sys

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

# Ensure ~/.local/bin is on PATH (where pip3 install --user puts binaries)
local_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
os.environ["PATH"] = local_bin + ":" + os.environ.get("PATH", "")

# Ensure uv is available
subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
print("uv installed")

# Install Python dependencies
subprocess.run(["uv", "sync", "--frozen"], check=True)
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
