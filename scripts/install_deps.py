"""Install Atelier dependencies in CML environment."""

import os
import subprocess

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

# Ensure uv is available
subprocess.run(["pip3", "install", "uv"], check=True)
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

print("All dependencies installed successfully.")
