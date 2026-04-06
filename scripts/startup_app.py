"""Start Atelier on CML — restart loop.

Follows the RAG Studio pattern: wraps the shell orchestrator
in a Python restart loop for resilience.
"""

import os
import subprocess

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

port = os.environ.get("CDSW_APP_PORT", "8090")
env = os.environ.copy()

while True:
    print(f"Starting Atelier on port {port}")
    result = subprocess.run(
        ["bash", "bin/start-app.sh", port],
        env=env,
    )
    print(f"Application exited with code {result.returncode}, restarting...")
