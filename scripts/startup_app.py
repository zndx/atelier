"""Start Atelier on CML — restart loop with exponential backoff.

Follows the RAG Studio pattern: wraps the shell orchestrator
in a Python restart loop for resilience. Backs off on repeated
failures to avoid crash-looping on persistent errors.
"""

import os
import subprocess
import time

root_dir = "/home/cdsw"
if os.getenv("IS_COMPOSABLE", ""):
    root_dir = "/home/cdsw/atelier"
os.chdir(root_dir)

port = os.environ.get("CDSW_APP_PORT", "8090")
env = os.environ.copy()

INITIAL_DELAY = 5
MAX_DELAY = 60
HEALTHY_THRESHOLD = 30  # seconds of uptime before resetting backoff

delay = INITIAL_DELAY

while True:
    print(f"Starting Atelier on port {port}")
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "bin/start-app.sh", port],
        env=env,
    )
    uptime = time.monotonic() - start

    if uptime >= HEALTHY_THRESHOLD:
        # Ran long enough to have started successfully — reset backoff
        delay = INITIAL_DELAY
    else:
        delay = min(delay * 2, MAX_DELAY)

    print(
        f"Application exited after {uptime:.0f}s (code {result.returncode}), "
        f"backing off {delay}s..."
    )
    time.sleep(delay)
