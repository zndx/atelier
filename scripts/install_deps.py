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

# Install Python package + dependencies into system Python.
# [viz] = pyarrow, pandas, sentence-transformers, umap (embedding pipeline)
# [agents] = claude-agent-sdk, anthropic (Terminal + smoke test + validation)
print("\n--- Installing Python dependencies ---")
subprocess.run([*pip, "install", "-e", ".[viz,agents]"], check=True)
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
# The official standalone installer replaces the deprecated npm package.
# See: https://github.com/anthropics/claude-code
#
# Falls back to npm install if the standalone installer fails
# (air-gapped environments without curl access to claude.ai).
print("\n--- Installing Claude Code CLI ---")
local_bin = os.path.expanduser("~/.local/bin")
os.makedirs(local_bin, exist_ok=True)

installed = False

# Primary: official standalone installer
try:
    subprocess.run(
        ["bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"],
        check=True,
    )
    # The installer puts claude in ~/.claude/local/bin or ~/.local/bin
    for candidate in [
        os.path.expanduser("~/.claude/local/bin/claude"),
        os.path.join(local_bin, "claude"),
        "/usr/local/bin/claude",
    ]:
        if os.path.exists(candidate):
            # Ensure it's on PATH via ~/.local/bin
            claude_dst = os.path.join(local_bin, "claude")
            if candidate != claude_dst:
                if os.path.islink(claude_dst) or os.path.exists(claude_dst):
                    os.remove(claude_dst)
                os.symlink(candidate, claude_dst)
                print(f"Symlinked {candidate} -> {claude_dst}")
            installed = True
            break
except Exception as e:
    print(f"Standalone installer failed: {e}")
    print("Falling back to npm install...")

# Fallback: npm install (deprecated but works in air-gapped envs)
if not installed:
    subprocess.run(
        ["bash", "-c", nvm_prefix + "npm install -g @anthropic-ai/claude-code"],
        check=True,
    )
    npm_prefix_g = subprocess.run(
        ["bash", "-c", nvm_prefix + "npm prefix -g"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    claude_src = os.path.join(npm_prefix_g, "bin", "claude")
    if not os.path.exists(claude_src):
        found = subprocess.run(
            ["bash", "-c", 'find "$HOME/.nvm" -type f -name claude 2>/dev/null | head -1'],
            capture_output=True, text=True,
        ).stdout.strip()
        if found:
            claude_src = found
    if os.path.exists(claude_src):
        claude_dst = os.path.join(local_bin, "claude")
        if os.path.islink(claude_dst) or os.path.exists(claude_dst):
            os.remove(claude_dst)
        os.symlink(claude_src, claude_dst)
        installed = True

if not installed:
    raise RuntimeError("Claude Code CLI installation failed (both standalone and npm)")

# Verify
subprocess.run([os.path.join(local_bin, "claude"), "--version"], check=True)
print("Claude Code CLI installed")

# Install PGlite server dependencies
print("\n--- Installing PGlite ---")
subprocess.run(
    ["bash", "-c", nvm_prefix + "cd scripts && npm install"],
    check=True,
)
print("PGlite dependencies installed")

# Install Node.js dependencies and build React frontend.
# NODE_OPTIONS --max-old-space-size=4096 gives vite enough heap to bundle
# the Embeddings page (2.9 MB pre-gzip). Default heap on some CAI runtimes
# can be ~512 MB and OOM during terser minify.
print("\n--- Building React UI ---")
subprocess.run(
    ["bash", "-c",
     nvm_prefix
     + "export NODE_OPTIONS='--max-old-space-size=4096' && "
     + "cd ui && npm install && npm run build"],
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
