# Nix/devenv in CML Container Environments: Feasibility Research

**Date:** 2026-04-06
**Goal:** Determine if Nix/devenv can provide reproducible dev environments inside Cloudera Machine Learning (CML) containers.

## Context

CML runs Kubernetes pods with:
- Non-root user `cdsw` (uid typically 8536)
- Persistent home at `/home/cdsw` (NFS-backed, survives session restarts)
- Ephemeral container filesystem (everything outside `/home/cdsw` is lost)
- Standard Kubernetes seccomp/capability restrictions
- No `sudo` or root access

The `devenv.nix` for atelier requires: Python 3.12, Node.js 22, PostgreSQL 16, Qdrant, protobuf, gRPC tools, Kerberos libs, and ~20 other packages.

---

## Q1: Can Nix be installed without root?

**Yes, with caveats.** There are four approaches:

### A. Nix built-in auto store (`~/.local/share/nix/root`)
- Since Nix 2.19+, running a static nix binary with `store auto` will automatically create `~/.local/share/nix/root` and use **Linux user namespaces** to bind-mount it as `/nix` per command.
- **Requires unprivileged user namespaces** (`unshare(CLONE_NEWUSER)`)
- No daemon, no root, no system install needed
- **BLOCKER in CML**: Kubernetes default seccomp profile blocks `unshare`/`CLONE_NEWUSER`. Docker's default seccomp also blocks it. CML almost certainly inherits this restriction.

### B. nix-portable
- Single static binary, downloads from GitHub releases (~26 MB)
- Stores everything in `$HOME/.nix-portable/` (configurable via `NP_LOCATION`)
- Tries bubblewrap first (needs user namespaces), **falls back to proot** if unavailable
- proot uses `ptrace` to intercept syscalls and rewrite `/nix/store` paths — no user namespaces needed
- **proot fallback is the key**: `ptrace` is allowed in default Kubernetes seccomp profiles
- **Performance penalty with proot**: significant overhead on build operations (5-10x slower), but acceptable for running pre-built binaries from cache
- Pre-configured with flakes, nix-command, and a default nixpkgs channel

### C. Determinate Systems installer with `--no-daemon`
- Requires creating `/nix/store` which needs root (or namespace tricks)
- `--init none` skips systemd service setup
- **Still needs root** to create `/nix` directory — not viable in CML

### D. nix-user-chroot
- Static binary that uses user namespaces to mount a user-writable dir as `/nix`
- Same namespace requirement as option A — blocked in CML

**Verdict: nix-portable with proot fallback is the only realistic option for CML.**

---

## Q2: Does Nix need `/nix/store`? Can it be relocated?

**Nix fundamentally assumes `/nix/store`.** All store paths are hashed with this prefix baked in. You cannot simply change it to `/home/cdsw/nix/store`.

However:
- **nix-portable virtualizes it**: the actual data lives in `~/.nix-portable/store/` but appears as `/nix/store` to processes via proot/bwrap
- **Chroot stores** (`~/.local/share/nix/root`) also virtualize it via namespace bind mounts
- **relocatable.nix** can produce deployment scripts that rewrite store paths, but this is for deploying pre-built artifacts, not running nix itself
- Upstream issue [NixOS/nix#10253](https://github.com/NixOS/nix/issues/10253) tracks allowing custom store paths — still open, no resolution

**Verdict: Cannot relocate the store. Must use virtualization (proot or namespaces).**

---

## Q3: Would user namespaces be available in CML?

**Almost certainly not.**

- Docker's default seccomp profile blocks `unshare` with `CLONE_NEWUSER`
- Kubernetes RuntimeDefault seccomp profile inherits Docker's restrictions
- CML runs on managed Kubernetes — administrators are unlikely to relax seccomp for ML workspaces
- Even if the syscall were allowed, CML containers might lack `CAP_SYS_ADMIN`
- The Kubernetes User Namespace feature (KEP-127) maps host UIDs but doesn't grant `unshare` inside the container

**Verdict: Assume user namespaces are unavailable. Plan for proot-only operation.**

---

## Q4: How large is a minimal Nix install? Disk impact?

Estimated sizes for CML `/home/cdsw` (NFS-backed):

| Component | Estimated Size |
|-----------|---------------|
| nix-portable binary | ~26 MB |
| Nix store (nix itself + deps) | ~200 MB |
| Python 3.12 + uv | ~300 MB |
| Node.js 22 + pnpm | ~150 MB |
| PostgreSQL 16 + pgvector | ~200 MB |
| Qdrant | ~100 MB |
| Kerberos, protobuf, gRPC, etc. | ~200 MB |
| Other tools (git, jq, ripgrep...) | ~150 MB |
| **Total estimated** | **~1.3-2 GB** |

With `nix-store --optimise` (hard-linking duplicates): ~25-35% reduction.

CML workspaces typically have 600 GB+ persistent storage, so 2 GB is negligible.

**Caveat**: First install downloads everything from cache.nixos.org. On a slow network or behind a proxy, this could take 10-20 minutes.

---

## Q5: Could we use nix-portable to just run specific packages?

**Yes, this is the most practical approach.** Example:

```bash
# Download nix-portable once
curl -L https://github.com/DavHau/nix-portable/releases/latest/download/nix-portable-x86_64 -o ~/nix-portable
chmod +x ~/nix-portable

# Run a specific package directly
~/nix-portable nix shell nixpkgs#nodejs_22 --command node --version

# Or run an entire devenv shell (if devenv is available)
~/nix-portable nix shell nixpkgs#devenv --command devenv shell
```

The key advantage: nix-portable handles the `/nix/store` virtualization transparently. You write normal nix expressions and everything "just works" (modulo proot overhead).

---

## Q6: Has anyone tried Nix in CML or similar K8s ML platforms?

No direct CML examples found. Related experiences:

- **Nix in CI containers** (GitHub Actions, GitLab CI): Common, but these typically have root access or relaxed seccomp
- **Nix in Docker**: Works with `--privileged` or custom seccomp profiles; default Docker is restrictive
- **nix-portable in restricted environments**: The explicit design goal — works on HPC clusters, shared servers, and locked-down environments via proot fallback
- **devenv in containers**: `devenv container build` generates OCI images — useful for building locally and deploying to CML as a custom runtime, but this is the opposite direction (Nix builds the container, doesn't run inside it)

---

## Q7: What about pre-building artifacts?

### Option A: `devenv container build`
Build an OCI container image locally with all tools baked in, push to a registry, use as CML custom runtime. This is the **most reliable** approach but means CML uses a pre-built image rather than installing on-the-fly.

### Option B: relocatable.nix
Produces self-extracting deployment scripts. Build on a Nix machine, copy the script to CML, run it to extract binaries into a user directory. No Nix needed on target. Uses `dd`, `tar`, `gzip`, `sed`, `ln` — all standard Unix tools.

### Option C: nix-bundle-exe
Bundles ELF executables with all shared libraries into a relocatable directory. No Nix or special runtime needed on target. Good for individual tools, less practical for an entire dev environment.

### Option D: unnix
New tool (alpha) that downloads pre-built binaries from Hydra/Devbox caches without evaluating Nix expressions. Uses bubblewrap for env isolation — **still needs user namespaces**, so blocked in CML.

---

## Practical Assessment

### Approach 1: Custom CML Runtime Image (RECOMMENDED)

**Viability: HIGH**

Use `devenv container build` or a Dockerfile-with-Nix to create a custom CML runtime image containing all tools. Push to your container registry. Configure CML projects to use this runtime.

**Pros:**
- Most reliable, no runtime surprises
- No proot overhead
- Same binaries as local dev
- CML natively supports custom runtime images

**Cons:**
- Requires rebuilding/pushing image when deps change
- Need access to a container registry
- Two-step workflow (build image locally, use in CML)

### Approach 2: nix-portable with proot (EXPERIMENTAL)

**Viability: MEDIUM**

Install nix-portable in `/home/cdsw/`, use proot fallback to run Nix commands. Could power an `install_deps.py` that bootstraps the environment on first session start.

**Pros:**
- Matches "the dream" — `install_deps.py` installs everything
- Uses persistent `/home/cdsw/` so survives session restarts
- No root, no special permissions needed (assuming ptrace allowed)

**Cons:**
- proot performance overhead on builds
- `ptrace` might be blocked by CML's seccomp profile (needs testing)
- Complex debugging if things break
- ~10-20 min first-time setup downloading from cache
- Running services (PostgreSQL, Qdrant) under proot is untested territory

**Critical test:** Run this in a CML session to check:
```bash
# Test if ptrace is available (needed for proot)
python3 -c "import ctypes; ctypes.CDLL('libc.so.6').ptrace(0, 0, 0, 0)"

# Or simply try nix-portable
curl -L https://github.com/DavHau/nix-portable/releases/latest/download/nix-portable-x86_64 -o ~/nix-portable
chmod +x ~/nix-portable
NP_RUNTIME=proot ~/nix-portable nix --version
```

### Approach 3: relocatable.nix Pre-built Bundle (FALLBACK)

**Viability: MEDIUM-HIGH**

Build a relocatable bundle on a Nix-equipped machine. Copy the self-extracting script to CML. Run it to populate `/home/cdsw/.local/atelier-env/`. Add to PATH.

**Pros:**
- No Nix or proot needed at runtime
- Standard Unix tools only
- Fast "install" (just extract)

**Cons:**
- Services (PostgreSQL, Qdrant) may not work — they expect `/nix/store` paths in configs
- Path length issues with store path truncation
- Must rebuild bundle for any dependency change
- Less tested than other approaches

### Approach 4: Hybrid (PRAGMATIC)

**Viability: HIGH**

- Use a custom CML runtime image for heavy/stable deps (Python 3.12, Node.js 22, PostgreSQL 16)
- Use nix-portable or direct binary downloads for lighter/changing tools
- Use `install_deps.py` to verify/install the lightweight tools into `/home/cdsw/`

This splits the problem: base image provides the foundation, `install_deps.py` adds project-specific tools.

---

## Blockers Summary

| Blocker | Severity | Workaround |
|---------|----------|------------|
| No root access | High | nix-portable, custom runtime image |
| No `/nix/store` | High | proot virtualization, custom image |
| User namespaces likely blocked | High | proot fallback, custom image |
| ptrace *might* be blocked | Medium | Test in CML; fall back to custom image |
| Disk space for store | Low | ~2 GB on 600 GB+ NFS is fine |
| Network speed for cache downloads | Low | One-time cost, cached in persistent home |
| proot performance overhead | Medium | Acceptable for running, slow for building |

## Recommendation

**Start with Approach 1 (custom runtime image)** as the reliable foundation. It is proven, requires no exotic runtime tricks, and CML explicitly supports custom runtimes.

**Simultaneously test Approach 2 (nix-portable)** in an actual CML session — the single curl+chmod test above will immediately reveal if proot works. If it does, nix-portable becomes a powerful option for iterating on dependencies without rebuilding container images.

**Approach 4 (hybrid)** is likely where you end up: custom image for the base, nix-portable for flexibility.
