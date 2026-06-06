# Nix/devenv in CML — Roadmap

## Context

CML Python runtimes lack Node.js, have pip version constraints, and require
manual binary downloads (Qdrant, etc.). devenv already defines the full stack.
Two paths to bring Nix into CML, both worth pursuing.

## Path 1: nix-portable + proot (CONFIRMED WORKING)

**Status:** Validated 2026-04-06. `nix (Nix) 2.20.6` runs in CML via proot.

```
cdsw@rzce1b6nvzxlq5g7:~$ NP_RUNTIME=proot ~/nix-portable nix --version
Installing git. Disable this by specifying the git executable path with 'NP_GIT'
nix (Nix) 2.20.6
```

- First run: ~15 min (store extraction over NFS, ~350 MB+ in `~/.nix-portable/`)
- Subsequent runs: instant (store persists on NFS)
- proot uses ptrace for syscall interception — CML's seccomp profile allows it

Next steps to validate:
- `nix shell nixpkgs#nodejs_22 --command node --version`
- `nix shell nixpkgs#qdrant --command qdrant --version`
- Test devenv integration: can `devenv shell` work via nix-portable?
- Benchmark proot overhead for PostgreSQL/Qdrant processes

**Risk:** Running long-lived server processes (PostgreSQL, Qdrant) under proot
is untested. May have I/O overhead from syscall interception.

## Path 2: devenv container build → custom CML runtime (production)

**Status:** Roadmap — not yet implemented.

`devenv container build` generates an OCI image from `devenv.nix`. Push to
a container registry, configure as the CML project's custom runtime.

Agent Studio already uses this pattern (`Agent Studio` kernel/edition is a
custom runtime image with all deps baked in).

Benefits:
- Proven pattern (Agent Studio does it)
- Fast startup (no install step needed)
- Same binaries as local dev
- No exotic runtime tricks

Steps to implement:
1. Add `containers` config to `devenv.nix`
2. `devenv container build` → OCI tarball
3. Push to ghcr.io/zndx/atelier-runtime or similar
4. Add custom runtime to `.project-metadata.yaml`
5. Install job becomes minimal (just build UI + download data)

**Tradeoff:** Must rebuild + push image when `devenv.nix` deps change.

## Comparison

| Aspect | nix-portable | Custom runtime |
|--------|-------------|---------------|
| Install time | ~15 min first run (NFS), instant after | Instant (baked in) |
| Maintenance | Self-updating | Rebuild on dep change |
| Complexity | Low (one binary) | Medium (CI pipeline) |
| Reliability | Depends on ptrace | Proven |
| Disk usage | ~1.3 GB in home | In container image |
| Same as local dev | Yes (devenv shell) | Yes (same Nix packages) |
