#!/usr/bin/env bash
# build_source_archive.sh — self-contained source tarball for offline CAI
# deployments (i.e. CAI workspaces that cannot reach github.com to git-clone
# the project + submodules at install time).
#
# Bundles:
#   - The main repo at HEAD (use a release branch so headers are stamped and
#     the version is bumped)
#   - The embedding-atlas submodule (required at runtime — its pre-built
#     dist/ is committed to the fork, so no Node toolchain needed in CAI)
#
# Intentionally NOT bundled:
#   - external/hermes-agent — dev-only reference, never installed in CAI
#
# Output: build/release/atelier-{version}.tar.gz

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Read version from pyproject.toml so the archive always matches.
VERSION="$(uv run python -c \
  "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"

PREFIX="atelier-${VERSION}"
OUTDIR="build/release"
TARBALL="${OUTDIR}/${PREFIX}.tar.gz"

mkdir -p "$OUTDIR"

# Stage to a temp dir so we can layer in the submodule before the final tar.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "→ staging main repo @ HEAD"
git archive --format=tar --prefix="${PREFIX}/" HEAD | tar -xf - -C "$WORK"

echo "→ staging external/embedding-atlas submodule"
if [[ ! -e external/embedding-atlas/.git ]]; then
  echo "ERROR: external/embedding-atlas is not initialized." >&2
  echo "       Run: git submodule update --init external/embedding-atlas" >&2
  exit 1
fi
( cd external/embedding-atlas && \
  git archive --format=tar --prefix="${PREFIX}/external/embedding-atlas/" HEAD ) \
  | tar -xf - -C "$WORK"

# Drop hermes-agent placeholder if present (git archive emits an empty dir
# from .gitmodules; we want it gone so the archive doesn't imply it ships).
rm -rf "$WORK/${PREFIX}/external/hermes-agent" 2>/dev/null || true

echo "→ writing ${TARBALL}"
( cd "$WORK" && tar -czf - "${PREFIX}" ) > "$TARBALL"

SIZE="$(du -h "$TARBALL" | cut -f1)"
SHA="$(sha256sum "$TARBALL" | cut -d' ' -f1)"

echo
echo "Archive: ${TARBALL}"
echo "Size:    ${SIZE}"
echo "SHA256:  ${SHA}"
