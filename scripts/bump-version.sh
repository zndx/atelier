#!/usr/bin/env bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# bump-version.sh — Bump the Atelier version across all canonical locations.
#
# Usage:
#   scripts/bump-version.sh --minor        # 0.2.0 → 0.3.0
#   scripts/bump-version.sh --patch        # 0.2.0 → 0.2.1
#   scripts/bump-version.sh --major        # 0.2.0 → 1.0.0
#   scripts/bump-version.sh 0.3.0          # explicit version
#
# Files updated:
#   pyproject.toml              (version = "X.Y.Z")
#   src/atelier/__init__.py     (__version__ = "X.Y.Z")
#   .project-metadata.yaml      (prototype_version: X.Y.Z)
#   ui/src/components/Layout.tsx (Atelier vX.Y.Z footer)
#
# service.py and gateway.py read __version__ at runtime — no file edit needed.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── Read current version ─────────────────────────────────────────
CURRENT=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

echo "Current version: $CURRENT"

# ── Compute new version ──────────────────────────────────────────
case "${1:-}" in
  --major) NEW="$((MAJOR + 1)).0.0" ;;
  --minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
  --patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
  "")      echo "Usage: $0 --major|--minor|--patch|X.Y.Z"; exit 1 ;;
  *)       NEW="$1" ;;
esac

echo "New version:     $NEW"

# ── Update files ─────────────────────────────────────────────────
sed -i "s/^version = \"$CURRENT\"/version = \"$NEW\"/" pyproject.toml
sed -i "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" src/atelier/__init__.py
sed -i "s/prototype_version: $CURRENT/prototype_version: $NEW/" .project-metadata.yaml
sed -i "s/Atelier v$CURRENT/Atelier v$NEW/" ui/src/components/Layout.tsx

# ── Verify ───────────────────────────────────────────────────────
echo ""
echo "Updated:"
grep -n "version.*$NEW" pyproject.toml | head -1
grep -n "__version__.*$NEW" src/atelier/__init__.py
grep -n "prototype_version.*$NEW" .project-metadata.yaml
grep -n "Atelier v$NEW" ui/src/components/Layout.tsx

echo ""
echo "Ready to commit. Suggested:"
echo "  git add pyproject.toml src/atelier/__init__.py .project-metadata.yaml ui/src/components/Layout.tsx"
echo "  git commit -m \"chore: bump version to v$NEW\""
echo "  git tag -a v$NEW -m \"v$NEW\""
echo "  git push origin trunk v$NEW"
