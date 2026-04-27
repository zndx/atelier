# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""apply_cloudera_header.py — stamp Cloudera proprietary header on source + docs.

Designed to run as part of cutting a release branch.  Trunk stays unmarked
(no headers in the developer's working tree); the release branch carries the
notice on every shipped artifact destined for CAI.

The script is **idempotent**: re-running on an already-stamped file is a no-op,
and re-running across calendar years skips files stamped in any recognized
prior year (extend ``LEGACY_SENTINELS`` as needed).

Usage:
    uv run python scripts/apply_cloudera_header.py             # stamp in place
    uv run python scripts/apply_cloudera_header.py --dry-run   # preview, no writes
    uv run python scripts/apply_cloudera_header.py --check     # CI gate; exit 1 if any tracked file is missing the header
    uv run python scripts/apply_cloudera_header.py --verbose   # list every changed/eligible path

Comment styles per file extension:
    .py .sh .bash .conf .yaml .yml .toml .feature .nix .ttl .rq .rego  → ``#``
    .ts .tsx .js .jsx .mjs .cjs .css .scss .proto                       → ``//``
    .sql                                                                → ``--``
    .md                                                                 → ``<!-- -->``

Files without a known style are skipped (e.g., ``.json`` has no comment syntax,
``.csv`` is data).  Generated proto stubs, lock files, license docs, and
external/ submodules are excluded explicitly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Wrap width for paragraph reflow inside the header (matches typical Python
# comment width — leaves room for the comment marker + space).
WRAP_WIDTH = 75

# ── EDIT THESE TWO BLOCKS WHEN ADJUSTING THE NOTICE ──────────────────────
COPYRIGHT_YEAR = datetime.now().year

HEADER_PARAGRAPHS: tuple[str, ...] = (
    f"Copyright (c) {COPYRIGHT_YEAR} Cloudera, Inc.  All rights reserved.",
    "",
    (
        "This file contains material proprietary to Cloudera, Inc., and is "
        "provided to authorized licensees solely for use in connection with "
        "the Cloudera AI (CAI) Application from which it was obtained.  It "
        "may not be copied, modified, redistributed, or used in any other "
        "manner without the express written consent of Cloudera, Inc."
    ),
)

# Sentinel: any occurrence in the file's first 30 lines marks it as
# already-stamped (idempotency guard).
SENTINEL = f"Copyright (c) {COPYRIGHT_YEAR} Cloudera, Inc."

# Older-year sentinels — add prior years here when re-stamping across years
# so previously-stamped releases are not re-stamped a second time.
LEGACY_SENTINELS: tuple[str, ...] = (
    "Copyright (c) 2025 Cloudera, Inc.",
)
# ─────────────────────────────────────────────────────────────────────────


HASH = "hash"      # # ...
SLASH = "slash"    # // ...
DASH = "dash"      # -- ...
HTML = "html"      # <!-- ... -->


EXT_STYLE: dict[str, str] = {
    ".py": HASH,
    ".sh": HASH,
    ".bash": HASH,
    ".conf": HASH,
    ".yaml": HASH,
    ".yml": HASH,
    ".toml": HASH,
    ".feature": HASH,
    ".nix": HASH,
    ".ini": HASH,
    ".cfg": HASH,
    ".ttl": HASH,
    ".rq": HASH,
    ".rego": HASH,
    ".ts": SLASH,
    ".tsx": SLASH,
    ".js": SLASH,
    ".jsx": SLASH,
    ".mjs": SLASH,
    ".cjs": SLASH,
    ".css": SLASH,
    ".scss": SLASH,
    ".proto": SLASH,
    ".sql": DASH,
    ".md": HTML,
}


# Filenames without a dotted extension treated as a particular style.
NAME_STYLE: dict[str, str] = {
    "justfile": HASH,
    "Justfile": HASH,
    "Dockerfile": HASH,
    "Makefile": HASH,
    ".env.example": HASH,
}


# Tracked path prefixes to exclude (matched against repo-relative path).
EXCLUDE_PREFIXES: tuple[str, ...] = (
    "external/",                # third-party submodules
    ".claude/",                 # local Claude Code config — not shipped
    ".github/",                 # CI; opt-in separately if you want headers there
    "data/",                    # synth datasets (CSV — no comment syntax anyway)
    "build/",                   # generated artifacts (also gitignored)
    "ui/dist/",                 # built UI bundle
    "ui/node_modules/",
    "policy/",                  # rego policies — small surface; skip for now
)


# Specific tracked files to skip.
EXCLUDE_FILES: set[str] = {
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
    "NOTICE.md",
    "NOTICE.txt",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".dockerignore",
    ".editorconfig",
    ".envrc",
    ".sops.yaml",
    "devenv.lock",
    "uv.lock",
    "ui/pnpm-lock.yaml",
    "ui/package-lock.json",
    "ui/tsconfig.tsbuildinfo",
    # Generated proto stubs.
    "src/atelier/proto/atelier_pb2.py",
    "src/atelier/proto/atelier_pb2.pyi",
    "src/atelier/proto/atelier_pb2_grpc.py",
}


def get_tracked_files(repo_root: Path) -> list[Path]:
    """List all tracked files via ``git ls-files``."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        repo_root / line
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def style_for(path: Path, repo_root: Path) -> str | None:
    """Return comment style for ``path`` or None if it should not be stamped."""
    rel = path.relative_to(repo_root)
    rel_str = str(rel)

    for prefix in EXCLUDE_PREFIXES:
        if rel_str.startswith(prefix):
            return None
    if rel_str in EXCLUDE_FILES or rel.name in EXCLUDE_FILES:
        return None
    if rel.name in NAME_STYLE:
        return NAME_STYLE[rel.name]
    if rel.suffix in EXT_STYLE:
        return EXT_STYLE[rel.suffix]
    return None


def is_already_stamped(text: str) -> bool:
    head = "\n".join(text.splitlines()[:30])
    if SENTINEL in head:
        return True
    return any(legacy in head for legacy in LEGACY_SENTINELS)


def _wrapped_lines() -> list[str]:
    """Reflow ``HEADER_PARAGRAPHS`` at ``WRAP_WIDTH``; keep blank-paragraph separators."""
    out: list[str] = []
    for paragraph in HEADER_PARAGRAPHS:
        if not paragraph:
            out.append("")
            continue
        out.extend(textwrap.wrap(paragraph, width=WRAP_WIDTH) or [""])
    return out


def render_header(style: str) -> str:
    """Render the header block, ending with a single blank line."""
    wrapped = _wrapped_lines()
    if style == HASH:
        body = "\n".join(("# " + line).rstrip() if line else "#" for line in wrapped)
        return body + "\n\n"
    if style == SLASH:
        body = "\n".join(("// " + line).rstrip() if line else "//" for line in wrapped)
        return body + "\n\n"
    if style == DASH:
        body = "\n".join(("-- " + line).rstrip() if line else "--" for line in wrapped)
        return body + "\n\n"
    if style == HTML:
        body = "\n".join(wrapped)
        return f"<!--\n{body}\n-->\n\n"
    raise ValueError(f"Unknown style: {style}")


def find_insertion_offset(text: str, path: Path) -> int:
    """For executables and Python, return the byte offset *after* the shebang
    (and PEP-263 encoding declaration for .py).  Otherwise 0.
    """
    if path.suffix not in (".py", ".sh", ".bash", ".mjs", ".cjs", ".js"):
        return 0
    lines = text.splitlines(keepends=True)
    if not lines:
        return 0
    idx = 0
    if lines[0].startswith("#!"):
        idx = 1
    if path.suffix == ".py" and idx < len(lines):
        if lines[idx].startswith("#") and (
            "coding:" in lines[idx] or "coding=" in lines[idx]
        ):
            idx += 1
    if idx == 0:
        return 0
    return sum(len(line) for line in lines[:idx])


def stamp_file(path: Path, style: str, *, dry_run: bool) -> bool:
    """Stamp ``path`` if it is missing the header.  Returns True if (would be) stamped."""
    text = path.read_text(encoding="utf-8")
    if is_already_stamped(text):
        return False
    header = render_header(style)
    insert_at = find_insertion_offset(text, path)
    new_text = text[:insert_at] + header + text[insert_at:]
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="preview without writing")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any tracked file is missing the header (CI gate)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every changed or would-change path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    files = get_tracked_files(repo_root)

    stamped: list[Path] = []
    already: list[Path] = []
    unhandled: list[Path] = []

    for path in files:
        if not path.is_file():
            continue
        style = style_for(path, repo_root)
        if style is None:
            unhandled.append(path)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            unhandled.append(path)
            continue
        if is_already_stamped(text):
            already.append(path)
            continue
        if args.check:
            stamped.append(path)
            continue
        try:
            stamp_file(path, style, dry_run=args.dry_run)
            stamped.append(path)
        except Exception as exc:
            print(f"FAIL: {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
            return 2

    eligible = len(files) - len(unhandled)
    print(f"Tracked files:        {len(files)}")
    print(f"Eligible (stampable): {eligible}")
    print(f"Already stamped:      {len(already)}")
    verb = "Would stamp" if (args.dry_run or args.check) else "Stamped"
    print(f"{verb}:          {len(stamped)}")
    print(f"Skipped (no style):   {len(unhandled)}")

    if args.verbose:
        if stamped:
            print("\nFiles to stamp:" if args.dry_run or args.check else "\nFiles stamped:")
            for p in stamped:
                print(f"  + {p.relative_to(repo_root)}")
        if unhandled and args.verbose:
            print("\nSkipped (no comment style or excluded):")
            for p in unhandled[:25]:
                print(f"  - {p.relative_to(repo_root)}")
            if len(unhandled) > 25:
                print(f"  ... and {len(unhandled) - 25} more")

    if args.check and stamped:
        print(
            f"\nFAIL: {len(stamped)} file(s) missing the Cloudera header.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
