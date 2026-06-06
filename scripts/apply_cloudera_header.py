"""apply_cloudera_header.py — stamp Cloudera proprietary header on source + docs.

Designed to run as part of cutting a release branch.  Trunk stays unmarked
(no headers in the developer's working tree); the release branch carries the
notice on every shipped artifact destined for CAI.

The script is **idempotent across calendar years**: ``is_already_stamped``
detects the notice by a year-agnostic pattern (:data:`STAMP_RE`), so re-running
never stacks a second header regardless of which year originally stamped a file.

Usage:
    uv run python scripts/apply_cloudera_header.py                # stamp in place (release leaves)
    uv run python scripts/apply_cloudera_header.py --strip        # remove headers (e.g. restore trunk)
    uv run python scripts/apply_cloudera_header.py --dry-run      # preview, no writes (stamp/strip)
    uv run python scripts/apply_cloudera_header.py --check        # release gate: exit 1 if any file is MISSING the header
    uv run python scripts/apply_cloudera_header.py --check-absent # trunk gate: exit 1 if any file HAS the header
    uv run python scripts/apply_cloudera_header.py --verbose      # list every affected path

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
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Wrap width for paragraph reflow inside the header (matches typical Python
# comment width — leaves room for the comment marker + space).
WRAP_WIDTH = 75

# ── EDIT THIS BLOCK WHEN ADJUSTING THE NOTICE ──────────────────────
COPYRIGHT_YEAR = datetime.now().year


def header_paragraphs(year: int) -> tuple[str, ...]:
    """The notice text, parameterized by copyright ``year`` (only the first
    line carries the year; the proprietary paragraph is year-independent)."""
    return (
        f"Copyright (c) {year} Cloudera, Inc.  All rights reserved.",
        "",
        (
            "This file contains material proprietary to Cloudera, Inc., and is "
            "provided to authorized licensees solely for use in connection with "
            "the Cloudera AI (CAI) Application from which it was obtained.  It "
            "may not be copied, modified, redistributed, or used in any other "
            "manner without the express written consent of Cloudera, Inc."
        ),
    )


HEADER_PARAGRAPHS: tuple[str, ...] = header_paragraphs(COPYRIGHT_YEAR)

# Year-agnostic stamp detector: matches the copyright line for ANY 4-digit
# year, so the script is genuinely idempotent across calendar boundaries.
# Re-running in a later year recognizes a file stamped in any prior year and
# never stacks a second header.  (Replaces the old year-keyed SENTINEL +
# hand-maintained LEGACY_SENTINELS scheme, which double-stamped on rollover.)
STAMP_RE = re.compile(r"Copyright \(c\) (\d{4}) Cloudera, Inc\.")
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
    return STAMP_RE.search(head) is not None


def stamped_year(text: str) -> int | None:
    """The copyright year present in the file's head, or None if unstamped."""
    m = STAMP_RE.search("\n".join(text.splitlines()[:30]))
    return int(m.group(1)) if m else None


def _wrapped_lines(paragraphs: tuple[str, ...] = HEADER_PARAGRAPHS) -> list[str]:
    """Reflow ``paragraphs`` at ``WRAP_WIDTH``; keep blank-paragraph separators."""
    out: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            out.append("")
            continue
        out.extend(textwrap.wrap(paragraph, width=WRAP_WIDTH) or [""])
    return out


def render_header(style: str, year: int = COPYRIGHT_YEAR) -> str:
    """Render the header block for ``year``, ending with a single blank line."""
    wrapped = _wrapped_lines(header_paragraphs(year))
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


def _norm_notice(text: str) -> str:
    """Strip comment delimiters/markers and collapse whitespace, so notice text
    can be compared regardless of comment style."""
    for tok in ("<!--", "-->", "/*", "*/", "//", "--", "#", "*"):
        text = text.replace(tok, " ")
    return " ".join(text.split())


def _header_block_len(rest: str, style: str) -> int | None:
    """Char length of the stamped header block at the very start of ``rest``
    (including its trailing blank-line separator), or None if no recognizable
    stamp is there.

    Detects the header's ACTUAL comment style from the text rather than trusting
    the file extension — some files were stamped in a language-correct style that
    differs from the extension table (Markdown with ``#``, CSS with ``/* */``),
    and we must strip exactly what is there.  Tries the byte-exact inverse of
    :func:`render_header` (any year, any style) first; otherwise consumes the
    leading comment block structurally (``<!-- -->``/``/* */`` block, or a
    ``//``/``--``/``#`` line run), consuming only lines whose accumulated text
    stays within the known notice (tolerating an ``...``-abbreviated variant), so
    it never removes a real comment that merely sits next to the notice."""
    year = stamped_year(rest)
    if year is not None:
        for st in (style, HASH, SLASH, DASH, HTML):
            rendered = render_header(st, year)
            if rest.startswith(rendered):
                return len(rendered)

    lines = rest.splitlines(keepends=True)
    if not lines:
        return None
    known = _norm_notice(" ".join(header_paragraphs(year or COPYRIGHT_YEAR)))
    first = lines[0].lstrip()
    marker: str | None = None
    block_closers = {"<!--": "-->", "/*": "*/"}
    opener = next((o for o in block_closers if first.startswith(o)), None)
    if opener is not None:
        end = next((i for i, ln in enumerate(lines) if block_closers[opener] in ln), None)
        if end is None or _norm_notice("".join(lines[: end + 1])).rstrip(". ") not in known:
            return None
        consumed = end + 1
    else:
        marker = next((m for m in ("//", "--", "#") if first.startswith(m)), None)
        if marker is None:
            return None
        last, prev, i = -1, "", 0
        while i < len(lines) and lines[i].lstrip().startswith(marker):
            cand = _norm_notice("".join(lines[: i + 1]))
            if cand.rstrip(". ") not in known:   # next comment line is NOT notice text
                break
            if cand != prev:                      # this line contributed notice text
                last = i
            prev, i = cand, i + 1
        consumed = last + 1                       # drop any trailing empty-comment lines
    if consumed == 0 or "Copyright (c)" not in _norm_notice("".join(lines[:consumed])):
        return None
    nxt = lines[consumed].strip() if consumed < len(lines) else None
    if nxt == "" or (opener is None and nxt == marker):
        consumed += 1   # blank / empty-comment line separating notice from content
    return sum(len(ln) for ln in lines[:consumed])


def strip_file(path: Path, style: str, *, dry_run: bool) -> bool:
    """Remove the header from ``path`` if present.  Returns True if it was (or
    would be) stripped.  Raises if a file is detected as stamped but the block
    can't be located — we fail loudly rather than silently corrupt."""
    text = path.read_text(encoding="utf-8")
    if not is_already_stamped(text):
        return False
    insert_at = find_insertion_offset(text, path)
    prefix, rest = text[:insert_at], text[insert_at:]
    block_len = _header_block_len(rest, style)
    if block_len is None:
        raise ValueError("stamped but header block not found at insertion offset")
    new_text = prefix + rest[block_len:]
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strip", action="store_true",
                      help="remove the header from every stamped file (inverse of stamping)")
    mode.add_argument("--check", action="store_true",
                      help="exit 1 if any tracked file is MISSING the header (release-leaf gate)")
    mode.add_argument("--check-absent", action="store_true",
                      help="exit 1 if any tracked file HAS the header (trunk gate — keep trunk unmarked)")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview without writing (stamp/strip only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every affected path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    files = get_tracked_files(repo_root)

    affected: list[Path] = []   # stamped / stripped / offending (per mode)
    inplace: list[Path] = []    # already in the desired state
    unhandled: list[Path] = []  # no comment style / excluded / unreadable

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
        stamped = is_already_stamped(text)

        if args.check:                       # gate: header must be PRESENT
            (inplace if stamped else affected).append(path)
            continue
        if args.check_absent:                # gate: header must be ABSENT
            (affected if stamped else inplace).append(path)
            continue

        try:
            if args.strip:
                (affected if strip_file(path, style, dry_run=args.dry_run)
                 else inplace).append(path)
            else:
                (affected if stamp_file(path, style, dry_run=args.dry_run)
                 else inplace).append(path)
        except Exception as exc:
            print(f"FAIL: {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
            return 2

    if args.strip:
        verb, inplace_label = ("Would strip" if args.dry_run else "Stripped"), "Already unmarked"
    elif args.check:
        verb, inplace_label = "Missing header", "Already stamped"
    elif args.check_absent:
        verb, inplace_label = "Has header", "Already unmarked"
    else:
        verb, inplace_label = ("Would stamp" if args.dry_run else "Stamped"), "Already stamped"

    for label, n in (("Tracked files", len(files)),
                     ("Eligible", len(files) - len(unhandled)),
                     (inplace_label, len(inplace)),
                     (verb, len(affected)),
                     ("Skipped (no style)", len(unhandled))):
        print(f"{label + ':':<22}{n}")

    if args.verbose and affected:
        print(f"\n{verb}:")
        for p in affected:
            print(f"  • {p.relative_to(repo_root)}")

    if args.check and affected:
        print(f"\nFAIL: {len(affected)} file(s) missing the Cloudera header.", file=sys.stderr)
        return 1
    if args.check_absent and affected:
        print(f"\nFAIL: {len(affected)} file(s) carry the Cloudera header; "
              f"trunk must stay unmarked.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
