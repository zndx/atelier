# Dark-mode direction LOCKED — keiretsu on the existing UI, not a parallel frontend

**Date:** 2026-07-23
**Decision (RH):** Atelier's dark mode is the **keiretsu** theme — Ægir's
RH-ratified (2026-07-19) zndx site theme — applied to the **existing AntD
`ui/`** via its CSS token sheet, a `data-mode` root toggle, and an AntD
theme bridge. PR #1's `ui-next/` Tailwind rewrite is **not merged**; it is
demoted to a design reference ("an awkward re-skin" is the anti-pattern:
parallel frontends fork the wire contract, the widget inventory, and the
build pipeline all at once).

## Why keiretsu won

Reviewed 2026-07-23 (~1,300 lines: `theme-keiretsu.css`, `viz/theme.py`,
`lineup_app.py`, `LineupPanel/PanelView/Lineup.tsx`, sweeps/reward apps;
live pipeline smoke-verified end-to-end):

- **A system, not a stylesheet** — stated laws (elevation = lightness,
  never hue; area×chroma budget, accents ≤ ~10%; contrast band 10–13:1
  capped below halation), derived OKLCH monochrome ramp, luminance-matched
  Cloudera hues, mirrored light mode, runtime `--cldr-accent` hook.
- **Token-portable** — adopting it in Atelier keeps every working screen,
  the tested widgets, and the verified API layer. PR #1 replaces all three
  (8 confirmed findings incl. a mistyped wire contract and an unbuildable
  dep; see 2026-07-08 review).
- **It themes the instrument layer** — the part PR #1 never touches:
  bokeh doc Theme + hv renderer theme + post-render `themed()` walk, with
  mode riding the embed session key. Atelier's embedding-atlas view and
  any future served viz slot straight into the same system.
- **The gesture gate is proven** — RH: "works very well in practice ...
  a genuine triumph." Capture-phase wheel interception without
  preventDefault, iPadOS modifier tracking, one-finger-scroll policy.

## Adoption sketch (next implementation pass)

1. Port `theme-keiretsu.css` into `ui/src/styles/` (token sheet verbatim;
   keiretsu stays the single `data-theme`, only `data-mode` flips).
2. `data-mode` toggle + AntD bridge: map kumo tokens into AntD's theme
   config (dark algorithm + token overrides), replacing hardcoded colors.
3. Mirror `aegir/src/aegir/viz/theme.py` for any Atelier-served viz;
   embedding-atlas dark param follows `data-mode`.
4. Fold in PR #1's good ideas as restyles of existing pages (Operate
   dual-pane layout, status pill language) — reference, not merge.

## Upstream intent (org design template)

Drive our patterns into **cldr-design-template** so zndx work informs the
common direction: the template has neither holoviews theming nor a proper
terminal — both hard requirements here. Candidate contributions: the
keiretsu ramp + laws, the hv/bokeh theme bridge, the PanelView gesture
gate, the air-gapped bokeh-serve embed topology, the ghostty terminal
pattern.

## Handoffs

- Ægir running-observations **§20**: adoption recorded + the review's four
  small Ægir-side findings handed over (kumo-positive token gap,
  `__lineupOpen` global collision, `_tap_hook` duplication, ACCENT map
  outside the luminance law).
- PR #1: feedback to the colleague pending RH review (draft on request);
  worktree `~/local/src/zndx/atelier-pr1` + :3001 dev server kept for
  design reference until the salvage pass is done.
