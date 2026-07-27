# Kumo hex sweep — hardcoded AntD-light palette replaced across 17 files

**Date:** 2026-07-27 (follows `153305_keiretsu_adoption_phase1.md`)

~131 hardcoded color sites swept onto kumo tokens (3 parallel sweepers +
manual polish; tsc green; dev server :3000 hot-reloaded fine). Pattern:
`var(--…kumo…)` in CSS contexts; kumo dark-ramp literals where var() can't
resolve (SVG attributes, AntD `twoToneColor`/minimap color computations,
alpha glows); AntD color-props kept to semantic presets.

## Deliberate exceptions (all documented in-code or here)

- **Fixed-dark surfaces**: header chrome (raster logo), Terminal +
  raw-markdown `<pre>` panels (#0d1117 GitHub-dark), Orchestrator hero
  (#001529), terminal status pill (kumo literals readable on fixed dark).
- **Flag colors without kumo hues**: role themes purple #722ed1 (Viz
  Director / parquet) + cyan #13c2c2 (Synth Generator) — the Ægir ACCENT
  flag-color precedent; luminance-match later or accept as flags.
- **No-collapse guards**: `#b7eb8f`/`#ffd591` borders kept where mapping
  would merge two distinct statuses; `consumed` → `--color-kumo-fill` (not
  recessed) to stay distinct from `pending`.
- **XYFlow dot grid** → literal #30363c (SVG attribute context).

## Visual-pass checklist (needs eyes + live backend)

1. AntD `Tag` with custom var() colors (DynamicAgentNode L62): antd forces
   white text on custom-color tags — verify contrast on kumo hues, may
   need preset map instead.
2. MiniMap `maskColor` rgba(0,0,0,0.08) — likely invisible in dark mode.
3. Light mode overall (esp. canvas nodes, dot grid, tint backgrounds).
4. Purple/cyan flags against the ramp; Orchestrator navy hero.
5. Status screen = proof screen (per decision note).
