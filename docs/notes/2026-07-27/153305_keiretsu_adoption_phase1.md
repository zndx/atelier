# Keiretsu adoption phase 1 LANDED — template-canonical tokens + AntD bridge

**Date:** 2026-07-27
**Decision refinement (RH):** PR #1 is ON HOLD (not merged, not closed);
Atelier implements the **cldr-design-template with our enhancements** —
Ægir's UI is the evidentiary standard (both modes, kumo + holoviews
properly integrated). Atelier is thereby the first true downstream consumer
of the consolidation plan's sync policy.

## What landed (ui/ — the existing AntD app, per the 07-23 decision)

- `ui/src/styles/theme-keiretsu.css` — **canonical copy** from
  cldr-design-template@db1a423 with provenance header ("do not edit
  locally; fix upstream, then re-sync").
- `ui/src/theme/colorMode.ts` — canonical copy, same provenance
  (`data-mode` on `<html>`, localStorage persistence, `cldr:color-mode`
  event, `useColorMode()` via useSyncExternalStore — Ægir-compatible).
- `ui/src/theme/kumo.ts` — **Atelier-born AntD bridge** (upstream-feedback
  candidate): TS mirror of the kumo ramp (dark+light) + `kumoAntdTheme()`
  mapping onto AntD v5 ConfigProvider (dark/default algorithm; ramp →
  colorBgLayout/Base/Container, text tiers, borders, status hues,
  brand-as-primary; borderRadius 4). Keiretsu laws stated in the header.
- `ui/src/styles/base.css` — body surfaces on kumo vars + the
  `.atelier-chrome` **deliberate exception**: header stays on the dark
  ramp in BOTH modes because `Cloudera.svg` embeds a raster PNG designed
  for dark chrome (no CSS recolor). Revisit when a light-mode logo lands.
- `ui/src/main.tsx` — `data-theme="keiretsu"` + `initColorMode()` before
  first render (no unthemed flash); imports the two stylesheets.
- `ui/src/App.tsx` — ConfigProvider now `kumoAntdTheme(useColorMode())`;
  stock `#1890ff`/defaultAlgorithm removed.
- `ui/src/components/Layout.tsx` — header on `.atelier-chrome`, nav/gear
  colors from the fixed chrome palette, **mode toggle** (Sun/Moon) beside
  the settings gear.

## Verification

- `tsc --noEmit` green (project config authoritative).
- Vite dev server up on :3000 (backend-independent), serving the app with
  theme-keiretsu.css in the module graph. **Visual pass + per-screen
  hardcoded-hex sweep pending** — gateway/stack was down this session, so
  screens render with dead data; do the sweep with eyes on real screens
  once `devenv up` runs (operator-launched).

## Next

1. Visual verification + hex sweep (Landing `#8c8c8c`, Status inline
   colors, etc.) with the stack up; Status is the proof screen.
2. Feed `kumo.ts` bridge upstream (template `docs/` guidance + optional
   module) per consolidation plan §5.
3. Then the lineup presentation phase (roadmap note 2026-07-23).
