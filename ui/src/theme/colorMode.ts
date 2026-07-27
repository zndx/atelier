/* Canonical: cldr-design-template@db1a423 app/src/theme/colorMode.ts
 * Downstream copy (Atelier) — sync policy: build/ZNDX_CONSOLIDATION_PLAN.md.
 * Do not edit locally; fix upstream, then re-sync. */

/**
 * App color mode: flips `data-mode` on <html>.
 * Site theme (`data-theme`) is independent — see siteTheme.ts
 * (cloudera | keiretsu); tokens adapt via the matching theme-*.css.
 *
 * Aegir-compatible: `applyColorMode` dispatches `cldr:color-mode` and
 * `useColorMode()` exposes the mode to React (embeds re-key on toggle).
 */
import { useSyncExternalStore } from 'react';

export type ColorMode = 'dark' | 'light';

const STORAGE_KEY = 'cldr-color-mode';
const EVENT = 'cldr:color-mode';

export function getColorMode(): ColorMode {
  if (typeof document === 'undefined') return 'dark';
  const attr = document.documentElement.getAttribute('data-mode');
  return attr === 'light' ? 'light' : 'dark';
}

export function applyColorMode(mode: ColorMode): void {
  document.documentElement.setAttribute('data-mode', mode);
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    /* private mode / blocked storage */
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: mode }));
}

/** Restore saved preference before first paint when possible. */
export function initColorMode(): ColorMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') {
      applyColorMode(stored);
      return stored;
    }
  } catch {
    /* ignore */
  }
  const current = getColorMode();
  applyColorMode(current);
  return current;
}

export function toggleColorMode(current: ColorMode): ColorMode {
  const next: ColorMode = current === 'dark' ? 'light' : 'dark';
  applyColorMode(next);
  return next;
}

function subscribe(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}

/** Live color mode as React state — re-renders consumers on toggle. */
export function useColorMode(): ColorMode {
  return useSyncExternalStore(subscribe, getColorMode, () => 'dark' as ColorMode);
}
