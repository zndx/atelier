/** Shared ghostty-web loader. One Promise for the whole session. */

export interface GhosttyTerminal {
  cols: number;
  rows: number;
  open: (el: HTMLElement) => void;
  write: (data: string) => void;
  loadAddon: (addon: unknown) => void;
  onData: (cb: (data: string) => void) => void;
  dispose: () => void;
}

export interface GhosttyGlobal {
  Ghostty: { load: (wasmUrl: string) => Promise<unknown> };
  Terminal: new (opts: Record<string, unknown>) => GhosttyTerminal;
  FitAddon: new () => { fit: () => void; dispose: () => void };
  instance: unknown;
}

declare global {
  interface Window {
    __ghostty?: GhosttyGlobal;
    __ghosttyLoading?: boolean;
    __ghosttyError?: Error;
    __ghosttyPromise?: Promise<GhosttyGlobal>;
  }
}

function injectLoader(): void {
  if (window.__ghostty || window.__ghosttyLoading) return;
  window.__ghosttyLoading = true;
  const s = document.createElement("script");
  s.type = "module";
  s.textContent = [
    "import { Ghostty, Terminal, FitAddon } from '/ghostty/ghostty-web.js';",
    "try {",
    "  const instance = await Ghostty.load('/ghostty/ghostty-vt.wasm');",
    "  window.__ghostty = { Ghostty, Terminal, FitAddon, instance };",
    "} catch (e) {",
    "  console.error('ghostty-web init failed:', e);",
    "  window.__ghosttyError = e;",
    "}",
    "window.dispatchEvent(new CustomEvent('ghostty-ready'));",
  ].join("\n");
  document.head.appendChild(s);
}

/** Start (or join) the one-shot WASM load. Safe to call from main + Terminal. */
export function ensureGhostty(): Promise<GhosttyGlobal> {
  if (window.__ghostty) return Promise.resolve(window.__ghostty);
  if (window.__ghosttyPromise) return window.__ghosttyPromise;

  window.__ghosttyPromise = new Promise((resolve, reject) => {
    const settle = () => {
      if (window.__ghostty) {
        resolve(window.__ghostty);
        return true;
      }
      if (window.__ghosttyError) {
        reject(window.__ghosttyError);
        return true;
      }
      return false;
    };
    if (settle()) return;
    window.addEventListener("ghostty-ready", () => {
      settle();
    }, { once: true });
    // Strict Mode can drop the CustomEvent between unmount and remount.
    const tick = window.setInterval(() => {
      if (settle()) window.clearInterval(tick);
    }, 50);
    injectLoader();
  });

  return window.__ghosttyPromise;
}
