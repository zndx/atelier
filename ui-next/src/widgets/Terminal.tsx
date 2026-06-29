import { useEffect, useRef, useState } from "react";
import { terminalWsUrl } from "../api/ws";
import { StatusDot, type Tone } from "../ui/StatusDot";

type ConnectionStatus = "connecting" | "connected" | "disconnected";

/** ghostty-web WASM types (loaded dynamically) */
interface GhosttyGlobal {
  Ghostty: { load: (wasmUrl: string) => Promise<unknown> };
  Terminal: new (opts: Record<string, unknown>) => GhosttyTerminal;
  FitAddon: new () => { fit: () => void; dispose: () => void };
  instance: unknown;
}

interface GhosttyTerminal {
  cols: number;
  rows: number;
  open: (el: HTMLElement) => void;
  write: (data: string) => void;
  loadAddon: (addon: unknown) => void;
  onData: (cb: (data: string) => void) => void;
  dispose: () => void;
}

declare global {
  interface Window {
    __ghostty?: GhosttyGlobal;
    __ghosttyLoading?: boolean;
    __ghosttyError?: Error;
  }
}

const STORAGE_KEY = "atelier-terminal-session";

function getOrCreateSessionId(explicitId?: string): string {
  if (explicitId) return explicitId;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) return stored;
  const id =
    crypto.randomUUID?.() ??
    "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
    });
  localStorage.setItem(STORAGE_KEY, id);
  return id;
}

function loadGhosttyScript() {
  if (window.__ghostty || window.__ghosttyLoading) return;
  window.__ghosttyLoading = true;
  const s = document.createElement("script");
  s.type = "module";
  s.textContent = [
    "import { Ghostty, Terminal, FitAddon } from '/ghostty/ghostty-web.js';",
    "try {",
    "  const instance = await Ghostty.load('/ghostty/ghostty-vt.wasm');",
    "  window.__ghostty = { Ghostty, Terminal, FitAddon, instance };",
    "} catch(e) {",
    "  console.error('ghostty-web init failed:', e);",
    "  window.__ghosttyError = e;",
    "}",
    "window.dispatchEvent(new CustomEvent('ghostty-ready'));",
  ].join("\n");
  document.head.appendChild(s);
}

export function Terminal({ sessionId: explicitSessionId }: { sessionId?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [connStatus, setConnStatus] = useState<ConnectionStatus>("connecting");
  const setConnStatusRef = useRef(setConnStatus);
  setConnStatusRef.current = setConnStatus;

  const stateRef = useRef<{
    term: GhosttyTerminal | null;
    fitAddon: { fit: () => void; dispose: () => void } | null;
    ws: WebSocket | null;
    ro: ResizeObserver | null;
    reconnectTimer: ReturnType<typeof setTimeout> | null;
    reconnectDelay: number;
    disposed: boolean;
    hasConnectedBefore: boolean;
  }>({
    term: null,
    fitAddon: null,
    ws: null,
    ro: null,
    reconnectTimer: null,
    reconnectDelay: 1000,
    disposed: false,
    hasConnectedBefore: false,
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const state = stateRef.current;
    state.disposed = false;
    const sessionId = getOrCreateSessionId(explicitSessionId);

    function connect() {
      if (state.disposed) return;
      const ws = new WebSocket(terminalWsUrl(sessionId));
      state.ws = ws;
      ws.onopen = () => {
        state.reconnectDelay = 1000;
        setConnStatusRef.current("connected");
        if (state.hasConnectedBefore && state.term) {
          state.term.write("\r\n\x1b[2m(reconnected)\x1b[0m\r\n");
        }
        state.hasConnectedBefore = true;
      };
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "text" && state.term) state.term.write(msg.data);
        } catch {
          if (state.term) state.term.write(evt.data);
        }
      };
      ws.onclose = () => {
        if (state.disposed) return;
        setConnStatusRef.current("disconnected");
        state.reconnectTimer = setTimeout(() => {
          setConnStatusRef.current("connecting");
          state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
          connect();
        }, state.reconnectDelay);
      };
      ws.onerror = () => {
        /* onclose handles reconnect */
      };
    }

    function initTerminal() {
      if (state.disposed || !el) return;
      const g = window.__ghostty;
      if (!g) {
        if (window.__ghosttyError) el.textContent = "Failed to load ghostty-web terminal.";
        return;
      }
      const term = new g.Terminal({
        ghostty: g.instance,
        cursorBlink: true,
        fontFamily: '"JetBrains Mono", Menlo, Monaco, "Courier New", monospace',
        fontSize: 13,
        scrollback: 5000,
        theme: {
          background: "#0a0a0f",
          foreground: "#e5e7eb",
          cursor: "#6366f1",
          selectionBackground: "rgba(99,102,241,0.35)",
        },
      });
      const fitAddon = new g.FitAddon();
      term.loadAddon(fitAddon);
      term.open(el);
      fitAddon.fit();
      state.term = term;
      state.fitAddon = fitAddon;

      let resizeTimer: ReturnType<typeof setTimeout>;
      const ro = new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          if (!state.disposed) fitAddon.fit();
        }, 50);
      });
      ro.observe(el);
      state.ro = ro;

      connect();
      term.onData((data: string) => {
        if (state.ws?.readyState === WebSocket.OPEN) {
          state.ws.send(JSON.stringify({ type: "input", data }));
        }
      });
    }

    loadGhosttyScript();
    if (window.__ghostty) {
      initTerminal();
    } else {
      const handler = () => initTerminal();
      window.addEventListener("ghostty-ready", handler, { once: true });
      const timeout = setTimeout(() => {
        if (!window.__ghostty && el) {
          el.style.color = "#e5e7eb";
          el.style.padding = "16px";
          el.style.fontFamily = "monospace";
          el.textContent = "ghostty-web loading timed out.";
        }
      }, 15000);
      return () => {
        window.removeEventListener("ghostty-ready", handler);
        clearTimeout(timeout);
      };
    }

    return () => {
      state.disposed = true;
      if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
      if (state.ws) {
        state.ws.onclose = null;
        state.ws.close();
      }
      if (state.ro) state.ro.disconnect();
      if (state.fitAddon) state.fitAddon.dispose();
      if (state.term) state.term.dispose();
    };
  }, [explicitSessionId]);

  const tone: Tone =
    connStatus === "connected" ? "green" : connStatus === "connecting" ? "amber" : "red";

  return (
    <div className="relative h-full w-full bg-surface-0">
      <div className="pointer-events-none absolute right-3 top-2 z-10">
        <StatusDot tone={tone} pulse={connStatus !== "connected"} label={connStatus} />
      </div>
      <div ref={containerRef} className="h-full w-full p-1" />
    </div>
  );
}

export default Terminal;
