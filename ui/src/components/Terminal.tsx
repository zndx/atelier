// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { useEffect, useRef, useState, type CSSProperties } from "react";

interface TerminalProps {
  style?: CSSProperties;
  sessionId?: string;
}

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
  const id = crypto.randomUUID?.()
    ?? "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
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

function Terminal({ style, sessionId: explicitSessionId }: TerminalProps) {
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

    function initTerminal() {
      if (state.disposed || !el) return;
      const g = window.__ghostty;
      if (!g) {
        if (window.__ghosttyError) {
          el.textContent = "Failed to load ghostty-web terminal.";
        }
        return;
      }

      const term = new g.Terminal({
        ghostty: g.instance,
        cursorBlink: true,
        fontFamily: 'Menlo, Monaco, "Courier New", monospace',
        fontSize: 13,
        scrollback: 5000,
        theme: {
          background: "#0d1117",
          foreground: "#c9d1d9",
          cursor: "#58a6ff",
          selectionBackground: "rgba(56,139,253,0.4)",
        },
      });

      const fitAddon = new g.FitAddon();
      term.loadAddon(fitAddon);
      term.open(el);
      fitAddon.fit();

      state.term = term;
      state.fitAddon = fitAddon;

      // ResizeObserver for auto-fit
      let resizeTimer: ReturnType<typeof setTimeout>;
      const ro = new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          if (!state.disposed) fitAddon.fit();
        }, 50);
      });
      ro.observe(el);
      state.ro = ro;

      // WebSocket connection
      connect();

      // Wire terminal input to WebSocket
      term.onData((data: string) => {
        if (state.ws?.readyState === WebSocket.OPEN) {
          state.ws.send(JSON.stringify({ type: "input", data }));
        }
      });
    }

    function connect() {
      if (state.disposed) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${proto}//${window.location.host}/ws/terminal/${sessionId}`;
      const ws = new WebSocket(wsUrl);
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
          if (msg.type === "text" && state.term) {
            state.term.write(msg.data);
          }
        } catch {
          // Non-JSON frame — write raw
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
        // onclose fires after onerror — reconnect handled there
      };
    }

    // Initialize ghostty-web
    loadGhosttyScript();
    if (window.__ghostty) {
      initTerminal();
    } else {
      const handler = () => initTerminal();
      window.addEventListener("ghostty-ready", handler, { once: true });
      // Timeout: if WASM doesn't load in 15s, show error
      const timeout = setTimeout(() => {
        if (!window.__ghostty && el) {
          el.style.color = "#c9d1d9";
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

  const statusColor =
    connStatus === "connected"
      ? "#3fb950"
      : connStatus === "connecting"
        ? "#d29922"
        : "#f85149";

  const statusLabel =
    connStatus === "connected"
      ? "connected"
      : connStatus === "connecting"
        ? "connecting..."
        : "disconnected";

  return (
    <div style={{ position: "relative", background: "#0d1117", ...style }}>
      {/* Connection status pill */}
      <div
        style={{
          position: "absolute",
          top: 6,
          right: 10,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          gap: 5,
          fontSize: 11,
          fontFamily: 'Menlo, Monaco, "Courier New", monospace',
          color: statusColor,
          opacity: 0.7,
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: statusColor,
            display: "inline-block",
          }}
        />
        {statusLabel}
      </div>
      {/* Terminal container */}
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          padding: "4px 0 0 4px",
        }}
      />
    </div>
  );
}

export default Terminal;
