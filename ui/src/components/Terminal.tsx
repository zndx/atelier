import { useEffect, useRef, useState, type CSSProperties } from "react";
import { ensureGhostty, type GhosttyTerminal } from "../lib/ghostty";

interface TerminalProps {
  style?: CSSProperties;
  sessionId?: string;
}

type ConnectionStatus = "connecting" | "connected" | "disconnected";

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
      if (state.disposed || !el || state.term) return;
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

    el.style.color = "#c9d1d9";
    el.style.fontFamily = "monospace";
    if (!window.__ghostty) {
      el.style.padding = "16px";
      el.textContent = "Loading terminal…";
    }

    void ensureGhostty()
      .then(() => {
        if (state.disposed) return;
        el.textContent = "";
        el.style.padding = "4px 0 0 4px";
        initTerminal();
      })
      .catch(() => {
        if (state.disposed || !el) return;
        el.style.padding = "16px";
        el.textContent = "Failed to load ghostty-web terminal.";
      });

    return () => {
      state.disposed = true;
      if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
      if (state.ws) {
        state.ws.onclose = null;
        state.ws.close();
        state.ws = null;
      }
      if (state.ro) {
        state.ro.disconnect();
        state.ro = null;
      }
      if (state.fitAddon) {
        state.fitAddon.dispose();
        state.fitAddon = null;
      }
      if (state.term) {
        state.term.dispose();
        state.term = null;
      }
    };
  }, [explicitSessionId]);

  // Status pill overlays the fixed-dark terminal surface (#0d1117 in both
  // modes), so kumo dark-ramp literals — theme vars would flip dark-on-dark
  // in light mode.
  const statusColor =
    connStatus === "connected"
      ? "#4ec491"
      : connStatus === "connecting"
        ? "#d99d54"
        : "#f28881";

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
