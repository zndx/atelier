// WebSocket helpers for the two gateway channels:
//   /ws/terminal/{session_id}  — Claude Agent SDK terminal (ghostty)
//   /ws/orchestration          — live agent-loop events for the canvas

export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

export function terminalWsUrl(sessionId: string): string {
  return wsUrl(`/ws/terminal/${sessionId}`);
}

export interface OrchestrationEvent {
  type:
    | "topology_reset"
    | "agent_spawned"
    | "agent_reasoning"
    | "agent_tool_call"
    | "agent_completed"
    | string;
  [k: string]: unknown;
}

// Connect to /ws/orchestration with auto-reconnect. Returns a disposer.
export function connectOrchestration(
  onEvent: (e: OrchestrationEvent) => void,
): () => void {
  let ws: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let delay = 1000;
  let disposed = false;

  const connect = () => {
    if (disposed) return;
    ws = new WebSocket(wsUrl("/ws/orchestration"));
    ws.onopen = () => {
      delay = 1000;
    };
    ws.onmessage = (evt) => {
      try {
        onEvent(JSON.parse(evt.data) as OrchestrationEvent);
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => {
      if (disposed) return;
      timer = setTimeout(() => {
        delay = Math.min(delay * 2, 30000);
        connect();
      }, delay);
    };
  };
  connect();

  return () => {
    disposed = true;
    if (timer) clearTimeout(timer);
    if (ws) {
      ws.onclose = null;
      ws.close();
    }
  };
}
