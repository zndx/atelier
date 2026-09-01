import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

interface SurfaceItem {
  project: string;
  title?: string;
  engine_target?: string;
  primary_ui: string;
}

function isIpHost(h: string): boolean {
  if (!h) return false;
  if (h.includes(":")) return true;
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(h);
}

/** LAN IP open → rebase this-host waffle links to that IP (ZT name may not resolve). */
function hrefForBrowser(url: string): string {
  const browse = window.location.hostname;
  if (!url || !isIpHost(browse)) return url;
  try {
    const u = new URL(url, window.location.origin);
    u.hostname = browse;
    if (window.location.protocol) u.protocol = window.location.protocol;
    return u.toString();
  } catch {
    return url;
  }
}

function itemTitle(it: SurfaceItem): string {
  if (it.title) return it.title;
  const p = it.project || "";
  if (!p) return "Peer";
  return p.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Survives Layout remounts so a failed refresh cannot flash empty. */
const lastGood: { items: SurfaceItem[] } = { items: [] };

export default function WaffleMenu() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<SurfaceItem[]>(() => lastGood.items);
  const [emptyMsg, setEmptyMsg] = useState("No peers advertising a primary UI.");
  const [loaded, setLoaded] = useState(lastGood.items.length > 0);

  const load = useCallback(async () => {
    try {
      const r = await fetch("/api/atelier/v1/federation/surfaces", {
        headers: { Accept: "application/json" },
      });
      const ctype = r.headers.get("content-type") || "";
      if (!ctype.includes("application/json")) {
        setLoaded(true);
        if (lastGood.items.length === 0) setEmptyMsg("Federation surfaces unreachable");
        return;
      }
      const data = await r.json();
      if (!r.ok || data.error) {
        setLoaded(true);
        if (lastGood.items.length === 0) {
          setEmptyMsg(data.error || "Federation surfaces unreachable");
        }
        return;
      }
      const next = Array.isArray(data.items) ? (data.items as SurfaceItem[]) : [];
      if (next.length > 0) {
        lastGood.items = next;
        setItems(next);
        setEmptyMsg("No peers advertising a primary UI.");
      } else if (lastGood.items.length === 0) {
        setItems([]);
        setEmptyMsg("No peers advertising a primary UI.");
      }
      setLoaded(true);
    } catch {
      setLoaded(true);
      if (lastGood.items.length === 0) setEmptyMsg("Federation surfaces unreachable");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!open) return;
    void load();
    const id = window.setInterval(() => void load(), 8000);
    return () => window.clearInterval(id);
  }, [open, load]);

  return (
    <>
      <button
        type="button"
        id="waffle-toggle"
        aria-label="Federated apps"
        aria-pressed={open}
        title="Federated primary UIs"
        className={`waffle-toggle${open ? " active" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        <svg className="icon-waffle" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
          <circle cx="5" cy="5" r="1.7" fill="currentColor" />
          <circle cx="12" cy="5" r="1.7" fill="currentColor" />
          <circle cx="19" cy="5" r="1.7" fill="currentColor" />
          <circle cx="5" cy="12" r="1.7" fill="currentColor" />
          <circle cx="12" cy="12" r="1.7" fill="currentColor" />
          <circle cx="19" cy="12" r="1.7" fill="currentColor" />
          <circle cx="5" cy="19" r="1.7" fill="currentColor" />
          <circle cx="12" cy="19" r="1.7" fill="currentColor" />
          <circle cx="19" cy="19" r="1.7" fill="currentColor" />
        </svg>
      </button>
      {createPortal(
        <aside id="waffle-rail" className="waffle-rail" hidden={!open}>
          <div className="waffle-rail-head">
            <span>Federation</span>
            <button type="button" className="waffle-close" aria-label="Close" onClick={() => setOpen(false)}>
              ×
            </button>
          </div>
          <ul className="waffle-list">
            {items.map((it) => (
              <li key={`${it.project}:${it.primary_ui}`}>
                <a
                  href={hrefForBrowser(it.primary_ui)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="waffle-project"
                >
                  {itemTitle(it)}
                </a>
              </li>
            ))}
          </ul>
          {!loaded && items.length === 0 && (
            <p className="waffle-empty">Discovering federation peers…</p>
          )}
          {loaded && items.length === 0 && <p className="waffle-empty">{emptyMsg}</p>}
        </aside>,
        document.body,
      )}
    </>
  );
}
