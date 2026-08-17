import { useCallback, useEffect, useState } from "react";

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

export default function WaffleMenu() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<SurfaceItem[]>([]);
  const [emptyMsg, setEmptyMsg] = useState("No peers advertising a primary UI.");

  const load = useCallback(async () => {
    try {
      const r = await fetch("/api/atelier/v1/federation/surfaces");
      const data = await r.json();
      if (!r.ok || data.error) {
        setItems([]);
        setEmptyMsg(data.error || "Federation surfaces unreachable");
        return;
      }
      setItems(data.items || []);
      setEmptyMsg("No peers advertising a primary UI.");
    } catch {
      setItems([]);
      setEmptyMsg("Federation surfaces unreachable");
    }
  }, []);

  useEffect(() => {
    if (open) void load();
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
        {items.length === 0 && <p className="waffle-empty">{emptyMsg}</p>}
      </aside>
    </>
  );
}
