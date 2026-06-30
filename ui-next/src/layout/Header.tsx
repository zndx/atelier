import { NavLink } from "react-router-dom";
import { Boxes, Settings as SettingsIcon } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { getFsmStatus, getStatus } from "../api/client";
import { StatusDot, type Tone } from "../ui/StatusDot";
import { Pill } from "../ui/Pill";
import { cn } from "../ui/cn";
import { fsmLabel, isRunning, stateTone } from "../lib/fsm";

const NAV = [
  { to: "/", label: "Operate", end: true },
  { to: "/workflows", label: "Workflows", end: false },
  { to: "/embeddings", label: "Embeddings", end: false },
  { to: "/agents", label: "Agents", end: false },
  { to: "/status", label: "Status", end: false },
];

function probeTone(ok: boolean | undefined): Tone {
  return ok ? "green" : "red";
}

export function Header() {
  // Header owns the global health poll (5s) — the one place that always
  // renders, so the status dots reflect live gRPC/PG/Qdrant reachability.
  const { data: status } = usePolling(getStatus, 5000);
  const { data: fsm } = usePolling(getFsmStatus, 5000);

  const running = isRunning(fsm?.state);
  const model = status?.config?.agent_model;

  return (
    <header className="flex items-center justify-between gap-4 border-b border-surface-3 bg-surface-1 px-6 py-3">
      {/* Left: brand + nav */}
      <div className="flex items-center gap-6 min-w-0">
        <NavLink to="/" className="flex items-center gap-2.5 shrink-0">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10">
            <Boxes className="h-5 w-5 text-accent" />
          </span>
          <span className="leading-tight">
            <span className="block text-lg font-semibold text-white">Atelier</span>
            <span className="block text-[11px] text-gray-500">Classification Workbench</span>
          </span>
        </NavLink>
        <nav className="hidden items-center gap-1 md:flex">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent/20 text-accent"
                    : "text-gray-400 hover:bg-surface-3 hover:text-gray-200",
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Right: health dots + run state + model + settings */}
      <div className="flex items-center gap-4 shrink-0">
        <div className="hidden items-center gap-3 lg:flex">
          <StatusDot tone={probeTone(status?.grpc?.ok)} label="gRPC" />
          <StatusDot tone={probeTone(status?.postgres?.ok)} label="PG" />
          <StatusDot tone={probeTone(status?.qdrant?.ok)} label="Qdrant" />
        </div>
        {fsm && (
          <Pill tone={stateTone(fsm.state)}>
            <StatusDot tone={stateTone(fsm.state)} pulse={running} />
            {fsmLabel(fsm.state)}
          </Pill>
        )}
        {model && (
          <span className="hidden font-mono text-xs text-gray-500 xl:inline">{model}</span>
        )}
        <NavLink
          to="/settings"
          aria-label="Settings"
          className={({ isActive }) =>
            cn(
              "rounded-md p-2 transition-colors",
              isActive ? "bg-accent/20 text-accent" : "text-gray-400 hover:bg-surface-3 hover:text-gray-200",
            )
          }
        >
          <SettingsIcon className="h-4 w-4" />
        </NavLink>
      </div>
    </header>
  );
}
