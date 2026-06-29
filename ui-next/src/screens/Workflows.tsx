import { RefreshCw } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { getFsmStatus, getStatus } from "../api/client";
import { PipelineCanvas } from "../widgets/PipelineCanvas";
import { Button } from "../ui/Button";
import { Pill } from "../ui/Pill";
import { StatusDot } from "../ui/StatusDot";
import { fsmLabel, isRunning, stateTone } from "../lib/fsm";
import type { RuntimeCapabilities } from "../lib/fsmPipelineLayout";

export default function Workflows() {
  const { data: fsm, refresh: refreshFsm } = usePolling(getFsmStatus, 5000);
  const { data: status, refresh: refreshStatus } = usePolling(getStatus, 15000);

  const cfg = status?.config ?? {};
  const capabilities: RuntimeCapabilities = {
    cautious_review_enabled: Boolean(cfg.cautious_review_enabled),
    overwatch_enabled: Boolean(cfg.overwatch_enabled),
    classify_agent_enabled: Boolean(cfg.classify_agent_enabled),
  };
  const state = fsm?.state ?? null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-surface-3 bg-surface-1 px-4 py-2.5">
        <div>
          <div className="text-sm font-semibold text-white">Pipeline Workflow</div>
          <div className="text-xs text-gray-500">FSM phases · skill attachments · live state</div>
        </div>
        <div className="flex items-center gap-2">
          <Pill tone={stateTone(state)}>
            <StatusDot tone={stateTone(state)} pulse={isRunning(state)} />
            {fsmLabel(state)}
          </Pill>
          <Button
            size="sm"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            onClick={() => {
              refreshFsm();
              refreshStatus();
            }}
          >
            Refresh
          </Button>
        </div>
      </div>
      <div className="min-h-0 flex-1">
        <PipelineCanvas state={state} capabilities={capabilities} />
      </div>
    </div>
  );
}
