import { useState } from "react";
import { Play, Square } from "lucide-react";
import { useDataset } from "../state/DatasetContext";
import { cancelFsm, startFsm } from "../api/client";
import { ApiError } from "../api/client";
import type { FSMStatus } from "../api/types";
import { Button } from "../ui/Button";
import { Select } from "../ui/Field";
import { Pill } from "../ui/Pill";
import { StatusDot } from "../ui/StatusDot";
import { useToast } from "../ui/Toast";
import { fsmLabel, isRunning, stateTone } from "../lib/fsm";

// Start / cancel a classification run against the selected source.
// `fsm` is the latest status (owned by the parent poll); `onChanged`
// nudges the parent to refetch immediately for snappy feedback.
export function RunControls({
  fsm,
  onChanged,
}: {
  fsm: FSMStatus | null;
  onChanged?: () => void;
}) {
  const { sources, activeSourceId, setActiveSourceId } = useDataset();
  const { push } = useToast();
  const [busy, setBusy] = useState(false);
  const running = isRunning(fsm?.state);

  const start = async () => {
    setBusy(true);
    try {
      const r = await startFsm(activeSourceId);
      if (r.started) push(`Run started on ${r.source_id ?? "source"}`, "success");
      else push(r.error ?? "Could not start run", "error");
    } catch (e) {
      push(e instanceof ApiError ? e.message : "Start failed", "error");
    } finally {
      setBusy(false);
      onChanged?.();
    }
  };

  const cancel = async () => {
    setBusy(true);
    try {
      const r = await cancelFsm("operator cancel");
      push(r.cancelled ? "Run cancelled" : "Nothing to cancel", r.cancelled ? "success" : "info");
    } catch (e) {
      push(e instanceof ApiError ? e.message : "Cancel failed", "error");
    } finally {
      setBusy(false);
      onChanged?.();
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Select
        value={activeSourceId ?? ""}
        onChange={(e) => setActiveSourceId(e.target.value || null, { userPicked: true })}
        className="max-w-[220px]"
        disabled={running || busy}
      >
        {sources.length === 0 && <option value="">No sources</option>}
        {sources.map((s) => (
          <option key={s.id} value={s.id}>
            {s.display_name || s.id}
          </option>
        ))}
      </Select>

      {running ? (
        <Button variant="danger" icon={<Square className="h-3.5 w-3.5" />} onClick={cancel} disabled={busy}>
          Cancel
        </Button>
      ) : (
        <Button
          variant="primary"
          icon={<Play className="h-3.5 w-3.5" />}
          onClick={start}
          disabled={busy || !activeSourceId}
        >
          Start run
        </Button>
      )}

      {fsm && (
        <Pill tone={stateTone(fsm.state)}>
          <StatusDot tone={stateTone(fsm.state)} pulse={running} />
          {fsmLabel(fsm.state)}
        </Pill>
      )}
    </div>
  );
}
