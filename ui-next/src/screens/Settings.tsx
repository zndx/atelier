import { useEffect, useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { getSettings, patchSettings, resetSettings } from "../api/client";
import { ApiError } from "../api/client";
import type { SettingMetadata, SettingsPayload } from "../api/types";
import { Card, CardHeader } from "../ui/Card";
import { Button } from "../ui/Button";
import { Pill } from "../ui/Pill";
import { Select, TextInput, Toggle } from "../ui/Field";
import { Spinner } from "../ui/Feedback";
import { useToast } from "../ui/Toast";

type Values = Record<string, unknown>;

function groupOf(meta: SettingMetadata): string {
  return meta.group || "General";
}

function Control({
  meta,
  value,
  onChange,
}: {
  meta: SettingMetadata;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (meta.choices && meta.choices.length > 0) {
    return (
      <Select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
        {meta.choices.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </Select>
    );
  }
  if (typeof value === "boolean") {
    return <Toggle checked={value} onChange={onChange} />;
  }
  const numeric = typeof value === "number" || meta.min !== undefined || meta.max !== undefined;
  if (numeric) {
    return (
      <TextInput
        type="number"
        value={value === null || value === undefined ? "" : String(value)}
        min={meta.min}
        max={meta.max}
        step={meta.step ?? "any"}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    );
  }
  return (
    <TextInput value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />
  );
}

export default function Settings() {
  const { push } = useToast();
  const [payload, setPayload] = useState<SettingsPayload | null>(null);
  const [values, setValues] = useState<Values>({});
  const [overlay, setOverlay] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const p = await getSettings();
      setPayload(p);
      setValues(p.values);
      setOverlay(new Set(p.overlay_keys));
    } catch {
      push("Could not load settings", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const commit = async (k: string, v: unknown) => {
    setValues((prev) => ({ ...prev, [k]: v }));
    try {
      const r = await patchSettings({ [k]: v });
      setOverlay(new Set(Object.keys(r.overlay)));
      push(`Updated ${k}`, "success");
    } catch (e) {
      push(e instanceof ApiError ? e.message : `Failed to update ${k}`, "error");
      load(); // revert to server truth
    }
  };

  const reset = async () => {
    try {
      await resetSettings();
      push("Settings reset to defaults", "success");
      load();
    } catch {
      push("Reset failed", "error");
    }
  };

  const grouped = useMemo(() => {
    if (!payload) return [] as [string, string[]][];
    const groups = new Map<string, string[]>();
    for (const k of Object.keys(payload.metadata)) {
      const g = groupOf(payload.metadata[k]);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(k);
    }
    return Array.from(groups.entries());
  }, [payload]);

  if (loading) return <Spinner className="py-24" />;
  if (!payload) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-sm text-gray-500">
            Runtime overlay — applied to the next run, resets when the gateway restarts.
          </p>
        </div>
        <Button size="sm" variant="secondary" icon={<RotateCcw className="h-3.5 w-3.5" />} onClick={reset}>
          Reset to defaults
        </Button>
      </div>

      {grouped.map(([group, keys]) => (
        <Card key={group}>
          <CardHeader title={group} />
          <div className="mt-4 divide-y divide-surface-3/60">
            {keys.map((k) => {
              const meta = payload.metadata[k];
              return (
                <div key={k} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0 sm:max-w-[60%]">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-200">{meta.label || k}</span>
                      {overlay.has(k) && <Pill tone="accent">modified</Pill>}
                    </div>
                    {meta.description && (
                      <p className="mt-0.5 text-xs text-gray-500">{meta.description}</p>
                    )}
                    <p className="font-mono text-[10px] text-gray-600">{k}</p>
                  </div>
                  <div className="w-full shrink-0 sm:w-56">
                    <Control meta={meta} value={values[k]} onChange={(v) => commit(k, v)} />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      ))}
    </div>
  );
}
