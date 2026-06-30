import { useState } from "react";
import { Bot, Wand2, ChevronRight } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { listAgents, listSkills } from "../api/client";
import { Card, CardHeader } from "../ui/Card";
import { Pill } from "../ui/Pill";
import { Spinner, EmptyState } from "../ui/Feedback";
import type { SkillInfo } from "../api/types";

function SkillRow({ skill }: { skill: SkillInfo }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border border-surface-3 bg-surface-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
      >
        <Wand2 className="h-3.5 w-3.5 shrink-0 text-accent" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-gray-200">{skill.title}</span>
          <span className="block truncate text-xs text-gray-500">{skill.description}</span>
        </span>
        <ChevronRight
          className={`h-4 w-4 shrink-0 text-gray-600 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && skill.content && (
        <pre className="max-h-72 overflow-auto border-t border-surface-3 px-3 py-2 text-[11px] leading-relaxed text-gray-400">
          {skill.content}
        </pre>
      )}
    </div>
  );
}

export default function Agents() {
  const { data: agentsData, loading: la } = usePolling(listAgents, 0);
  const { data: skillsData, loading: ls } = usePolling(listSkills, 0);

  const agents = agentsData?.agents ?? [];
  const skills = skillsData?.skills ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Agents &amp; Skills</h1>
        <p className="text-sm text-gray-500">
          Registered agent roles and the skill commands available to the workbench.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-500">Agents</h2>
        {la ? (
          <Spinner />
        ) : agents.length === 0 ? (
          <EmptyState icon={<Bot />} title="No agents registered" />
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {agents.map((a) => (
              <Card key={a.id}>
                <CardHeader
                  icon={<Bot className="h-4 w-4" />}
                  title={a.name}
                  subtitle={a.role}
                  actions={a.tool_ids?.length ? <Pill tone="accent">{a.tool_ids.length} tools</Pill> : null}
                />
                <p className="mt-2 text-xs leading-relaxed text-gray-400">{a.description}</p>
              </Card>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Skills ({skills.length})
        </h2>
        {ls ? (
          <Spinner />
        ) : skills.length === 0 ? (
          <EmptyState icon={<Wand2 />} title="No skills found" />
        ) : (
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {skills.map((s) => (
              <SkillRow key={s.id} skill={s} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
