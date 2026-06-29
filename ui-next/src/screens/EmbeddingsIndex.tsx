import { Link } from "react-router-dom";
import { ScatterChart, ArrowRight } from "lucide-react";
import { useDataset } from "../state/DatasetContext";
import { Card } from "../ui/Card";
import { Pill } from "../ui/Pill";
import { EmptyState } from "../ui/Feedback";

export default function EmbeddingsIndex() {
  const { datasets } = useDataset();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Embeddings</h1>
        <p className="text-sm text-gray-500">
          Late-interaction projections of classified entities. Pick a dataset to open the atlas.
        </p>
      </div>

      {datasets.length === 0 ? (
        <EmptyState
          icon={<ScatterChart />}
          title="No datasets yet"
          description="Complete a classification run to produce an embedding dataset."
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {datasets.map((d) => (
            <Link key={d.id} to={`/embeddings/${encodeURIComponent(d.id)}`}>
              <Card className="group transition-colors hover:border-accent/40">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-white">{d.name}</div>
                    <div className="mt-0.5 font-mono text-xs text-gray-500">
                      {d.row_count?.toLocaleString() ?? "—"} entities · v{d.version_number}
                    </div>
                  </div>
                  <ArrowRight className="h-4 w-4 shrink-0 text-gray-600 transition-colors group-hover:text-accent" />
                </div>
                {d.is_active && (
                  <div className="mt-3">
                    <Pill tone="green">active</Pill>
                  </div>
                )}
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
