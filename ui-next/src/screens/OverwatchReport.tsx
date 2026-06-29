import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePolling } from "../hooks/usePolling";
import { getOverwatchReport } from "../api/client";
import { Card } from "../ui/Card";
import { Spinner, Banner } from "../ui/Feedback";

export default function OverwatchReport() {
  const { runId } = useParams<{ runId: string }>();
  const { data, error, loading } = usePolling(
    () => getOverwatchReport(runId!),
    0,
    [runId],
  );

  return (
    <div className="space-y-4">
      <Link
        to="/status"
        className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to status
      </Link>
      <div>
        <h1 className="text-2xl font-bold text-white">Overwatch Report</h1>
        <p className="font-mono text-xs text-gray-500">{runId}</p>
      </div>
      {loading ? (
        <Spinner />
      ) : error ? (
        <Banner tone="error">No report found for this run.</Banner>
      ) : (
        <Card>
          <div className="prose-invert max-w-none text-sm leading-relaxed text-gray-300 [&_a]:text-accent [&_code]:font-mono [&_h1]:mb-3 [&_h1]:mt-0 [&_h1]:text-lg [&_h1]:font-bold [&_h1]:text-white [&_h2]:mb-2 [&_h2]:mt-5 [&_h2]:text-base [&_h2]:font-semibold [&_h2]:text-white [&_h3]:mt-4 [&_h3]:font-semibold [&_h3]:text-gray-200 [&_li]:my-1 [&_p]:my-2 [&_pre]:overflow-auto [&_pre]:rounded-md [&_pre]:bg-surface-1 [&_pre]:p-3 [&_strong]:text-gray-100 [&_table]:text-xs [&_ul]:list-disc [&_ul]:pl-5">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {data?.report ?? ""}
            </ReactMarkdown>
          </div>
        </Card>
      )}
    </div>
  );
}
