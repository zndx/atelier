import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { EmbeddingAtlasView } from "../widgets/EmbeddingAtlasView";

export default function EmbeddingsDetail() {
  const { datasetId } = useParams<{ datasetId: string }>();
  if (!datasetId) return null;
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-surface-3 bg-surface-1 px-4 py-2">
        <Link
          to="/embeddings"
          className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          All datasets
        </Link>
      </div>
      <div className="min-h-0 flex-1">
        <EmbeddingAtlasView datasetId={datasetId} />
      </div>
    </div>
  );
}
