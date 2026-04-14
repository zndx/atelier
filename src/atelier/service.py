"""Atelier gRPC Servicer — routes RPCs to business logic modules.

Follows the Fine Tuning Studio pattern: the servicer is a thin router.
All business logic lives in separate modules.
"""

import json

from atelier.db.dao import AtelierDao
from atelier.proto import atelier_pb2, atelier_pb2_grpc


class AtelierServicer(atelier_pb2_grpc.AtelierServicer):
    """Top-level gRPC servicer for the Atelier application."""

    def __init__(self):
        self._dao = None

    @property
    def dao(self) -> AtelierDao:
        if self._dao is None:
            self._dao = AtelierDao()
        return self._dao

    def HealthCheck(self, request, context):
        return atelier_pb2.HealthCheckResponse(
            status="ok",
            version="0.1.0",
        )

    def ListAgents(self, request, context):
        agents = self.dao.list_agents()
        return atelier_pb2.ListAgentsResponse(
            agents=[
                atelier_pb2.AgentMetadata(
                    id=a["id"],
                    name=a["name"] or "",
                    description=a["description"] or "",
                    role=a["role"] or "",
                    tool_ids=json.loads(a["tool_ids"] or "[]"),
                )
                for a in agents
            ]
        )

    def GetAgent(self, request, context):
        agent = self.dao.get_agent(request.id)
        if agent is None:
            return atelier_pb2.GetAgentResponse()
        return atelier_pb2.GetAgentResponse(
            agent=atelier_pb2.AgentMetadata(
                id=agent["id"],
                name=agent["name"] or "",
                description=agent["description"] or "",
                role=agent["role"] or "",
                tool_ids=json.loads(agent["tool_ids"] or "[]"),
            )
        )

    def ListDataSources(self, request, context):
        sources = self.dao.list_data_sources()
        return atelier_pb2.ListDataSourcesResponse(
            sources=[
                atelier_pb2.DataSource(
                    id=s["id"],
                    source_type=s["source_type"] or "",
                    source_uri=s["source_uri"] or "",
                    display_name=s["display_name"] or "",
                    vocabulary_mode=s["vocabulary_mode"] or "",
                    created_at=s["created_at"] or "",
                    metadata_json=s["metadata"] or "{}",
                )
                for s in sources
            ]
        )

    def ListDatasets(self, request, context):
        source_id = request.source_id or None
        datasets = self.dao.list_datasets(source_id=source_id)
        return atelier_pb2.ListDatasetsResponse(
            datasets=[
                atelier_pb2.ClassificationDataset(
                    id=ds["id"],
                    name=ds["name"] or "",
                    parquet_path=ds["parquet_path"] or "",
                    description=ds["description"] or "",
                    row_count=int(ds["row_count"] or 0),
                    source_id=ds.get("source_id") or "",
                    version_number=int(ds.get("version_number") or 1),
                    is_active=bool(ds.get("is_active", True)),
                    summary=ds.get("summary") or "",
                    fsm_run_id=ds.get("fsm_run_id") or "",
                    created_at=ds.get("created_at") or "",
                )
                for ds in datasets
            ]
        )

    def GetFSMStatus(self, request, context):
        from atelier.classify import get_fsm_status
        status = get_fsm_status()
        if status is None:
            return atelier_pb2.FSMStatusResponse(
                state="IDLE", progress_json="{}"
            )
        return atelier_pb2.FSMStatusResponse(
            run_id=status.get("id", ""),
            state=status.get("state", "IDLE"),
            started_at=status.get("started_at", ""),
            updated_at=status.get("updated_at", ""),
            progress_json=json.dumps(status.get("progress", {})),
            error=status.get("error") or "",
        )

    def StartClassification(self, request, context):
        import threading
        from atelier.classify import get_fsm
        from atelier.classify.pipeline import run_classification_pipeline
        from atelier.config import load_config

        cfg = load_config()
        fsm = get_fsm(self.dao)

        # Check if already running
        current = fsm.get_status()
        if current and current.state not in ("IDLE", "CONVERGED", "ERROR"):
            return atelier_pb2.StartClassificationResponse(
                run_id=current.id,
                started=False,
                error=f"Pipeline already running in state {current.state.value}",
            )

        # Pipeline owns run creation via fsm.start_run() — don't create
        # a duplicate run here (avoids double-run bug).
        conn = request.connection_name or None
        db = request.database or "default"
        sample_size = request.sample_size or 50

        def _background():
            run_classification_pipeline(
                cfg, fsm,
                connection_name=conn,
                database=db,
                sample_size=sample_size,
            )

        t = threading.Thread(target=_background, daemon=True)
        t.start()

        return atelier_pb2.StartClassificationResponse(
            started=True,
        )
