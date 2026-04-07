"""Atelier gRPC Servicer — routes RPCs to business logic modules.

Follows the Fine Tuning Studio pattern: the servicer is a thin router.
All business logic lives in separate modules.
"""

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
            )
        )

    def ListDatasets(self, request, context):
        datasets = self.dao.list_datasets()
        return atelier_pb2.ListDatasetsResponse(
            datasets=[
                atelier_pb2.ClassificationDataset(
                    id=ds["id"],
                    name=ds["name"] or "",
                    parquet_path=ds["parquet_path"] or "",
                    description=ds["description"] or "",
                    row_count=int(ds["row_count"] or 0),
                )
                for ds in datasets
            ]
        )
