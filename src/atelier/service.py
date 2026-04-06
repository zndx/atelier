"""Atelier gRPC Servicer — routes RPCs to business logic modules.

Follows the Fine Tuning Studio pattern: the servicer is a thin router.
All business logic lives in separate modules.
"""

from atelier.proto import atelier_pb2, atelier_pb2_grpc


class AtelierServicer(atelier_pb2_grpc.AtelierServicer):
    """Top-level gRPC servicer for the Atelier application."""

    def HealthCheck(self, request, context):
        return atelier_pb2.HealthCheckResponse(
            status="ok",
            version="0.1.0",
        )

    def ListAgents(self, request, context):
        # Placeholder — will return keystone agent definitions
        return atelier_pb2.ListAgentsResponse(agents=[])

    def GetAgent(self, request, context):
        return atelier_pb2.GetAgentResponse()

    def ListDatasets(self, request, context):
        return atelier_pb2.ListDatasetsResponse(datasets=[])
