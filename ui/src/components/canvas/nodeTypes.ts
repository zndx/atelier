import type { NodeTypes } from "@xyflow/react";
import OrchestratorNode from "./OrchestratorNode";
import KeystoneNode from "./KeystoneNode";
import DynamicAgentNode from "./DynamicAgentNode";
import ArtifactNode from "./ArtifactNode";
import FsmPhaseNode from "./FsmPhaseNode";
import SkillNode from "./SkillNode";

export const nodeTypes: NodeTypes = {
  orchestrator: OrchestratorNode,
  keystone: KeystoneNode,
  dynamicAgent: DynamicAgentNode,
  artifact: ArtifactNode,
  fsmPhase: FsmPhaseNode,
  skill: SkillNode,
};
