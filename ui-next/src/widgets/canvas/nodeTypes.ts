import type { NodeTypes } from "@xyflow/react";
import FsmPhaseNode from "./FsmPhaseNode";
import SkillNode from "./SkillNode";

// Only the live FSM pipeline node types are used by ui-next.
export const nodeTypes: NodeTypes = {
  fsmPhase: FsmPhaseNode,
  skill: SkillNode,
};
