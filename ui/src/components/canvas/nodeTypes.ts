// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import type { NodeTypes } from "@xyflow/react";
import OrchestratorNode from "./OrchestratorNode";
import KeystoneNode from "./KeystoneNode";
import DynamicAgentNode from "./DynamicAgentNode";
import ArtifactNode from "./ArtifactNode";

export const nodeTypes: NodeTypes = {
  orchestrator: OrchestratorNode,
  keystone: KeystoneNode,
  dynamicAgent: DynamicAgentNode,
  artifact: ArtifactNode,
};
