// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import Terminal from "../components/Terminal";

export default function TerminalPage() {
  return (
    <div style={{ flex: 1, minHeight: 0 }}>
      <Terminal style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
