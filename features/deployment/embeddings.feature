# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@deployment @embeddings
Feature: Embeddings integration
  Validates that the Embeddings page is properly integrated
  with the React frontend and can render parquet datasets.

  @tier-0
  Scenario: embedding-atlas is declared as npm dependency
    Given the file "ui/package.json" exists
    Then it declares dependency "embedding-atlas"

  @tier-0
  Scenario: Embeddings page component exists
    Then the file "ui/src/pages/Embeddings.tsx" exists

  @tier-0
  Scenario: React Router is configured
    Given the file "ui/src/App.tsx" exists
    Then it contains "react-router-dom"

  @tier-0
  Scenario: Dataset preparation script exists
    Then the file "scripts/prepare_gittables_sample.py" exists
