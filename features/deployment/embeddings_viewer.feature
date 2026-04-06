@deployment @embeddings
Feature: Embeddings Viewer integration
  Validates that the Embeddings Viewer is properly integrated
  with the React frontend and can render parquet datasets.

  @tier-0
  Scenario: embedding-atlas is declared as npm dependency
    Given the file "ui/package.json" exists
    Then it declares dependency "embedding-atlas"

  @tier-0
  Scenario: Embeddings Viewer page component exists
    Then the file "ui/src/pages/EmbeddingsViewer.tsx" exists

  @tier-0
  Scenario: React Router is configured
    Given the file "ui/src/App.tsx" exists
    Then it contains "react-router-dom"

  @tier-0
  Scenario: Dataset preparation script exists
    Then the file "scripts/prepare_gittables_sample.py" exists
