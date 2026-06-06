@agent @tier-0
Feature: CAI Data Platform — unified Hive + Filesystem surface
  The Data Platform card in Status.tsx lists both Hive connections and
  local filesystem mounts in a single dropdown.  Filesystem entries
  expose their annotations.csv vocabulary mirroring what Hive does
  with a `{db}.annotations` table.

  Scenario: /api/data-platforms merges Hive connections with filesystem mounts
    Given the gateway has seeded local filesystem sources
    When I GET "/api/data-platforms"
    Then the response lists at least one filesystem platform
    And every filesystem platform's source_uri starts with "file://"
    And every hive platform's source_uri starts with "hive://"

  Scenario: Meta-tagging filesystem entry surfaces its annotations.csv
    Given the gateway has seeded local filesystem sources
    When I GET "/api/filesystem-sources/meta-tagging/stats"
    Then the stats response is ok
    And the stats response includes an annotation_count > 0
    And the stats response's vocab_uri ends with "annotations.csv"

  Scenario: Filesystem vocabulary loads from an annotations.csv file
    Given a local filesystem mount with an annotations.csv
    When I load annotations from the filesystem path
    Then the resulting category set has at least 50 categories

  Scenario: Pipeline's vocabulary loader resolves file:// vocab_uri
    Given a local filesystem mount with an annotations.csv
    When I resolve vocabulary for a file:// vocab_uri against the mount
    Then the resolved category set has at least 50 categories
