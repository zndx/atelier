# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: Governance SDK integration
  Atlas and Ranger clients for pushing classification types and tagging
  entities. All probes use mocked HTTP — no real Atlas/Ranger needed.

  Scenario: GovernanceClient builds from AtelierConfig
    Given an AtelierConfig with Atlas URL "https://atlas:21000"
    When I create a GovernanceClient from config
    Then the Atlas client base URL contains "api/atlas/v2"

  Scenario: GovernanceClient without Atlas URL raises on access
    Given an AtelierConfig with no Atlas URL
    When I create a GovernanceClient from config
    Then accessing atlas raises RuntimeError

  Scenario: ClassificationTag confidence is numeric
    Given a ClassificationTag with confidence 0.95
    Then the Atlas payload confidence attribute is "0.950"

  Scenario: SyncResult has separate entity_guid field
    Given a SyncResult with entity_name "col_email" and entity_guid "guid-1"
    Then entity_guid is "guid-1"
    And entity_name is "col_email"

  Scenario: TaxonomyNode maps to ClassificationTag with superTypes
    Given a TaxonomyNode "SENS.PID.EMAIL" with parent "SENS.PID"
    Then the ClassificationTag has super_types ["SENS.PID"]

  Scenario: QualifiedName builds Hive entity paths
    Given qualified name for "mydb.mytable" cluster "prod"
    Then table qualified name is "mydb.mytable@prod"
    And column qualified name for "email" is "mydb.mytable.email@prod"

  # URL normalization — handles malformed Cloudera SE-provided URLs
  Scenario: CDPUrlResolver coerces SE-provided URL with full API path
    Given Atlas URL "https://host/dl/cdp-proxy-api/atlas/api/atlas/v2"
    When I resolve the Atlas URL
    Then the resolved URL ends with "api/atlas/v2"
    And the resolved URL does not contain "api/atlas/v2/api/atlas/v2"

  Scenario: CDPUrlResolver coerces URL with partial API path
    Given Atlas URL "https://host/dl/cdp-proxy-api/atlas/api/atlas"
    When I resolve the Atlas URL
    Then the resolved URL ends with "api/atlas/v2"

  Scenario: CDPUrlResolver handles clean CDP proxy URL
    Given Atlas URL "https://host/dl/cdp-proxy-api/atlas"
    When I resolve the Atlas URL
    Then the resolved URL ends with "api/atlas/v2"

  Scenario: CDPUrlResolver handles bare host URL
    Given Atlas URL "https://atlas-host:21000"
    When I resolve the Atlas URL
    Then the resolved URL ends with "api/atlas/v2"
