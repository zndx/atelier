# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: ML Artifact Set vocabulary signature + compatibility
  As an Atelier operator extending classification onto new data, I need
  the system to detect when an artifact set's training vocabulary
  matches (or differs from) my current data source's vocabulary so I
  can decide whether the model's predictions will be meaningful.

  Scenario: Vocab signature is deterministic
    Given a class list with codes "EMAIL,PHONE,SSN"
    When I compute the vocab signature
    Then the signature should be a 64-character hex string

  Scenario: Vocab signature is stable under reordering
    Given a class list with codes "EMAIL,PHONE,SSN"
    And a class list with codes "SSN,EMAIL,PHONE"
    When I compute the signatures of both lists
    Then the two signatures should be equal

  Scenario: Compatibility is "ok" for identical class sets
    Given an artifact class list "EMAIL,PHONE,SSN"
    And a candidate class list "EMAIL,PHONE,SSN"
    When I check compatibility
    Then the compatibility status should be "ok"
    And the compatibility report should report 0 missing codes
    And the compatibility report should report 0 extra codes

  Scenario: Compatibility is "superset" when artifact is a subset
    Given an artifact class list "EMAIL,PHONE"
    And a candidate class list "EMAIL,PHONE,SSN,DOB"
    When I check compatibility
    Then the compatibility status should be "superset"
    And the compatibility report should report 0 missing codes
    And the compatibility report should report 2 extra codes

  Scenario: Compatibility is "partial" with overlapping but disjoint sets
    Given an artifact class list "EMAIL,PHONE,SSN"
    And a candidate class list "EMAIL,PHONE,DOB"
    When I check compatibility
    Then the compatibility status should be "partial"
    And the compatibility report should report 1 missing codes
    And the compatibility report should report 1 extra codes

  Scenario: Compatibility is "disjoint" with zero overlap
    Given an artifact class list "EMAIL,PHONE"
    And a candidate class list "SSN,DOB"
    When I check compatibility
    Then the compatibility status should be "disjoint"
    And the compatibility report should report 2 missing codes
    And the compatibility report should report 2 extra codes
