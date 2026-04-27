# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: ML classifier mass functions
  As a classification pipeline, I need CatBoost and SVM mass functions
  to convert ML predictions into DST belief assignments.

  Background:
    Given the mock annotations vocabulary is loaded
    And a frame of discernment from the vocabulary

  Scenario: CatBoost mass converts probabilities to belief assignment
    When I compute CatBoost mass from {"ICE.SENSITIVE.PID.CONTACT.EMAIL": 0.7, "ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME": 0.2, "ICE.METADATA.TIMESTAMP": 0.1}
    Then the ML mass function should not be vacuous
    And the ML top singleton should be "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And theta mass should be approximately 0.15

  Scenario: CatBoost adaptive discount increases with high variance
    When I compute CatBoost mass with variance from {"ICE.SENSITIVE.PID.CONTACT.EMAIL": 0.7, "ICE.METADATA.TIMESTAMP": 0.3} and {"ICE.SENSITIVE.PID.CONTACT.EMAIL": 0.25, "ICE.METADATA.TIMESTAMP": 0.25}
    Then the ML mass function should not be vacuous
    And theta mass should be greater than 0.20

  Scenario: SVM mass converts calibrated probabilities
    When I compute SVM mass from {"ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME": 0.6, "ICE.SENSITIVE.PID.IDENTITY.DOB": 0.3, "ICE.METADATA.TIMESTAMP": 0.1}
    Then the ML mass function should not be vacuous
    And the ML top singleton should be "ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME"
    And theta mass should be approximately 0.20

  Scenario: Mass functions return vacuous on empty input
    When I compute CatBoost mass for empty probabilities
    And I compute SVM mass for empty probabilities
    Then both mass functions should be vacuous
