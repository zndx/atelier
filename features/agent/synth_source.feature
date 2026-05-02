# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: OOTB synth data source — zip and directory mounts
  The OOTB synth corpus can be mounted either as an uncompressed
  directory or streamed directly from the zip archive.  The directory
  takes precedence when both are present.

  Scenario: Directory mount is preferred over zip
    Given a synth directory mount with 2 tables
    And a synth zip mount with 3 tables
    When I resolve the synth mount
    Then the resolver returns the directory path

  Scenario: Zip mount is used when no directory is present
    Given only a synth zip mount with 3 tables
    When I resolve the synth mount
    Then the resolver returns the zip path

  Scenario: No artifacts present returns None
    Given no synth artifacts present
    When I resolve the synth mount
    Then the resolver returns None

  Scenario: Load from zip streams columns and curated reference
    Given only a synth zip mount with 3 tables
    When I load the synth source from that mount
    Then I get 3 tables
    And each column has its curated reference attached

  Scenario: Load from directory yields the same samples as zip
    Given a synth directory mount with 2 tables
    When I load the synth source from that mount
    Then I get 2 tables
    And every column belongs to its parent table

  Scenario: Stats report the mount kind
    Given only a synth zip mount with 3 tables
    When I read synth source stats from that mount
    Then the stats report mount kind "zip"
    And the stats report 3 tables
