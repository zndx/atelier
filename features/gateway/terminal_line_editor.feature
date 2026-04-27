# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@gateway @tier-0
Feature: Web Terminal line editor — arrow keys, history, readline shortcuts
  Parity with claude-code / mistral vibe / kimi-cli REPLs: Up/Down
  browse history, Left/Right move cursor mid-line, Home/End + Ctrl-A/E
  jump to edges, Ctrl-W deletes a word, Ctrl-U/K kill partial lines,
  Delete removes at cursor. Every edit is dispatched through helpers
  on TerminalSession so the internal buffer and the on-wire cursor
  stay consistent.

  Background:
    Given a fresh TerminalSession

  Scenario: Typing echoes and advances cursor
    When the terminal receives "hello"
    Then the line buffer is "hello"
    And the cursor is at column 5

  Scenario: Left arrow moves the cursor without altering the buffer
    When the terminal receives "hello"
    And the terminal receives an arrow-left key 3 times
    Then the cursor is at column 2
    And the line buffer is "hello"

  Scenario: Insert at mid-line shifts the tail
    When the terminal receives "hello"
    And the terminal receives an arrow-left key 3 times
    And the terminal receives "X"
    Then the line buffer is "heXllo"
    And the cursor is at column 3

  Scenario: Backspace at mid-cursor deletes the char before cursor
    When the terminal receives "heXllo"
    And the terminal receives an arrow-left key 3 times
    And the terminal receives a backspace key
    Then the line buffer is "hello"
    And the cursor is at column 2

  Scenario: Delete key removes the char at cursor
    When the terminal receives "hello"
    And the terminal receives an arrow-left key 3 times
    And the terminal receives a delete key
    Then the line buffer is "helo"
    And the cursor is at column 2

  Scenario: Home / Ctrl-A jumps to start; End / Ctrl-E jumps to end
    When the terminal receives "abcdef"
    And the terminal receives ctrl-a
    Then the cursor is at column 0
    When the terminal receives ctrl-e
    Then the cursor is at column 6

  Scenario: Ctrl-K kills from cursor to end of line
    When the terminal receives "hello world"
    And the terminal receives an arrow-left key 6 times
    And the terminal receives ctrl-k
    Then the line buffer is "hello"
    And the cursor is at column 5

  Scenario: Ctrl-U kills from cursor to start of line
    When the terminal receives "hello world"
    And the terminal receives an arrow-left key 5 times
    And the terminal receives ctrl-u
    Then the line buffer is "world"
    And the cursor is at column 0

  Scenario: Ctrl-W deletes one whitespace-delimited word back
    When the terminal receives "foo bar baz"
    And the terminal receives ctrl-w
    Then the line buffer is "foo bar "
    And the cursor is at column 8

  Scenario: Up arrow browses history backwards
    Given the terminal has history entries "first", "second", "third"
    When the terminal receives an arrow-up key
    Then the line buffer is "third"
    When the terminal receives an arrow-up key
    Then the line buffer is "second"
    When the terminal receives an arrow-up key
    Then the line buffer is "first"

  Scenario: Down arrow past newest entry restores the in-progress edit
    Given the terminal has history entries "old"
    When the terminal receives "draft"
    And the terminal receives an arrow-up key
    Then the line buffer is "old"
    When the terminal receives an arrow-down key
    Then the line buffer is "draft"

  Scenario: Editing exits browse mode — next Up starts from newest entry
    Given the terminal has history entries "ls", "pwd"
    When the terminal receives an arrow-up key
    Then the line buffer is "pwd"
    When the terminal receives "X"
    Then the line buffer is "pwdX"
    When the terminal receives an arrow-up key
    Then the line buffer is "pwd"

  Scenario: Dedupes adjacent history entries
    When a command "ls" is recorded in history
    And a command "ls" is recorded in history
    And a command "pwd" is recorded in history
    Then the history has 2 entries
    And the newest history entry is "pwd"

  Scenario: Unknown CSI sequences are ignored
    When the terminal receives "abc"
    And the terminal receives a raw escape sequence "\x1b[99Z"
    Then the line buffer is "abc"
    And the cursor is at column 3
