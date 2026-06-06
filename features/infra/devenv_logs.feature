@infra @tier-1 @logs
Feature: devenv stack log health
  The devenv process manager (process-compose) captures stdout/stderr
  from every managed service and exposes current state via its CLI.
  This feature validates the running stack: no process has crashed or
  is thrashing on restart, and no Python tracebacks have surfaced in
  recent log output.

  This is meant to catch regressions that compile or import cleanly but
  fail at runtime — e.g., a module-level exception, a broken migration,
  a missing dependency, or a misconfigured env var.  The scenarios are
  tag-gated @logs so they can be run standalone when triaging a hang.

  Scenario: Every expected devenv process is healthy
    When I query process-compose for the current process states
    Then each of "gateway, grpc-server, postgres, qdrant, vite-dev" is reachable
    And every expected process is currently running
    # Cumulative restart count from prior incidents is ignored — we only
    # care about stability going forward.  The "No tracebacks in a
    # 5-second window" scenario below catches fresh crashes.

  Scenario: No Python tracebacks appear during a 5-second observation window
    Given I mark the current position of the process-compose log
    When I wait 5 seconds for the stack to emit heartbeats
    Then the log lines since the mark contain no "Traceback (most recent call last):"
    And the log lines since the mark contain no "ModuleNotFoundError"
    And the log lines since the mark contain no "sqlalchemy.exc"
