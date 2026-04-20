@gateway @tier-1 @slow
Feature: Classification pipeline via gateway
  Exercise the complete data flow: start pipeline via HTTP, wait for
  convergence, then verify results appear as a dataset. The OOTB
  sample source has classifiable columns across multiple tables
  with a known curated reference — pipeline results are fully verifiable.

  Scenario: OOTB sample classification produces viewable dataset
    When I POST "/api/fsm/start?source_id=ootb-sample" to the gateway
    Then the response status should be 200
    And the response JSON should contain "started"
    When I poll "/api/fsm/status" until state is terminal
    Then the FSM state should be "CONVERGED"
    When I GET "/api/datasets?source_id=ootb-sample" from the gateway
    Then the response status should be 200
    And the response JSON "datasets" should be a non-empty list
    And the latest dataset should have row_count greater than 100

  Scenario: Pipeline run appears in FSM history
    When I GET "/api/fsm/runs" from the gateway
    Then the response status should be 200
    And the response JSON "runs" should be a non-empty list
    And the latest run should have state "CONVERGED"
