@infra @health @qdrant
Feature: Qdrant health

  @tier-1
  Scenario: Qdrant health endpoint responds
    When I check the Qdrant health endpoint
    Then the response status is 200
