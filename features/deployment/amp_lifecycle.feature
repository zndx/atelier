@deployment @amp
Feature: AMP deployment lifecycle
  Validates .project-metadata.yaml structure and the install-then-start
  workflow defined for Automated Machine Learning Prototype deployment.

  @tier-0
  Scenario: AMP metadata file is valid
    Given the file ".project-metadata.yaml" exists
    When I parse the AMP metadata
    Then it has a "name" field
    And it has a "runtimes" section
    And it has a "tasks" section

  @tier-0
  Scenario: AMP tasks follow create_job/run_job pattern
    Given the AMP metadata is loaded
    Then a "create_job" task with entity_label "install_deps" exists
    And a "run_job" task with entity_label "install_deps" exists
    And a "start_application" task exists

  @tier-0
  Scenario: Install script is valid Python
    When I compile "scripts/install_deps.py" with py_compile
    Then no SyntaxError is raised

  @tier-cai
  Scenario: AMP install job completes successfully
    Given I am in a CAI project session
    When I run the install dependencies job
    Then the job exits with code 0
    And "atelier" is importable in system Python
    And "node --version" succeeds
    And the directory "ui/dist" exists

  @tier-cai
  Scenario: AMP application starts and serves HTTP
    Given the install job has completed
    When I start the application via startup_app.py
    Then the HTTP gateway responds on CDSW_APP_PORT within 30 seconds
    And the gRPC server responds on port 50051
