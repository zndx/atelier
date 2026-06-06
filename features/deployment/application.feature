@deployment @application
Feature: CAI Application modality
  An Application is a long-running web service bound to CDSW_APP_PORT.
  CAI's reverse proxy routes subdomain traffic to this port.

  @tier-0
  Scenario: start-app.sh binds to 127.0.0.1 when CDSW_APP_PORT is set
    Given CDSW_APP_PORT is set to "8090"
    When I parse bin/start-app.sh for the HOST variable
    Then HOST is "127.0.0.1"

  @tier-0
  Scenario: start-app.sh binds to 0.0.0.0 for local dev
    Given CDSW_APP_PORT is not set
    When I parse bin/start-app.sh for the HOST variable
    Then HOST is "0.0.0.0"

  @tier-1
  Scenario: Full application stack starts locally
    When I run bin/start-app.sh in the background
    Then the HTTP gateway responds on port 8090 within 30 seconds
    And the gRPC server responds on port 50051
