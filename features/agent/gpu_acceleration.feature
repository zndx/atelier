@agent @tier-0
Feature: GPU acceleration configuration and probing
  The pipeline surfaces a single source of truth about GPU availability
  and acceleration methods via preflight_gpu() and /api/acceleration.
  CPU-only hosts must see a graceful fallback; GPU hosts must surface
  resolved devices and active methods.

  Scenario: Preflight returns a GpuInfo with resolved_devices
    When I call preflight_gpu
    Then the result has a resolved_devices list
    And resolved_device matches the first entry's prefix

  Scenario: Config exposes gpu section fields
    When I load the atelier config
    Then the config has classify_gpu_enabled
    And the config has classify_gpu_shard_threshold
    And the config has classify_gpu_sage_chunk

  Scenario: Acceleration endpoint returns methods + config dicts
    When I call GET /api/acceleration
    Then the response has a methods object
    And the response has a config object
    And the response summary is a non-empty string

  Scenario: MultiDeviceEncoder lazy-loads replicas
    Given embedding configured with at least one device
    When I fetch the encoder
    Then only the first replica is loaded initially
    And the device_count matches the configured device list
