@agent @tier-0 @gpu
Feature: GPU SAGE kernel produces sensible importance rankings
  The vectorized GPU SAGE kernel replaces the sage-importance library
  in the GPU path.  This scenario validates that on the OOTB sample
  corpus, the top features the kernel identifies match intuition —
  column name, sample values, and value description should dominate
  over null_ratio and numeric_ratio, which are numeric-only signals
  that provide little text for cosine classification.

  The @gpu tag gates this feature — it is skipped when no CUDA device
  is present.  No devenv services are required (pure in-process kernel).

  Scenario: GPU SAGE runs on OOTB sample within a reasonable budget
    Given the OOTB sample corpus loaded as ColumnFeatures
    When I run gpu_sage with 32 permutations
    Then the result has one importance value per feature
    And the elapsed time is under 60 seconds
    And the top feature by absolute importance is one of "column_name, value_description, sample_values"
