@agent @tier-0
Feature: Runtime overlay — full parameter surface
  Every SETTINGS_METADATA key round-trips through the overlay:
  validates on set, writes into the AtelierConfig dataclass on apply,
  and reverts on clear. Covers each control type across each tab so
  that a new free-win entry is defended by at least one scenario.

  Scenario Outline: Setting "<key>" to <value> overlays the config
    Given the config overlay is empty
    When I set overlay "<key>" to <value>
    And I apply the overlay to a loaded config
    Then the resulting config has <key> equal to <value>

    # Convergence tab
    Examples: Convergence
      | key                                   | value |
      | classify_bootstrap_max_iterations     | 10    |
      | classify_bootstrap_k_threshold        | 0.30  |
      | classify_bootstrap_coverage_target    | 0.95  |
      | classify_bootstrap_max_total_llm_calls| 10000 |
      | classify_bootstrap_clarity_target     | 0.15  |

    # Evidence & Fusion tab
    Examples: Evidence
      | key                                      | value      |
      | classify_fusion_strategy                 | "yager"    |
      | classify_discount_svm                    | 0.25       |
      | classify_discount_pattern_theta          | 0.30       |
      | classify_discount_name_match_exact       | 0.55       |
      | classify_discount_catboost_variance_scale| 2.0        |
      | classify_discount_confusable_ratio_threshold | 2.5    |

    # Sampling tab
    Examples: Sampling
      | key                               | value           |
      | mc_min_corpus_size                | 500             |
      | mc_sample_fraction                | 0.35            |
      | mc_min_per_stratum                | 5               |
      | mc_max_frontier_columns           | 1000            |
      | row_mc_k                          | 15              |
      | row_mc_strategy                   | "random"        |

    # Training tab
    Examples: Training
      | key                                 | value |
      | classify_catboost_iterations        | 1500  |
      | classify_catboost_depth             | 8     |
      | classify_catboost_learning_rate     | 0.05  |
      | classify_sage_permutations          | 256   |
      | classify_shap_top_k                 | 5     |

    # LLM & System tab
    Examples: LLM and System
      | key                             | value     |
      | classify_llm_discount           | 0.18      |
      | classify_llm_max_tokens         | 16384     |
      | classify_llm_temperature        | 0.50      |
      | classify_llm_columns_per_call   | 100       |
      | classify_llm_max_retries        | 5         |
      | classify_embedding_batch_size   | 128       |
      | classify_gpu_enabled            | "false"   |
      | classify_gpu_shard_threshold    | 50000     |

  Scenario Outline: Boolean switch "<key>" toggles cleanly
    Given the config overlay is empty
    When I set overlay "<key>" to <value>
    And I apply the overlay to a loaded config
    Then the resulting config has <key> equal to bool <value>

    Examples:
      | key                                       | value |
      | classify_bootstrap_frontier_svm_retrain   | false |
      | row_mc_enabled                            | true  |
      | row_mc_adaptive_escalation                | false |
      | classify_sage_enabled                     | true  |
      | classify_shap_enabled                     | false |

  Scenario Outline: Invalid value for "<key>" is rejected
    Given the config overlay is empty
    When I set overlay "<key>" to <value>
    Then a ValueError is raised by the overlay

    Examples: Out-of-range
      | key                                | value   |
      | classify_bootstrap_max_iterations  | 100     |
      | mc_sample_fraction                 | 1.50    |
      | classify_catboost_depth            | 20      |

    Examples: Bad enum
      | key                                | value       |
      | row_mc_strategy                    | "triangle"  |
      | classify_gpu_enabled               | "sometimes" |

  Scenario: Metadata exposes a non-empty group for every key
    Given the config overlay is empty
    Then every SETTINGS_METADATA entry declares a tab group
