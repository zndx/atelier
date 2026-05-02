# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Row-level Monte Carlo sampling for bootstrap iteration diversity.

Selects different row subsets from a column's value reservoir across bootstrap
iterations, so each pass sees fresh evidence. Three strategies mirror Signals'
ImpalaSampler at the application level:

  - head: positional first-k (current pipeline behavior, baseline)
  - random: seeded random sample (fast, unstructured diversity)
  - stratified: value-length quantile bins with pattern-frequency tie-breaker
    (maximizes information yield per sample)

The stratified strategy partitions values by length into quantile bins, then
within each bin prefers values whose pattern signatures are least common across
the reservoir. This directly exploits the pipeline's pattern detectors and
maximizes diversity for the sentence-transformer embedding text.

Zero-cost when disabled: ``select_row_sample()`` is never called, and
``ColumnSample.values`` remains ``all_values[:5]`` as before.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.classify.sampler import ColumnSample

logger = logging.getLogger(__name__)

# Import pattern regexes from features module (lazy to avoid circular import)
_PATTERNS: dict | None = None


def _get_patterns() -> dict:
    global _PATTERNS
    if _PATTERNS is None:
        from atelier.classify.features import _PATTERNS as feat_patterns
        _PATTERNS = feat_patterns
    return _PATTERNS


# ── Configuration ────────────────────────────────────────────────


@dataclass(frozen=True)
class RowMCConfig:
    """Row-level MC configuration from HOCON."""

    enabled: bool = False
    k: int = 10
    strategy: str = "stratified"
    iterations: int = 3
    max_iterations: int = 5
    adaptive_escalation: bool = True

    @classmethod
    def from_cfg(cls, cfg) -> RowMCConfig:
        """Build from an AtelierConfig."""
        return cls(
            enabled=cfg.row_mc_enabled,
            k=cfg.row_mc_k,
            strategy=cfg.row_mc_strategy,
            iterations=min(cfg.row_mc_iterations, cfg.row_mc_max_iterations),
            max_iterations=cfg.row_mc_max_iterations,
            adaptive_escalation=cfg.row_mc_adaptive_escalation,
        )


# ── Public API ───────────────────────────────────────────────────


def select_row_sample(
    col: ColumnSample,
    k: int = 10,
    strategy: str = "stratified",
    iteration: int = 0,
    seed: int = 42,
) -> list[str]:
    """Select k values from col.all_values using the given strategy.

    Updates col.values in-place and returns the selected values.
    If all_values has fewer than k values, returns all of them.
    """
    pool = col.all_values
    if not pool:
        return col.values

    if len(pool) <= k:
        col.values = list(pool)
        return col.values

    rng = random.Random(seed + iteration)

    if strategy == "head":
        selected = _strategy_head(pool, k, iteration)
    elif strategy == "random":
        selected = _strategy_random(pool, k, rng)
    elif strategy == "stratified":
        selected = _strategy_stratified(pool, k, iteration, rng)
    else:
        logger.warning("Unknown row MC strategy %r, falling back to head", strategy)
        selected = _strategy_head(pool, k, iteration)

    col.values = selected
    return selected


# ── Strategies ───────────────────────────────────────────────────


def _strategy_head(pool: list[str], k: int, iteration: int) -> list[str]:
    """Sliding window: offset by iteration × k, wrapping around."""
    n = len(pool)
    start = (iteration * k) % n
    if start + k <= n:
        return pool[start:start + k]
    # Wrap around
    return pool[start:] + pool[:k - (n - start)]


def _strategy_random(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Seeded random sample without replacement."""
    return rng.sample(pool, k)


def _strategy_stratified(
    pool: list[str],
    k: int,
    iteration: int,
    rng: random.Random,
) -> list[str]:
    """Quantile-based selection with pattern-frequency tie-breaker.

    1. Partition values into 3 quantile bins by string length
    2. Within each bin, score by pattern rarity (prefer rare patterns)
    3. Rotate selection offset by iteration for diversity
    4. Sample proportionally from each bin
    """
    if len(pool) <= k:
        return list(pool)

    # Bin by value length into 3 quantiles
    lengths = [(i, len(v)) for i, v in enumerate(pool)]
    lengths.sort(key=lambda x: x[1])

    n = len(lengths)
    tercile_1 = n // 3
    tercile_2 = 2 * n // 3

    bins: list[list[int]] = [
        [idx for idx, _ in lengths[:tercile_1]],
        [idx for idx, _ in lengths[tercile_1:tercile_2]],
        [idx for idx, _ in lengths[tercile_2:]],
    ]
    # Remove empty bins
    bins = [b for b in bins if b]

    # Pattern-frequency scoring within each bin
    patterns = _get_patterns()
    pattern_counts = _count_pattern_hits(pool, patterns)

    scored_bins: list[list[tuple[int, float]]] = []
    for bin_indices in bins:
        scored = []
        for idx in bin_indices:
            # Rarity score: sum of (1 / count) for each matching pattern
            # Higher score = rarer pattern matches = more informative
            score = 0.0
            for pat_name, pat_re in patterns.items():
                if pat_re.match(pool[idx].strip()):
                    count = pattern_counts.get(pat_name, 1)
                    score += 1.0 / count
            scored.append((idx, score))
        # Sort by score descending (rarest patterns first)
        scored.sort(key=lambda x: -x[1])
        scored_bins.append(scored)

    # Allocate k across bins proportionally, with rotation offset
    selected_indices: list[int] = []
    per_bin = max(1, k // len(scored_bins))
    remainder = k - per_bin * len(scored_bins)

    for i, scored in enumerate(scored_bins):
        alloc = per_bin + (1 if i < remainder else 0)
        alloc = min(alloc, len(scored))
        # Rotate offset by iteration within this bin
        offset = (iteration * alloc) % max(len(scored), 1)
        for j in range(alloc):
            pick = (offset + j) % len(scored)
            selected_indices.append(scored[pick][0])

    # Deduplicate (rotation can cause overlaps in small pools)
    seen: set[int] = set()
    unique: list[int] = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    # Fill if we lost values to dedup
    if len(unique) < k:
        all_indices = set(range(len(pool)))
        available = list(all_indices - seen)
        rng.shuffle(available)
        unique.extend(available[:k - len(unique)])

    return [pool[i] for i in unique[:k]]


def _count_pattern_hits(pool: list[str], patterns: dict) -> dict[str, int]:
    """Count how many values in the pool match each pattern."""
    counts: dict[str, int] = {}
    for pat_name, pat_re in patterns.items():
        count = sum(1 for v in pool if pat_re.match(v.strip()))
        if count > 0:
            counts[pat_name] = count
    return counts
