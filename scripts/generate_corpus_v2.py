#!/usr/bin/env python3
"""scripts/generate_corpus_v2.py — Phase C of the corpus expansion plan.

Produces ~200 synthetic columns per taxonomy node (~57k total rows
across 287 nodes) for training the factorized NHSVM head.  Per column:

  - 5 sample values from a generator (v1 → hand-coded ICE → template
    → inferred, in priority order)
  - A column name from per-code NAME_VARIANTS_BY_CODE (v1) merged
    with the historical 8-style naming rotation
  - 4 sibling column names from per-code TABLE_CONTEXT_BY_CODE (v1,
    70% of the time) or cross-domain mixing (30%)
  - A synthetic table name from the same template (or
    cross-domain default)
  - A column type inferred from a 10-sample probe of the generator

Output: ``build/data/svm_training/corpus_v2/synth_rows.jsonl`` — one
JSON record per row matching the ``Row`` dataclass shape that
``reflect_nhsvm.load_rows`` produces.  Phase D loads these directly.

Resume-safe: if the output exists and `manifest.json` matches the
expected per-code targets, skips work for codes already at target.

Usage:
  python scripts/generate_corpus_v2.py
  python scripts/generate_corpus_v2.py --rows-per-node 200
  python scripts/generate_corpus_v2.py --rows-per-node 100  # smaller smoke run
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import logging
import random
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflect_nhsvm import build_category_set
from atelier.classify.enrichment_loader import load_enrichment_payloads
from atelier.classify.svm_classifier import build_svm_text

log = logging.getLogger("generate_corpus_v2")

# Module-level constants (legacy paths); the active output is parameterized
# via the --output flag and threaded through `generate_corpus`.
DEFAULT_OUTPUT_DIR = Path("build/data/svm_training/corpus_v2")
DEFAULT_PAYLOADS = Path("build/data/svm_training/enrichment_payloads.json")
GENERATORS_V1 = Path("build/lib/generated/generators_v1.py")

# Cross-domain fallback table names and sibling pools — used when a
# code lacks per-code TABLE_CONTEXT_BY_CODE entries.
FALLBACK_TABLE_NAMES = [
    "users", "customers", "accounts", "transactions", "events",
    "orders", "products", "audit_log", "sessions", "subscriptions",
    "device_registry", "communication_log", "payment_history",
    "permission_grants", "system_metadata", "incident_reports",
]
FALLBACK_SIBLINGS = [
    "user_id", "created_at", "updated_at", "status", "type",
    "amount", "currency", "country_code", "ip_address", "session_id",
    "first_name", "last_name", "email", "phone", "address",
    "device_id", "transaction_id", "reference_id", "name", "description",
    "is_active", "version", "metadata", "source", "channel",
]


# ──────────────────────────────────────────────────────────────────────
# Load generators_v1 if present
# ──────────────────────────────────────────────────────────────────────

def load_generators_v1() -> dict:
    """Load v1 generators + name + table-context dicts.

    Returns {} if generators_v1.py doesn't exist; otherwise returns
    {generators_by_code, name_variants_by_code, table_context_by_code}.
    """
    if not GENERATORS_V1.exists():
        log.info("  generators_v1.py not present — will use only ICE + template + inferred")
        return {
            "generators_by_code": {},
            "name_variants_by_code": {},
            "table_context_by_code": {},
        }
    spec = importlib.util.spec_from_file_location("generators_v1", GENERATORS_V1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n_gens = len(getattr(mod, "GENERATORS_BY_CODE", {}))
    n_names = len(getattr(mod, "NAME_VARIANTS_BY_CODE", {}))
    n_tables = len(getattr(mod, "TABLE_CONTEXT_BY_CODE", {}))
    log.info("  loaded generators_v1.py: %d codes have generators, %d have name variants, %d have table contexts",
             n_gens, n_names, n_tables)
    return {
        "generators_by_code": dict(getattr(mod, "GENERATORS_BY_CODE", {})),
        "name_variants_by_code": dict(getattr(mod, "NAME_VARIANTS_BY_CODE", {})),
        "table_context_by_code": dict(getattr(mod, "TABLE_CONTEXT_BY_CODE", {})),
    }


# ──────────────────────────────────────────────────────────────────────
# Type inference (matches what svm_generator_experiment does at line 554)
# ──────────────────────────────────────────────────────────────────────

def infer_column_type(samples: list[str]) -> str:
    """Infer a plausible SQL type from generator output samples."""
    if not samples:
        return "string"
    numeric_count = 0
    date_like = 0
    bool_like = 0
    max_len = 0
    for v in samples:
        v_str = str(v).strip()
        max_len = max(max_len, len(v_str))
        # Numeric check
        try:
            float(v_str.replace(",", ""))
            numeric_count += 1
        except (ValueError, OverflowError):
            pass
        # Date check (loose)
        if re.match(r"^\d{4}-\d{2}-\d{2}", v_str):
            date_like += 1
        # Boolean check
        if v_str.lower() in {"true", "false", "yes", "no", "1", "0"}:
            bool_like += 1
    n = len(samples)
    if numeric_count >= n * 0.8:
        # Distinguish int vs float
        has_decimal = any("." in str(v) for v in samples)
        return "double" if has_decimal else "bigint"
    if date_like >= n * 0.8:
        return "timestamp"
    if bool_like >= n * 0.8:
        return "boolean"
    if max_len > 200:
        return "text"
    return "string"


# ──────────────────────────────────────────────────────────────────────
# Column name generation — merges v1 names with style rotations
# ──────────────────────────────────────────────────────────────────────

def _snake_case(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")


def _camel_case(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


OPAQUE_PREFIXES = ["val_", "col_", "dat_", "fld_", "attr_", "f", "c", "v"]


def synth_column_name(
    cat,
    rng: random.Random,
    *,
    v1_names: list[str] | None = None,
    variant_idx: int = 0,
    opaque: bool = False,
) -> str:
    """Produce a per-column name.

    Pool priority:
      1. v1_names from NAME_VARIANTS_BY_CODE[code]  (Opus-authored)
      2. 8-style rotation from label/abbrev/common_names
      3. Opaque (val_42, col_07) as ~25% of names per SHAP-priority spec
    """
    if opaque:
        prefix = rng.choice(OPAQUE_PREFIXES)
        style = rng.randint(0, 3)
        if style == 0:
            return f"{prefix}{rng.randint(1, 999)}"
        elif style == 1:
            return f"{prefix}{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 4)))}"
        elif style == 2:
            return f"{prefix}{rng.choice(string.ascii_lowercase)}{rng.randint(1, 99)}"
        else:
            return f"{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))}{rng.randint(1, 99)}"

    # 60% chance to draw from v1_names pool when available
    if v1_names and rng.random() < 0.6:
        return rng.choice(v1_names)

    # Otherwise style rotation from label/abbrev
    label = getattr(cat, "label", "") or ""
    abbrev = getattr(cat, "abbrev", "") or ""
    # If label is just a code-like string (digits/dots), prefer abbrev —
    # this happens with the Hive-format taxonomy where label == code.
    if not label or re.match(r"^[\d.]+$", label):
        label = abbrev
    base = _snake_case(label) or _snake_case(abbrev) or f"col_{cat.code.replace('.', '_')}"

    style = variant_idx % 8
    if style == 0:
        return base
    elif style == 1:
        return _camel_case(base)
    elif style == 2:
        return base.upper()
    elif style == 3 and abbrev:
        return abbrev.strip().lower()
    elif style == 4:
        prefix = rng.choice(["user_", "customer_", "src_", "raw_", "dim_", "fact_"])
        return prefix + base
    elif style == 5:
        suffix = rng.choice(["_val", "_field", "_data", "_info", "_raw", "_v2"])
        return base + suffix
    elif style == 6:
        # Two-word combo
        prefix = rng.choice(["primary", "secondary", "alt", "backup", "legacy", "current"])
        return f"{prefix}_{base}"
    else:
        # Numeric discriminator
        return f"{base}_{variant_idx % 10 + 1}" if variant_idx else base


# ──────────────────────────────────────────────────────────────────────
# Table-context selection
# ──────────────────────────────────────────────────────────────────────

def pick_table_context(
    rng: random.Random,
    *,
    code_templates: list[dict] | None,
    fallback_table_names: list[str],
    fallback_siblings: list[str],
) -> tuple[str, list[str]]:
    """Return (table_name, sibling_columns).

    70% from per-code templates (v1) when available, 30% cross-domain.
    """
    if code_templates and rng.random() < 0.7:
        template = rng.choice(code_templates)
        table_names = template.get("table_names", []) or fallback_table_names
        siblings_pool = template.get("siblings", []) or fallback_siblings
        table_name = rng.choice(table_names)
        # Pick 4 siblings from the pool
        if len(siblings_pool) >= 4:
            siblings = rng.sample(siblings_pool, 4)
        else:
            siblings = list(siblings_pool)
            while len(siblings) < 4:
                siblings.append(rng.choice(fallback_siblings))
        return table_name, siblings

    table_name = rng.choice(fallback_table_names)
    siblings = rng.sample(fallback_siblings, 4)
    return table_name, siblings


# ──────────────────────────────────────────────────────────────────────
# Lean-text helper — matches reflect_nhsvm.build_texts_and_labels output
# so embeddings produced here are interchangeable with Phase D's.
# ──────────────────────────────────────────────────────────────────────

def _build_lean_text(row_data: dict) -> str:
    """Reconstruct the encoder-input lean text from a generated row dict.

    Mirrors ``scripts/reflect_nhsvm.build_texts_and_labels`` so the
    encoder sees the same string shape at gating time and at training
    time.
    """
    from atelier.classify.features import _closest_siblings
    col = row_data["column"]
    siblings_full = row_data.get("siblings_full") or []
    siblings = _closest_siblings(col, siblings_full, k=4)
    sibling_part = ", ".join(s.replace("_", " ") for s in siblings)
    base = build_svm_text(col, row_data.get("column_type", ""),
                           row_data.get("sample_values", []))
    return f"{base} | siblings: {sibling_part}" if sibling_part else base


# ──────────────────────────────────────────────────────────────────────
# Diversity gating + marginal-coverage stopping
# ──────────────────────────────────────────────────────────────────────

class _CodeAcceptor:
    """Per-code acceptance state for the diversity gate + coverage stop.

    Maintains running statistics over the rows accepted so far for a
    single code and decides whether each new candidate should be
    accepted or rejected.  Surfaces a stopping reason when the code's
    generator has exhausted its useful diversity or saturated the
    embedding-space neighborhood.
    """

    # ModernBERT's general-purpose encoder embeds same-template lean
    # text at ~0.97 baseline cosine sim within a code (per
    # intra_code_mean_sim measurements in corpus_metrology, 0.95-0.99
    # across codes).  An absolute threshold of 0.95 would reject every
    # candidate after the first 2-3 (observed empirically on corpus_v3
    # first attempt: 287/287 codes hit diversity_exhausted at 2-3 rows).
    # The threshold needs to sit ABOVE the encoder's baseline so only
    # near-exact duplicates are rejected.
    DIVERSITY_THRESHOLD = 0.995          # reject if cos sim to nearest > this
    REJECTION_WINDOW_SIZE = 20
    REJECTION_WINDOW_RATE = 0.6          # > this rate → diversity_exhausted
    # Centroid-radius growth threshold also scaled down to match
    # ModernBERT's tight intra-code cluster.
    COVERAGE_EPS = 0.002
    COVERAGE_WINDOW = 10

    def __init__(self, target_count: int, *, enable_diversity: bool,
                  enable_coverage: bool):
        self.target = target_count
        self.enable_diversity = enable_diversity
        self.enable_coverage = enable_coverage
        self.accepted_rows: list[dict] = []
        self.accepted_emb: list[np.ndarray] = []  # L2-normalized
        self.rejection_window: collections.deque[bool] = collections.deque(
            maxlen=self.REJECTION_WINDOW_SIZE)
        self.radius_history: collections.deque[float] = collections.deque(
            maxlen=self.COVERAGE_WINDOW)
        self.centroid: np.ndarray | None = None
        self.stopping_reason: str | None = None

    def consider(self, row_data: dict, embedding: np.ndarray) -> str:
        """Returns the outcome of considering this candidate:
           'accepted' | 'rejected_diversity' |
           'stopped_diversity_exhausted' | 'stopped_coverage_saturated' |
           'stopped_target_met'

        The stopped_* outcomes also set self.stopping_reason; further
        consider() calls should not be made.
        """
        if self.stopping_reason is not None:
            return f"stopped_{self.stopping_reason}"

        emb = (embedding / (np.linalg.norm(embedding) + 1e-12)).astype(np.float32)

        # 1. Diversity gate
        if self.enable_diversity and self.accepted_emb:
            sims = np.array([float(np.dot(emb, e)) for e in self.accepted_emb])
            if float(sims.max()) > self.DIVERSITY_THRESHOLD:
                self.rejection_window.append(True)
                # Sliding-window rejection-rate test
                if (len(self.rejection_window) == self.REJECTION_WINDOW_SIZE
                        and sum(self.rejection_window) / self.REJECTION_WINDOW_SIZE
                            > self.REJECTION_WINDOW_RATE):
                    self.stopping_reason = "diversity_exhausted"
                    return "stopped_diversity_exhausted"
                return "rejected_diversity"

        # 2. Accept
        self.rejection_window.append(False)
        self.accepted_rows.append(row_data)
        self.accepted_emb.append(emb)

        # 3. Update centroid + radius
        if self.centroid is None:
            self.centroid = emb.copy()
        else:
            n = len(self.accepted_emb)
            self.centroid = ((n - 1) * self.centroid + emb) / n
            cn = float(np.linalg.norm(self.centroid))
            self.centroid = self.centroid / (cn + 1e-12)
        max_radius = max(
            1.0 - float(np.dot(self.centroid, e)) for e in self.accepted_emb
        )
        self.radius_history.append(max_radius)

        # 4. Coverage-saturation check
        if (self.enable_coverage
                and len(self.radius_history) == self.COVERAGE_WINDOW
                and (self.radius_history[-1] - self.radius_history[0])
                    < self.COVERAGE_EPS):
            self.stopping_reason = "coverage_saturated"
            return "stopped_coverage_saturated"

        # 5. Target met?
        if len(self.accepted_rows) >= self.target:
            self.stopping_reason = "target_met"
            return "stopped_target_met"

        return "accepted"


def _encode_candidates(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode candidate lean texts with ModernBERT (mean-pooled).

    Imported lazily — ModernBERT load is slow and only paid when gating
    is enabled.
    """
    from reflect_nhsvm_modernbert import encode_modernbert
    return encode_modernbert(texts, batch_size=batch_size, pooling="mean")


# ──────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────

def generate_corpus(
    *,
    synth_examples_per_code: int,
    payloads_path: Path,
    output_dir: Path,
    seed: int = 42,
    enable_diversity_gate: bool = True,
    enable_marginal_coverage_stop: bool = True,
    encoding_batch_size: int = 64,
) -> dict:
    """Generate up to synth_examples_per_code columns per taxonomy node.

    With diversity_gate and marginal_coverage_stop both enabled (the
    default), per-code count may fall below the target — that's by
    design: an exhausted generator's marginal example adds redundancy
    to the encoder's view, not new signal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    synth_rows_path = output_dir / "synth_rows.jsonl"
    manifest_path = output_dir / "manifest.json"
    log.info("=== Phase C: corpus generation at scale ===")
    log.info("Loading category set...")
    cat_set = build_category_set()
    cats_by_code = {c.code: c for c in cat_set.all_categories}
    log.info("  %d nodes total", len(cats_by_code))

    log.info("Loading enrichment payloads from %s...", payloads_path)
    payloads = load_enrichment_payloads(json_path=payloads_path)

    # NOTE: upstream ICE / template / inferred fallback is DELIBERATELY
    # NOT loaded here.  ICE = Information Content Entity (BFO/CCO
    # ontology surface managed upstream in the Aegir project — NOT
    # legacy).  The SVM stage requires its own v1 (agent-authored)
    # generators because the metrology + refinement machinery can only
    # steer generators that the agent owns end-to-end.  ICE continues
    # to exist as a first-class upstream surface, just outside this
    # channel's training corpus.  Codes without v1 coverage are
    # skipped and surfaced in the manifest's `n_nodes_skipped` so the
    # next coverage audit picks them up as gaps for the agent to fill.
    log.info("Loading generators_v1 (Phase B output)...")
    v1 = load_generators_v1()
    # Build a v1 callable lookup: code → list[callable]
    # GENERATORS_BY_CODE in v1 is dict[code, list[callable]] (Phase B output)
    v1_gens_by_code: dict[str, list] = {}
    for code, gen_list in v1["generators_by_code"].items():
        if not isinstance(gen_list, list):
            gen_list = [gen_list]  # backward compat: single callable
        v1_gens_by_code[code] = [g for g in gen_list if callable(g)]
    log.info("  v1 generators available for %d codes", len(v1_gens_by_code))

    def _build_candidate(
        cat, code: str, mnemonic: str, gen_source: str,
        candidate_gens: list, v1_names: list[str] | None,
        v1_table_ctx: list[dict] | None, rng: random.Random,
        variant_idx: int,
    ) -> dict | None:
        """Produce one candidate row dict, or None if the generator
        emits an all-empty sample (shouldn't happen post-validation).
        Logic mirrors the original inline body but extracted for the
        gated-vs-ungated dispatcher below.
        """
        gen = (rng.choice(candidate_gens) if len(candidate_gens) > 1
               else candidate_gens[0])

        sample_values: list[str] = []
        for _ in range(5):
            try:
                sample_values.append(str(gen(rng)))
            except Exception:  # noqa: BLE001
                sample_values.append("")
        if not any(s for s in sample_values):
            return None

        # Column type probe + 20% adjacent-type diversity
        probe = sample_values + [str(gen(rng)) for _ in range(5)]
        col_type = infer_column_type(probe)
        if rng.random() < 0.2:
            adjacent_map = {
                "string": ["varchar", "text", "char"],
                "varchar": ["string", "text"],
                "text": ["string", "varchar"],
                "bigint": ["int", "integer", "long"],
                "double": ["float", "decimal", "numeric"],
                "boolean": ["bool", "bit"],
                "timestamp": ["datetime", "date"],
            }
            col_type = rng.choice(adjacent_map.get(col_type, [col_type]))

        opaque = rng.random() < 0.25
        col_name = synth_column_name(
            cat, rng, v1_names=v1_names,
            variant_idx=variant_idx, opaque=opaque,
        )
        table_name, siblings = pick_table_context(
            rng, code_templates=v1_table_ctx,
            fallback_table_names=FALLBACK_TABLE_NAMES,
            fallback_siblings=FALLBACK_SIBLINGS,
        )
        siblings_full = [col_name] + siblings

        return {
            "table": table_name,
            "column": col_name,
            "column_type": col_type,
            "sample_values": sample_values,
            "siblings_full": siblings_full,
            "mnemonic": mnemonic,
            "code": code,
            "gen_source": gen_source,
        }

    # Coverage check: per-code, do we have ANY generator available?
    rng = random.Random(seed)
    rows: list[dict] = []
    per_code_counts: dict[str, int] = {}
    per_code_stopping_reason: dict[str, str] = {}
    skipped_no_gen: list[str] = []
    gating_enabled = enable_diversity_gate or enable_marginal_coverage_stop
    CANDIDATE_BATCH = 16
    CANDIDATE_HARD_CAP_MULTIPLIER = 4

    if gating_enabled:
        log.info(
            "Diversity gate: %s; coverage stop: %s (encoding batch size %d)",
            "ON" if enable_diversity_gate else "OFF",
            "ON" if enable_marginal_coverage_stop else "OFF",
            encoding_batch_size,
        )
    else:
        log.info("Gating DISABLED — generating fixed %d examples per code",
                 synth_examples_per_code)

    t0 = time.time()
    for cat_idx, code in enumerate(sorted(cats_by_code.keys())):
        cat = cats_by_code[code]
        mnemonic = getattr(cat, "abbrev", "") or ""

        if code in v1_gens_by_code and v1_gens_by_code[code]:
            candidate_gens = v1_gens_by_code[code]
            gen_source = "v1"
        else:
            # No upstream-ICE fallback — see note above (upstream ICE
            # is BFO/CCO ontology surface managed in Aegir, outside
            # this channel's scope).  The coverage audit must catch
            # this code and route it to the agent for authoring before
            # the next corpus generation.
            skipped_no_gen.append(code)
            continue

        v1_names = v1["name_variants_by_code"].get(code)
        v1_table_ctx = v1["table_context_by_code"].get(code)

        if not gating_enabled:
            # Fast path: no encoding, fixed count per code
            for col_i in range(synth_examples_per_code):
                cand = _build_candidate(
                    cat, code, mnemonic, gen_source, candidate_gens,
                    v1_names, v1_table_ctx, rng, col_i,
                )
                if cand is None:
                    continue
                rows.append(cand)
                per_code_counts[code] = per_code_counts.get(code, 0) + 1
            per_code_stopping_reason[code] = "target_met"
        else:
            # Gated path: batch-encode candidates and accept/reject per code
            acceptor = _CodeAcceptor(
                synth_examples_per_code,
                enable_diversity=enable_diversity_gate,
                enable_coverage=enable_marginal_coverage_stop,
            )
            variant_idx = 0
            candidates_seen = 0
            hard_cap = max(
                synth_examples_per_code * CANDIDATE_HARD_CAP_MULTIPLIER, 100,
            )
            while acceptor.stopping_reason is None and candidates_seen < hard_cap:
                # Build a batch of candidates
                batch: list[dict] = []
                while len(batch) < CANDIDATE_BATCH and candidates_seen < hard_cap:
                    cand = _build_candidate(
                        cat, code, mnemonic, gen_source, candidate_gens,
                        v1_names, v1_table_ctx, rng, variant_idx,
                    )
                    candidates_seen += 1
                    variant_idx += 1
                    if cand is not None:
                        batch.append(cand)
                if not batch:
                    break
                # Encode batch
                texts = [_build_lean_text(c) for c in batch]
                embs = _encode_candidates(texts, batch_size=encoding_batch_size)
                # Decide sequentially
                for cand, emb in zip(batch, embs):
                    outcome = acceptor.consider(cand, emb)
                    if outcome.startswith("stopped_"):
                        break
            # Record results for this code
            for r in acceptor.accepted_rows:
                rows.append(r)
            per_code_counts[code] = len(acceptor.accepted_rows)
            per_code_stopping_reason[code] = (
                acceptor.stopping_reason or "hard_cap_hit"
            )

        if (cat_idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            log.info("  %d/%d nodes done  %d rows  %.1fs",
                     cat_idx + 1, len(cats_by_code), len(rows), elapsed)

    elapsed = time.time() - t0
    log.info("Generated %d total rows across %d nodes in %.1fs (%d skipped for no generator)",
             len(rows), len(per_code_counts), elapsed, len(skipped_no_gen))

    # Stopping-reason summary
    if gating_enabled:
        reason_counts: dict[str, int] = {}
        for r in per_code_stopping_reason.values():
            reason_counts[r] = reason_counts.get(r, 0) + 1
        log.info("Stopping reasons: %s", reason_counts)

    # Persist
    with synth_rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "synth_examples_per_code_target": synth_examples_per_code,
        "diversity_gate_enabled": enable_diversity_gate,
        "marginal_coverage_stop_enabled": enable_marginal_coverage_stop,
        "n_rows_total": len(rows),
        "n_nodes_covered": len(per_code_counts),
        "n_nodes_skipped": len(skipped_no_gen),
        "skipped_codes": sorted(skipped_no_gen),
        "per_code_counts": per_code_counts,
        "per_code_stopping_reason": per_code_stopping_reason,
        "elapsed_sec": round(elapsed, 1),
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s and %s", synth_rows_path, manifest_path)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-examples-per-code", type=int, default=40,
                    help="Target synthetic columns per taxonomy node "
                         "(default 40; encoder's lean-text depth is ~5 sample "
                         "values, so 40 distinct examples already give 8x that)")
    ap.add_argument("--rows-per-node", type=int, default=None,
                    help="DEPRECATED alias for --synth-examples-per-code")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Output directory for synth_rows.jsonl + manifest.json")
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS,
                    help="Enrichment payloads JSON path")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility")
    ap.add_argument("--diversity-gate", dest="diversity_gate",
                    action="store_true", default=True,
                    help="Reject candidates with cos-sim > 0.95 to any "
                         "accepted-same-code example (default ON)")
    ap.add_argument("--no-diversity-gate", dest="diversity_gate",
                    action="store_false",
                    help="Disable the diversity gate")
    ap.add_argument("--marginal-coverage-stop", dest="marginal_coverage_stop",
                    action="store_true", default=True,
                    help="Stop adding examples per code when the centroid "
                         "max-radius growth falls below ε (default ON)")
    ap.add_argument("--no-marginal-coverage-stop", dest="marginal_coverage_stop",
                    action="store_false",
                    help="Disable marginal-coverage stopping")
    ap.add_argument("--encoding-batch-size", type=int, default=64,
                    help="ModernBERT encoding batch size (only used when "
                         "gating is enabled)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    target = args.synth_examples_per_code
    if args.rows_per_node is not None:
        log.warning("--rows-per-node is deprecated; use --synth-examples-per-code")
        target = args.rows_per_node

    manifest = generate_corpus(
        synth_examples_per_code=target,
        payloads_path=args.payloads,
        output_dir=args.output,
        seed=args.seed,
        enable_diversity_gate=args.diversity_gate,
        enable_marginal_coverage_stop=args.marginal_coverage_stop,
        encoding_batch_size=args.encoding_batch_size,
    )
    print()
    print("=== Corpus generation complete ===")
    print(f"  total rows: {manifest['n_rows_total']}")
    print(f"  nodes covered: {manifest['n_nodes_covered']}")
    print(f"  nodes skipped (no generator): {manifest['n_nodes_skipped']}")
    print(f"  output: {args.output / 'synth_rows.jsonl'}")
    print(f"  manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
