# Evolve Classification

End-to-end orchestration of Atelier's GEPA-shaped classification evolution
loop. Absorbs reviewer feedback artifacts (scoring xlsx), audits cosine
signal quality, evolves ontology enrichment text via reflective LLM
proposals, and reports lift against a drift-stable agent-mediated
reference verifier.

Runs **inside the App pod** where Qdrant (`127.0.0.1:6333`), PGlite
(`127.0.0.1:5440`), and the live taxonomy collection are reachable.
The Session pod does not have these on localhost.

This skill chains five existing scripts and one optional transform-layer
step (deferred until the apply path is built):

| Phase | Script | Purpose |
|---|---|---|
| 1 | (probe) | Environment readiness check |
| 2 | `scripts/update_reference_from_xlsx.py` | Absorb reviewer corrections |
| 3 | `scripts/audit_cosine_signal.py --live` | Cosine signal quality baseline |
| 4 | `scripts/enrichment_evolution.py --llm` | Reflective rewrite proposals |
| 5 | *(transform layer — deferred)* | Apply proposals + re-encode |
| 6 | (re-score) | Verify against drift-stable reference |
| 7 | (report) | Summarize state, deltas, next steps |

## Argument: $ARGUMENTS

Parse the argument for:

- **`run_id`** (required) — pipeline run directory under `build/results/`,
  e.g. `7bbe4533`. Provides classifications.json + scoring_summary.md
  for traces.
- **`--xlsx <path>`** (optional) — reviewer-flagged scoring xlsx to absorb
  via Phase 2. If absent, Phase 2 is skipped.
- **`--scope <name>`** (optional, default `full`) — one of:
    - `full` — Phases 1-7 in order. Resumes per saved state.
    - `audit-only` — Phases 1, 3, 7. No mutations.
    - `reference-only` — Phases 1, 2, 7. Reviewer absorption only.
    - `evolve-only` — Phases 1, 4, 7. Skip reference + audit.
    - `report` — Phase 7 only, read state from prior run.
- **`--cohort <name>`** (optional, default `umbrellas`) — enrichment-
  evolution cohort. Currently only `umbrellas` is wired.
- **`--mode <name>`** (optional, default `subset`) — taxonomy presentation
  mode for the reference-update LLM. `full` or `subset`. A/B settled in
  favor of `subset` on 2026-05-23.
- **`--dry-run`** — skip `--apply` on the reference update + skip the
  expensive Phase 4 LLM calls. Cheap inspection mode.

Examples:
- `evolve-classification 7bbe4533 --xlsx AtelierResultsVsPromptSolution1f2bad3eV4-0522.xlsx`
- `evolve-classification 7bbe4533 --scope audit-only`
- `evolve-classification 7bbe4533 --scope evolve-only --cohort umbrellas`

## Principles (non-negotiable)

Read these memories before starting:
- `feedback_mnemonics_with_codes.md` — dual-format references guard
  against structural drift; never apply corrections in mnemonic-only form.
- `feedback_no_silent_dst_degradation.md` — fallbacks are
  deployment-degraded; log loudly, tag in artifacts.
- `feedback_dynamic_annotations.md` — "100% coverage" not "N/N";
  vocabulary cardinality is runtime-selected.
- `project_enrichment_evolution_methodology.md` — GEPA mapping, mutation
  operators, phase 0-1 vs 2-4 boundary.
- `project_atelier_taxonomy_quirks.md` — known confusable clusters that
  inform cohort scoping.

Non-negotiable rules in this skill:

1. **State preservation.** Every phase writes to
   `build/evolution_state/<run_id>/state.json` with phase status,
   artifact paths, costs, and decisions. Resume reads this and skips
   completed phases.

2. **Back-pressure gates.** Before any phase that spends > $0.50 of LLM
   calls, print the expected cost and the decision criteria; halt
   pending acknowledgement when running interactively (skip the halt
   under `--scope full` when state already shows operator opted in).

3. **No silent applies on subtle corrections.** The reference update's
   color-aware routing already enforces this; do NOT add bypass
   arguments. Manual review queues exist for sibling distinctions,
   granularity loosenings, light-red taxonomy reviews, and yellow row
   notices.

4. **Reference is verifier-of-record.** Re-scoring is always against
   the dual-format reference using captured codes (drift-stable).
   Never re-resolve mnemonics for scoring.

5. **Resume-safe artifacts.** Every script invocation writes to a
   versioned output path; re-invoking does not overwrite the prior
   run's artifacts. Phase 2 uses xlsx_basename; Phase 4 uses
   cohort_<name>_v<N>.

---

## Phase 1 — Environment probe

```python
import json, os, sys
from pathlib import Path

sys.path.insert(0, "src")

run_id = "$run_id"  # parsed from $ARGUMENTS
run_dir = Path(f"build/results/{run_id}")
state_dir = Path(f"build/evolution_state/{run_id}")
state_dir.mkdir(parents=True, exist_ok=True)
state_path = state_dir / "state.json"

if state_path.exists():
    state = json.loads(state_path.read_text())
    print(f"Resuming evolution state from {state_path}")
else:
    state = {"run_id": run_id, "phases": {}, "decisions": {}}
    print(f"Fresh evolution state for run {run_id}")

# Probe sequence — every check must pass before proceeding
import socket

def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

checks = []

# 1. Run dir + classifications.json
cls_path = run_dir / "classifications.json"
checks.append(("classifications.json", cls_path.exists(),
               f"missing {cls_path} — provide a valid run_id"))

# 2. Agent-mediated reference (dual format)
ref_path = Path("build/data/agent_mediated/agent_mediated.json")
ref_is_dual = False
if ref_path.exists():
    ref = json.loads(ref_path.read_text())
    sample = next(iter(ref.values()), None)
    ref_is_dual = isinstance(sample, dict) and "mnemonic" in sample
checks.append(("agent_mediated.json", ref_path.exists(),
               f"reference missing at {ref_path}"))
checks.append(("reference is dual-format", ref_is_dual,
               "run scripts/migrate_reference_to_dual_format.py first"))

# 3. Taxonomy cache
tax_path = Path("build/data/taxonomy/taxonomy_cache.json")
checks.append(("taxonomy cache", tax_path.exists(),
               "run update_reference_from_xlsx.py --refresh-taxonomy"))

# 4. Qdrant reachable
checks.append(("Qdrant on 127.0.0.1:6333", _tcp_reachable("127.0.0.1", 6333),
               "Qdrant not reachable — are you in the App pod?"))

# 5. PGlite reachable
checks.append(("PGlite on 127.0.0.1:5440", _tcp_reachable("127.0.0.1", 5440),
               "PGlite not reachable — are you in the App pod?"))

# 6. hive-poc connection (cmldata)
hive_ok = False
try:
    import cml.data_v1 as cmldata
    cmldata.get_connection("hive-poc")
    hive_ok = True
except Exception as exc:
    hive_msg = f"cmldata get_connection failed: {exc}"
else:
    hive_msg = ""
checks.append(("hive-poc connection", hive_ok, hive_msg or "(transient)"))

# 7. LLM credentials (one path must work)
from atelier.config import load_config
cfg = load_config()
has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
has_bedrock = (cfg.agent_model or "").startswith("arn:")
checks.append(("LLM credentials", has_api_key or has_bedrock,
               "neither ANTHROPIC_API_KEY nor ATELIER_AGENT_MODEL ARN set"))

# Report
print("\nEnvironment probe:")
all_pass = True
for name, ok, fail_msg in checks:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}")
    if not ok:
        print(f"      ↳ {fail_msg}")
        all_pass = False

if not all_pass:
    print("\nFAIL: environment not ready. Address the gaps above.")
    sys.exit(2)

state["phases"]["1_probe"] = {"status": "complete", "checks": [c[0] for c in checks]}
state_path.write_text(json.dumps(state, indent=2))
print("\nPhase 1 complete.")
```

If any check fails: halt with the specific remediation. Common failures:
- Not in App pod → reschedule the invocation from the Web Terminal Agent
- Reference not dual-format → run the migration first (one-line)
- Taxonomy cache stale → `--refresh-taxonomy` rebuilds it from hive-poc
- LLM credentials absent → check secrets handoff to the pod

---

## Phase 2 — Reference update from xlsx

**Skip when `--xlsx` is not provided OR scope is `audit-only` / `evolve-only`.**

This phase absorbs reviewer-flagged corrections from a scoring xlsx
using the color-aware extraction + correction-type routing:

| Color | Route | Apply policy |
|---|---|---|
| Red (`FFFF0000`) on col D | LLM correction prompt | Auto-apply iff high-conf + subtree_correction/granularity_tightening |
| Light red (`FFF4CCCC`) on col D | LLM taxonomy-review prompt (allows TAXONOMY_GAP) | Never auto-apply; goes to taxonomy_review queue |
| Yellow (`FFFFFF00`) on cols A-E | No LLM call | Goes to row_review queue |

**Cost gate: ~$1-2 for a 45-row xlsx. Halt-and-confirm if `> $2`.**

```python
xlsx = "$xlsx"  # parsed from $ARGUMENTS
mode = "$mode"  # default subset
dry_run = "$dry_run"  # boolean
apply_flag = "" if dry_run else "--apply"

# Cost estimate
import openpyxl
wb = openpyxl.load_workbook(xlsx, data_only=False)
n_flagged = 0
for sn in wb.sheetnames:
    if sn == "Overview": continue
    ws = wb[sn]
    for row in ws.iter_rows(min_row=2):
        d = next((c for c in row if c.column_letter == "D"), None)
        flagged_d = d and d.fill and d.fill.start_color and \
            str(d.fill.start_color.rgb or "") in ("FFFF0000", "FFF4CCCC")
        flagged_yellow = any(
            c.fill and c.fill.start_color and
            str(c.fill.start_color.rgb or "") == "FFFFFF00"
            for c in row if c.column_letter in "ABCDE"
        )
        if flagged_d or flagged_yellow:
            n_flagged += 1
est_cost = n_flagged * 0.04  # Opus, subset mode, single call per row
print(f"\nPhase 2: ~{n_flagged} flagged rows × ~$0.04 = ~${est_cost:.2f}")
if est_cost > 2.0:
    print("WARNING: estimated cost exceeds $2. Halt and confirm if interactive.")

# Invoke
import subprocess
cmd = [
    "python", "scripts/update_reference_from_xlsx.py", xlsx,
    "--mode", mode, "--run-dir", f"build/results/{run_id}",
]
if apply_flag:
    cmd.append(apply_flag)
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    sys.exit(3)

# Parse status counts from corrections.json
xlsx_base = Path(xlsx).stem
corr_path = Path(f"build/data/agent_mediated/corrections_{xlsx_base}.json")
corr = json.loads(corr_path.read_text())
from collections import Counter
status_counts = Counter(r["status"] for r in corr["records"])

state["phases"]["2_reference_update"] = {
    "status": "complete",
    "xlsx": xlsx,
    "n_flagged": n_flagged,
    "status_counts": dict(status_counts),
    "corrections_path": str(corr_path),
    "review_queues": {
        "manual_review": f"build/data/agent_mediated/manual_review_{xlsx_base}.md",
        "taxonomy_review": f"build/data/agent_mediated/taxonomy_review_{xlsx_base}.md",
        "row_review": f"build/data/agent_mediated/row_review_{xlsx_base}.md",
    },
    "applied": status_counts.get("high_apply", 0) if not dry_run else 0,
}
state_path.write_text(json.dumps(state, indent=2))
print(f"\nPhase 2 complete. Applied {state['phases']['2_reference_update']['applied']} corrections.")
```

After Phase 2: surface the review queues to the operator. Mention:
- N items in manual_review (operator decides apply or skip)
- N items in taxonomy_review (curator decides taxonomy edits)
- N items in row_review (informational only)

---

## Phase 3 — Cosine signal audit (baseline)

**Skip under `reference-only` scope.**

Runs the three-audit signal-quality probe against the run's
classifications.json plus a live K=25 Qdrant query for the top
confusable clusters.

```python
import subprocess
cmd = [
    "python", "scripts/audit_cosine_signal.py",
    f"build/results/{run_id}", "--live",
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    sys.exit(3)

audit_path = Path("build/diag/cosine_signal_audit.json")
audit = json.loads(audit_path.read_text())

# Surface key metrics
a1 = audit["audit_1_subtree_correctness"]
a3 = audit["audit_3_default_pick_bias"]

productive_pct = (
    100 * (a1["subtree_correct"] + a1["exact_internal"])
    / max(a1["total_internal_top1_with_ref"], 1)
)
cross_pct = (
    100 * a1["cross_subtree"]
    / max(a1["total_internal_top1_with_ref"], 1)
)

top_centroid = a3["top10"][0] if a3["top10"] else None
centroid_msg = (
    f"{top_centroid['code']} ({top_centroid['pct']}%)"
    if top_centroid else "—"
)

print(f"\nAudit summary:")
print(f"  Productive internal-node picks: {productive_pct:.1f}%")
print(f"  Cross-subtree (drift) picks:    {cross_pct:.1f}%")
print(f"  Top centroid code (bias):       {centroid_msg}")

state["phases"]["3_audit"] = {
    "status": "complete",
    "audit_path": str(audit_path),
    "report_path": "build/diag/cosine_signal_audit.md",
    "productive_pct": productive_pct,
    "cross_subtree_pct": cross_pct,
    "top_centroid_code": top_centroid["code"] if top_centroid else None,
    "top_centroid_pct": top_centroid["pct"] if top_centroid else None,
}
state_path.write_text(json.dumps(state, indent=2))
```

**Decision heuristics for the operator (and for the agent in autonomous mode):**

- **Cross-subtree > 70%** → enrichment evolution is the right next move
  (Phase 4). Most internal-node picks are noise; reflective rewrites
  should attack the centroid codes.
- **Cross-subtree 40-70%** → enrichment evolution AND consider re-running
  the audit with a wider K to catch rank-3 confusables that the rewrite
  should target.
- **Cross-subtree < 40%** → signal quality is decent. Hold on evolution;
  focus on the ML classifiers (SVM/CatBoost) instead — they may be
  contributing more error than cosine at this point.
- **Top centroid > 15% of all columns** → that tag's enrichment is too
  generic; rewrite is high-leverage. Pass the code to Phase 4 as a
  forced cohort member if it isn't already in the umbrella cohort.

---

## Phase 4 — Enrichment evolution (Phase 0+1 of GEPA loop)

**Skip under `reference-only` or `audit-only` scope.**

Runs the GEPA-shaped reflective rewrite proposal step against the
chosen cohort. Default cohort = 32 umbrella-template nodes.

**Cost gate: ~$1-2 for the umbrella cohort × Opus subset mode.
Halt-and-confirm under interactive mode if the cohort is wider.**

```python
cohort = "$cohort"  # default "umbrellas"
dry_run = "$dry_run"
llm_flag = [] if dry_run else ["--llm"]

# Cost estimate based on cohort size
if cohort == "umbrellas":
    n_nodes = 32
else:
    print(f"WARNING: cohort '{cohort}' size unknown; assuming 50 nodes")
    n_nodes = 50
est_cost = n_nodes * 0.05  # Opus, ~3K-4K input + 600 output tokens per node
print(f"\nPhase 4 cohort '{cohort}': ~{n_nodes} nodes × ~$0.05 = ~${est_cost:.2f}")
if dry_run:
    print("  --dry-run: skipping LLM calls; producing prompts + traces only.")

cmd = [
    "python", "scripts/enrichment_evolution.py",
    f"build/results/{run_id}",
    "--cohort-name", cohort,
] + llm_flag

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    sys.exit(3)

# Find the latest cohort_<cohort>_v<N> directory
import re
ee_root = Path("build/enrichment_evolution")
versions = sorted(
    [d for d in ee_root.iterdir() if d.is_dir() and d.name.startswith(f"cohort_{cohort}_v")],
    key=lambda d: int(d.name.rsplit("_v", 1)[1]),
)
latest = versions[-1] if versions else None
if latest is None:
    print(f"ERROR: no cohort_{cohort}_v* directory produced")
    sys.exit(3)

candidates_path = latest / "candidates.json"
candidates = json.loads(candidates_path.read_text())

state["phases"]["4_evolution"] = {
    "status": "complete",
    "cohort": cohort,
    "version": latest.name,
    "n_nodes": len(candidates["candidates"]),
    "candidates_path": str(candidates_path),
    "summary_path": str(latest / "summary.md"),
    "nodes_dir": str(latest / "nodes"),
    "model": candidates.get("model"),
    "llm_called": candidates.get("llm_called", False),
}
state_path.write_text(json.dumps(state, indent=2))
print(f"\nPhase 4 complete. Proposals at {latest}.")
```

After Phase 4: surface the per-node markdown artifacts. Each one
contains: current definition + children + attraction statistics +
proposed rewrite + targeted-edit traceability. These are
operator-reviewable; the operator picks an acceptance set before
Phase 5.

---

## Phase 5 — Transform-layer application *(deferred)*

**This phase is not yet implemented.** The transform-layer machinery
that converts accepted enrichment proposals into structured records,
re-encodes the affected ColBERT vectors, and writes to a versioned
Qdrant collection is still in design. See conversation 2026-05-23 for
the design sketch.

Until built, Phase 5 prints a placeholder report and exits without
applying. The operator can review Phase 4 proposals manually in
`build/enrichment_evolution/cohort_<cohort>_v<N>/nodes/` and apply via
whatever side-channel they have for editing the enrichment payloads
(direct Qdrant payload PATCH, regeneration via `enrich_annotations.py`,
etc.).

```python
print("\nPhase 5: transform layer not yet wired.")
print(f"  Review proposals at: {state['phases']['4_evolution']['nodes_dir']}/")
print(f"  Pick acceptance set; apply via your enrichment-edit pathway.")
state["phases"]["5_apply"] = {"status": "deferred"}
state_path.write_text(json.dumps(state, indent=2))
```

When Phase 5 lands: it should snapshot the current Qdrant collection
(versioned), apply accepted proposals via re-encoding, swap the
"current" pointer in `taxonomy_registry` only after Phase 6 verifies
no regression.

---

## Phase 6 — Verification re-score *(partial, baseline-only until Phase 5)*

**Until Phase 5 is built, this phase only re-scores the current
classifications.json against the (possibly updated) reference.** That
captures the lift from Phase 2 alone, not from enrichment evolution.

```python
ref_path = Path("build/data/agent_mediated/agent_mediated.json")
ref = json.loads(ref_path.read_text())

# Extract captured codes (dual-format)
ref_codes = {}
for qkey, entry in ref.items():
    if isinstance(entry, dict) and entry.get("code"):
        ref_codes[qkey] = entry["code"]

cls = json.loads(cls_path.read_text())
scorable = 0
strict = 0
for c in cls:
    qkey = f"{c.get('table_name')}.{c.get('column_name')}"
    r = ref_codes.get(qkey)
    if not r: continue
    scorable += 1
    if c.get("predicted_code") == r:
        strict += 1
pct = 100 * strict / max(scorable, 1)
print(f"\nDrift-stable re-score: {strict}/{scorable} = {pct:.2f}%")

state["phases"]["6_rescore"] = {
    "status": "complete_baseline_only",
    "strict": strict,
    "scorable": scorable,
    "strict_pct": pct,
    "note": "Phase 5 not yet wired; this is reference-update-only lift",
}
state_path.write_text(json.dumps(state, indent=2))
```

When Phase 5 lands: this phase should re-encode and re-run the pipeline
on the affected subset (cheap: just the columns whose targeted-nodes
changed), then compute the delta over the Phase 6 baseline captured
above. That's the verifier signal that drives Phase 5's per-transform
accept/reject loop.

---

## Phase 7 — Report

```python
print("\n" + "=" * 60)
print(f"Evolve-classification report — run {run_id}")
print("=" * 60)
for k in sorted(state["phases"].keys()):
    p = state["phases"][k]
    print(f"\n{k}:  status={p['status']}")
    for kk, vv in p.items():
        if kk in ("status",): continue
        if isinstance(vv, dict):
            print(f"  {kk}:")
            for kkk, vvv in vv.items():
                print(f"    {kkk}: {vvv}")
        else:
            print(f"  {kk}: {vv}")

# Recommended next moves
print("\nRecommended next moves:")
if "2_reference_update" in state["phases"]:
    p = state["phases"]["2_reference_update"]
    queues = p.get("review_queues", {})
    if queues:
        print("  - Review the manual_review queue + apply accepted items")
        print("  - Triage the taxonomy_review queue with the ontology curator")
if "4_evolution" in state["phases"]:
    p = state["phases"]["4_evolution"]
    print(f"  - Review {p['n_nodes']} enrichment proposals at {p['nodes_dir']}/")
    print( "  - When Phase 5 transform layer lands, apply the acceptance set + verify")
if "3_audit" in state["phases"]:
    p = state["phases"]["3_audit"]
    if p.get("cross_subtree_pct", 0) < 40:
        print("  - Cross-subtree rate is low; consider shifting focus to SVM / CatBoost work")
    if (p.get("top_centroid_pct") or 0) > 15:
        print(f"  - Centroid bias on {p['top_centroid_code']} is the highest-leverage rewrite target")

print("\nState persisted to", state_path)
```

---

## Resume semantics

`--scope full` re-running on the same `run_id` reads the prior state.json
and skips phases whose `status == "complete"`. To force a phase to
re-run: delete its entry from state.json before invocation. To start
fresh: delete the entire `build/evolution_state/<run_id>/` directory.

## State file shape

```json
{
  "run_id": "7bbe4533",
  "phases": {
    "1_probe": {"status": "complete", "checks": [...]},
    "2_reference_update": {"status": "complete", "xlsx": "...", ...},
    "3_audit": {"status": "complete", "productive_pct": ..., ...},
    "4_evolution": {"status": "complete", "cohort": "umbrellas", ...},
    "5_apply": {"status": "deferred"},
    "6_rescore": {"status": "complete_baseline_only", ...},
    "7_report": {"status": "complete"}
  },
  "decisions": {}
}
```

## Cost envelope (typical full run)

| Phase | Cost | Time |
|---|---|---|
| 1 — probe | $0 | <5s |
| 2 — reference update (45 rows × Opus) | ~$1-2 | ~3-5 min |
| 3 — audit (live K=25) | ~$0 (Qdrant only) | ~10-30s |
| 4 — evolution (32 nodes × Opus) | ~$1-2 | ~3-5 min |
| 5 — transform apply (deferred) | TBD | TBD |
| 6 — rescore (baseline-only until Phase 5) | $0 | <5s |
| 7 — report | $0 | <5s |
| **Full run** | **~$2-4** | **~8-15 min** |

## Failure modes + recovery

- **LLM rate limit / network blip** → script raises; resume re-invokes
  with same args; the per-row try/except already isolates failures so
  Phase 2 and Phase 4 are robust to partial failures.
- **Reference file corrupted by interrupted apply** → restore from
  `agent_mediated.json.bak` (always written before any apply).
- **Qdrant unreachable** → Phase 3 live mode degrades to Audits 1+3
  only. Phase 4 doesn't need Qdrant. Phase 5 (when built) does.
- **Cost overrun** → halt at the back-pressure gate before the next LLM
  phase; resume after operator decision.

## What this skill does NOT do

- Does not invoke a full pipeline re-run (that's `bin/start-app.sh`
  territory + the operator restart workflow).
- Does not write to `default.annotations` — that source is immutable;
  all curation goes via the transform layer (Phase 5, deferred).
- Does not promote a new Qdrant collection to "current" — that's a
  taxonomy_registry write that Phase 5 will own once built.
- Does not curate the agent-mediated reference from scratch — use
  `/curate-agent-mediated` for that. This skill only absorbs reviewer
  corrections via xlsx feedback.

## Related skills

- `/bootstrap-environment` — first-time setup; enrichment + curation +
  initial SVM training.
- `/curate-agent-mediated` — per-table reference curation from scratch.
- `/train-svm` — synthetic-corpus SVM training after the reference is
  stable.

This skill (`/evolve-classification`) is the *steady-state* iteration
mode that runs against an existing pipeline run + reviewer feedback,
not the initial bootstrap.
