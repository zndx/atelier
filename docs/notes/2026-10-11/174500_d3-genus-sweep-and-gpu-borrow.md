# Genus-first d3 sweep, then give the GPUs back

2026-10-11. Follow-up to `170843_d3-sweep-adoption.md`.

## The first Choice is the genus, not the point

Live Qdrant on `:6355`:

- `sdg_aperture`: 33 points. Payloads still carry the old notations
  (`16.2` LIMS, `19.1` MFG) and constituent chords.
- `sdg_domains`: 1,065 points. Same IRIs have moved (LIMS is code `9`,
  with four children). 89 nodes are structural genera (broader and
  narrower). 52 of those sit at code depth ≤ 2. Widest child set in
  that cut is Material Entity at 91.
- Four aperture points are absent from `sdg_domains`: `FINTECH`,
  `BIOTECH`, `MBSE`, `MFG_LOGISTICS`.

The 33 points mix genus chords and species, and the constituent lists
are MaxSim neighbors. A Choice over those points would ask d3 a
different question from the tree, and it would feed the maxsim channel
into the sweep. The Choice criteria should be the genus-level SKOS
terms (code, prefLabel, definition). The points stay a retrieval index.

That set fits one Choice (the cap is 255). It is the right first sweep.
The distribution over genera is the cross-subtree mass `llm_to_mass`
needs. The walk then descends children of the genera that carry mass.
Those child codes have to be nodes of the NHSVM head (`code_to_idx`).
The pinned `annotations.csv` has no LIMS/MFG/FINTECH strings, and the
cached heads are keyed by `SDG.*` paths. Gate the sweep on that
membership. A genus the head does not contain has no aligned subtree.

## GPUs

Six RTX 4090s, 24 GB. Standing occupancy: vLLM TP=4 on 0–3 (the
Qwen3.8 thinking replica, ~21 GB each), a Gaius Python process on 4,
`moshi-server` on 5. YuniKorn SoR is
`signals/config/scheduler/federation-queues.yaml`: heavy guaranteed 4
and fenced, parent max 6, leftover pair shared by extract / light /
medium / agent-rtc. Peers request a queue share. They do not write
that file. Atelier is an allowed peer.

d3 BF16 is about 52 GB and the runtime is one device
(`D3.from_pretrained(..., device=cuda:0)`). Six independent BF16
endpoints will not start. Six endpoints means six data-parallel
processes, one quantized replica per card, columns sharded across
them, each server's lock covering only its own shard. Prove one 4-bit
(or better) checkpoint loads in that runtime, on one 4090, on the
widest child-set prompt, before any gang claim.

## Phase the borrow

Heavy is the thinking revisit. A sweep that holds all six cards cannot
call `thinking` until it gives them back. Split `ClassificationFlow`'s
single `sweep` step:

1. Airflow activity asserts a temporary share: heavy floor to 0,
   decide leaf guaranteed 6. `all_done` restores the recorded floors
   on success and on failure.
2. Six System One processes. Metaflow fans columns out. First question
   is the genus Choice. Walk stays on the same endpoint.
3. Stop the six processes. Restore heavy TP=4, the Gaius process, and
   Moshi.
4. Thinking revisit runs on the restored heavy endpoint, on the
   disagreement set only.

Do not interleave the revisit inside the borrow. Do not preempt an
active Hermes session without the existing agent-rtc delay.
