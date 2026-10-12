# Qwen3.8 thinking authors a synthetic relational corpus

The Metaflow training path still uses the procedural in-situ generator
in `insitu_corpus.py`. This slice is the separate test of whether
local Qwen3.8 thinking can author the tables through a grok ACP
harness. Upstream column labels score that output after the process
exits. They are not in the workpack.

## Environment

`src/atelier/classify/qwen_synth_harness.py` writes an isolated
directory. The catalogue is an ontology slice with columns `code`,
`label`, `abbrev`, `parent_code`, and `description`, plus ancestors
of the requested codes. `example_values` and `common_names` are not
copied. `TASK.md` lists the codes and the row count. The skill
`.grok/skills/synthetic-relational-corpus/SKILL.md` and the agent
`.grok/agents/relational-synth.md` are copied into that directory.

The process is `grok -p` with `--agent`, `--sandbox strict`,
`--max-turns`, and tools `read_file`, `write`, `list_dir`. Shell, web,
and MCP tools are removed. Read rules deny the Atelier checkout and
`~/.grok/sessions`. On this host `strict` refuses to start until
`bwrap` is on `PATH`; the harness prepends a runnable nix-store
bubblewrap when the system image does not provide one. Vendor skill
trees are turned off with `GROK_CLAUDE_SKILLS_ENABLED` and
`GROK_CURSOR_SKILLS_ENABLED`. The workdir is outside the git checkout
so repo skills are not discovered. An isolated one-word probe used
5035 input tokens; the same probe from the Atelier checkout used
20279.

The model id `qwen-thinking` is `Qwen/Qwen3.8-27B` at
`http://127.0.0.1:8081/v1` with `api_key = "local"`. That block lives
in the user grok config. A `GROK_CONFIG` overlay drops `[model.*]`,
which is why an overlay file did not register the model. The default
model stays `grok-4.7`. `qwen-thinking-cerebras` points at
`qwen-3.8-27b` on `https://api.cerebras.ai/v1` and reads
`CEREBRAS_API_KEY`. It was not called.

## Local trial

Codes `SDG.ICE.FINANCIAL_REPORT` and
`SDG.MATERIAL_ENTITY.FINANCIAL_FUND`, 8 data rows, turn cap 12.
The run took 123 seconds, 5 turns, and exit code 0. Grok reported
model `Qwen/Qwen3.8-27B`, 31750 input tokens, 4908 output tokens, and
`reasoning_tokens` 0, so this client did not surface a separate
reasoning trace from the thinking endpoint.

Ten tool calls, every path under the workpack: read the skill,
`ontology.csv`, and `TASK.md`; write
`out/tables/financial_report.csv`, `out/tables/financial_fund.csv`,
and `out/manifest.json`; re-read those three. The session log has no
`naming_map`, `base_rows`, or checkout path.

Both requested codes appear once. `parse_ok` is true and coverage is
1. Name-token Jaccard against the footprint columns for those codes
is 0.708. Value-token Jaccard is 0.055. The tables the model wrote:

```csv
financial_report_id,report_title,period_start,period_end,published_on,currency,status
FR-2026-0001,Quarterly Operations Review Q1 2026,2026-01-01,2026-03-31,2026-04-28,USD,published
```

```csv
fund_id,fund_name,custodian,opened_on,currency,committed_amount,status
FND-101,Harbor Infrastructure Fund,Meridian Trust,2024-03-11,USD,25000000,active
```

Sibling columns are operational fields and are omitted from the
manifest. Scoring calls `load_labeled_columns` only after generation.
The workpack and score stayed in `/tmp/qwen-synth-run`.

123 seconds for two codes is fast enough to iterate the harness
locally. Cerebras remains the budgeted route and was not used.
Precondition still trains from `prepare_insitu`.
