# Encrypted Deployment Defaults (SOPS + age)

Atelier ships with **encrypted deployment defaults** so a CAI operator
can stand up a working instance by entering only four environment
variables — their two AWS Bedrock credentials, a direct Anthropic API
key (for overwatch), plus a single age private key that unlocks
everything else.

## Why

Every CAI deployment needs a dozen-ish environment variables: Bedrock
model ARNs, Atlas / Ranger URLs, feature toggles, governance flags,
subagent model IDs, and — for UAT runs — a ground-truth CSV for
accuracy measurement. Most of those values are identical across
every deployment of the same Atelier release; only the AWS credentials
and the Anthropic key are operator-specific. Rather than documenting
a long checklist for every customer, we **encrypt the defaults and
the ground-truth fixture** into the repository with
[SOPS](https://getsops.io) and ship one key alongside the deployment.

The operator paste-sets the key; everything else is already wired up.

## Operator workflow (what to tell your CAI users)

Set four environment variables on the CAI Application, then start it:

| Name | Value | Source |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Bedrock access key | your AWS / IAM team |
| `AWS_SECRET_ACCESS_KEY` | Bedrock secret | your AWS / IAM team |
| `ANTHROPIC_API_KEY` | direct Anthropic API key | Anthropic Console |
| `SOPS_AGE_KEY` | full `AGE-SECRET-KEY-1…` string | provided out-of-band by the Atelier maintainer |

On startup, `bin/start-app.sh` runs the shared
`bin/bootstrap-secrets.sh` utility, which decrypts both `.env.cai.enc`
(dotenv defaults) and `features/fixtures/ground_truth.csv.enc`
(meta-tagging answer key) with the age key you provided. The dotenv
values source into the environment where HOCON's `${?VAR}` substitution
picks them up; the decrypted CSV materializes at
`build/data/ground_truth.csv` and `ATELIER_GROUND_TRUTH_URI` points at
it so `evaluation_report.json` carries real accuracy numbers. No
per-customer checklist to maintain.

**Overrides still work.** Any explicit `ATELIER_*` env var on the CAI
Application wins over the encrypted default — so an operator who
wants a different Bedrock ARN just sets `ATELIER_AGENT_MODEL` directly
and that value takes precedence.

### Alternative: pointing at a key file

If the operator already has the age key on disk (e.g. mounted from a
secret store), they can set `SOPS_AGE_KEY_FILE=/path/to/key.txt`
instead of pasting the key content. `bin/start-app.sh` supports both.

## Maintainer workflow

The age **public** key is committed in `.sops.yaml`; the **private**
key is held by the Atelier maintainer and distributed out-of-band to
each CAI operator.

### First-time setup

Place your age private key at `~/.config/sops/age/keys.txt` — the
public key must match the `age: age1…` line in `.sops.yaml`. The
devenv shell provides both `sops` and `age` binaries.

### Editing defaults

```bash
just decrypt-secrets          # .env.cai.enc → .env.cai (plaintext, gitignored)
$EDITOR .env.cai              # add / change values
just encrypt-secrets          # .env.cai → .env.cai.enc
git add .env.cai.enc
git commit -m "chore: update CAI deployment defaults"
```

The plaintext `.env.cai` is excluded by `.gitignore`; only the
encrypted `.env.cai.enc` is tracked. SOPS encrypts each value
independently, so diffs show which keys changed even though their
values are opaque.

### Editing the ground-truth fixture

The meta-tagging answer key (what `evaluation_report.json` compares
predictions against) ships encrypted under the BDD fixtures tree so
committed secrets live with the corpus they validate.

```bash
# From the maintainer's reviewer xlsx
uv run python -m atelier.overwatch.ingest_ground_truth \
    ~/path/to/Atelier_Results_Default_DB_4-16.xlsx \
    --out build/data/ground_truth.csv

# Encrypt into features/fixtures/ and commit the ciphertext only
just encrypt-gt
git add features/fixtures/ground_truth.csv.enc
git commit -m "chore: update ground-truth answer key"
```

To inspect the current key without re-running the xlsx ingest:

```bash
just decrypt-gt               # decrypts into build/data/ground_truth.csv
$PAGER build/data/ground_truth.csv
```

Both the plaintext CSV (in `build/`) and `.env.cai` are ignored by
git; only the `.enc` ciphertexts are tracked.

### Rotating the key

```bash
age-keygen -o new-key.txt                                    # generate replacement pair
# update .sops.yaml: replace the age: age1... line with the new public key
sops updatekeys .env.cai.enc                                 # re-encrypt deployment defaults
sops updatekeys features/fixtures/ground_truth.csv.enc       # AND the ground-truth fixture
git commit -am "chore: rotate CAI deployment key"
# distribute the new private key to operators via the same out-of-band channel
```

`sops updatekeys` rewrites the encrypted file's recipient list in
place — nothing about the plaintext values changes, so this is a
zero-content-drift rotation. Run it against **every** encrypted
artifact so the new key unlocks the whole set.

### Adding a second recipient (e.g. ops team shared key)

Add a second `age:` entry under the matching `creation_rules` block
in `.sops.yaml`, then run `sops updatekeys .env.cai.enc`. Either
private key will decrypt.

## How this fits with HOCON

SOPS only populates environment variables. HOCON (`config/base.conf`)
already treats all configuration as environment-overridable via the
`${?VAR}` pattern:

```hocon
agents {
  model = "claude-opus-4-7"
  model = ${?ATELIER_AGENT_MODEL}     # env wins when set
}
```

SOPS decryption runs **before** the gRPC server loads HOCON, so from
HOCON's perspective the encrypted values are just ordinary environment
variables.

### What belongs in `.env.cai.enc` vs `config/base.conf`

- **`.env.cai.enc`** — deployment-specific defaults that differ
  between environments but aren't operator secrets per se (model
  ARNs, Knox endpoints, feature toggles, subagent IDs). Values that
  are *derivable from context* and you don't want every operator
  to rediscover.
- **`config/base.conf`** — true defaults that hold for *every*
  deployment; structural knobs that belong in source control in
  plaintext (pipeline thresholds, port numbers, fusion strategy).
- **Operator-entered env vars** — genuine per-deployment secrets
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, the `SOPS_AGE_KEY`
  itself). These never live in the repository.

## Security notes

- `SOPS_AGE_KEY` decrypts **only this project's `.env.cai.enc`.**
  Losing it costs you these defaults; gaining it grants no AWS,
  Cloudera, or third-party privilege on its own.
- Each customer should get the same age private key (defaults are
  identical across deployments) — per-customer secrets, if any, stay
  in the CAI Application's own environment variables.
- Rotate the key whenever a recipient leaves the operator pool.
- The age public key in `.sops.yaml` is intentionally committed;
  public keys are meant to be public.

## Reference

- `.sops.yaml` — recipient rules (covers `.env.cai.enc` + `features/fixtures/*.csv.enc`)
- `.env.cai.enc` — encrypted deployment defaults (committed)
- `features/fixtures/ground_truth.csv.enc` — encrypted ground-truth CSV (committed)
- `bin/bootstrap-secrets.sh` — shared decrypt utility; runs from `bin/start-app.sh`, `devenv` enterShell, and `just bootstrap-secrets`
- `bin/start-app.sh` — CAI startup; invokes bootstrap-secrets then sources `.env.cai`
- `justfile` helpers:
  - `bootstrap-secrets` — run the shared decrypt utility
  - `decrypt-secrets` / `encrypt-secrets` — dotenv editing workflow
  - `decrypt-gt` / `encrypt-gt` — ground-truth CSV editing workflow
- `devenv.nix` — provides `sops` + `age` in the dev shell; runs bootstrap in `enterShell`
- [SOPS docs](https://getsops.io) · [age docs](https://age-encryption.org)
