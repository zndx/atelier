# Scenario-Oriented Documentation

## What Was Built

Four mdbook pages in `docs/src/scenarios/` modeled after the signals project's BDD documentation but adapted for Atelier's deployment-centric architecture.

### Pages Created

| Page | Purpose |
|------|---------|
| `overview.md` | Coverage matrix by domain, scenario counts by tier, why BDD for deployment |
| `testing.md` | Test infrastructure: tier system, running commands, feature organization, step discovery, config-driven BDD |
| `deployment.md` | Narrative walkthrough of all four CAI modalities (Project, AMP, Application, Studio) with d2 diagram and inline Gherkin showing what each scenario validates |
| `runtime-profile.md` | Deep dive on the runtime profile concept: failure modes it prevents, when to extend it, the import chain / script / config / migration checks |

### Approach vs Signals

The signals docs use coverage tables and infrastructure reference. The Atelier docs go further:

- **Narrative-first**: Each deployment modality section explains *why* the constraint exists before showing the scenario
- **Inline Gherkin**: Key scenarios are quoted directly in the docs so colleagues see the spec without opening feature files
- **Failure mode tables**: The runtime profile page maps each check to the specific deployment failure it prevents
- **d2 diagram**: Deployment modalities page includes a d2 relationship diagram

### Also Fixed

- Removed `multilingual = false` from `book.toml` (invalid field in mdbook 0.5.2)
- Disabled katex preprocessor (0.9.4 incompatible with mdbook 0.5.2, same as signals)
- Updated `architecture/overview.md` d2 diagram: "SQLite State DB" → "PostgreSQL" with PGlite tooltip
- Added Scenarios section to `SUMMARY.md`
