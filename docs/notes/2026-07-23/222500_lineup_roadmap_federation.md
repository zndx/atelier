# Lineup roadmap — Atelier presentation → logical alignment → federated lineup

**Date:** 2026-07-23 (addendum to `220252_keiretsu_dark_mode_direction.md`)
**Framing (RH):** Ægir was the *advance* work; Atelier is where the standard
gets set — the patterns prove template-worthy by surviving a second product.

## Sequence

1. **Keiretsu visual adoption** (decision note §"Adoption sketch") — tokens,
   `data-mode`, AntD bridge, Status proof screen.
2. **Atelier lineup presentation**, seeded from the currently-unlinked
   Landing top-row cards (`Landing.tsx:249-280` — Skills / Entities / Terms;
   the other cards already link out). Mapping: Terms → Lexicon-lens trail
   over the SDG vocabulary (sibling of aegir `lens/terms`); Entities →
   dataset/schema lens; Skills → skill catalog as notes. Card presentation
   deliberately fluid until the visual theme settles.
3. **Logical alignment** (Ægir ⟷ Atelier): one shared note contract — id
   scheme, kinds, wikilink grammar, viz-attachment fields, roots-as-refs.
   Same artifact as the consolidation plan's note-contract doc (template
   `build/ZNDX_CONSOLIDATION_PLAN.md` §4): alignment spec ≡ template
   deliverable. Kasten-prefixed ids are already federation-shaped
   (`<root>/<kasten>/<note>`).
4. **Federated lineup** — cross-project trails over `zndx.engine.v1`
   (signals-protocol): a wikilink in an Atelier panel resolves a note from
   Ægir's kasten and vice versa, over the authenticated, capability-
   negotiated engine channel. Substrate already present: signals-protocol
   submodule (proto + specification), `src/zndx/engine/v1/` stubs,
   dual registration on both engines (running-note §10 — born from the
   mirrored-proto ≠ interoperable-contract federation finding).

## Verify-path → verify-HermiT (RH sharpening, same evening)

Federated verify-path is HermiT over the ontology SPANNING both projects:
**SDG proper + each project's extension + the trail's classification
assertions.** The mechanism: relational entities are classified against
*in-situ* SKOS annotations (Atlas glossaries — already read by
`governance/atlas_source.py:read_glossary`, surface forms included), which
are subsumed into SDG as a **project-specific extension module** — the
`load_skos(vocab, overlays=…)` composition in aegir's `domain_index`
promoted to a named sdg extension namespace (blending precedent:
`meta_tagging_overlay.build_blended_vocabulary`). The DST sdg: name-lock
means pipeline outputs are sdg:-coded assertions already — trail panels
carry IRIs by construction.

Payoff: cross-project contradiction becomes FORMALLY DETECTABLE — the same
entity annotated in situ (Ægir Atlas) vs classified by Atelier's pipeline,
if inconsistent, surfaces as unsatisfiability pinned to a browsable trail.

Open design point (Ægir's call, we hand the requirements recommendation):
the **SKOS→OWL lifting policy**. `skos:broader` ≠ `rdfs:subClassOf`;
HermiT sees OWL-DL, not annotations. Lift `subClassOf` only where the
in-situ hierarchy is genuinely taxonomic; otherwise use SDG's
property-centric idiom (`classifies`/`describesProperty`/`hasValueType`/
`hasUnit`). Naive lifting fails both ways: vacuous reasoning (annotations
invisible) or spurious inconsistency (broader lifted wholesale).
Extensions stay namespaced modules imported at reasoning time; promotion
into SDG proper is Ægir's ownership per the shared-surface directive.

## Why this beats upstream FedWiki structurally

FedWiki federates static JSON pages over bare HTTP. Ours federates over a
protocol that also carries inference and live instruments: remote panels
can be LIVE (an Ægir chord rendering inside an Atelier trail with its own
session theme), and ⚖ verify-path extends to trails whose panels span both
KBs — federated reasoning over a federated trail. Engine-level facilities
(identity, capability discovery, GPU-aware co-tenancy) come for free.

## The zndx triad (RH, same evening)

**Ontology in Ægir · Classification in Atelier · Knowledge in Gaius** — the
classical epistemic pipeline, distributed: what can exist (SDG/SKOS frame) →
what is this (DST belief/plausibility judgment) → what we know (Gaius: KB as
navigable spatial layouts, TDA 19×19 projection, agent swarms, and the
*intrinsic verifiability* doctrine — environment as verification oracle,
the same commitment as verify-HermiT made at the knowledge stratum).

Implications: (1) federation is THREE-way — verified trail outputs (HermiT
verdicts, curated references) have Gaius as their natural destination, and
Gaius's spatial projection becomes another lineup lens; (2) protocol
reconciliation needed — Gaius chose KServe OIP as peer protocol (running
note §8) while `zndx.engine.v1` is dual-registered Ægir+Atelier only:
Gaius registers v1 alongside OIP, or the federation face bridges. Both
belong in the §21 proposal with the note contract + SKOS lifting policy.

## Bookkeeping

- Running-observations §21 (federation intent + note-contract proposal) is
  deliberately DEFERRED until alignment work starts — §20 already carries
  the adoption record.
- Landing top-row cards should keep Atlas Lexicon naming as they gain
  links (they already do: Entities / Terms).
