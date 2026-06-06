# Universal Vocabulary Provenance

`universal_vocabulary.json` is Atelier's BFO/IAO-grounded universal taxonomy.
Every term carries a code, label, description, and (for leaves only) an
abbreviation that is independently documented in a public ontology, RFC,
or industry standard. Class-level entries (parent nodes) carry no abbrev
to avoid any naming-convention crosstalk with customer taxonomies.

The `notation` field is intentionally empty across all entries. It exists
in the schema to carry SKOS-style numeric codes from customer-supplied
domain extensions (loaded at runtime from Hive `annotations` tables); the
universal layer has no public-source numeric encoding and therefore
contributes none.

## Audit checks

- No `abbrev` value begins with `C_` (customer-internal class-prefix
  convention).
- No `notation` value is set in this file.
- Every retained leaf `abbrev` traces to a public source listed below.

The BDD scenario `features/agent/evidence_independence.feature` enforces
the first two invariants.

## Public source per retained term

### Root

| Code | Term / Source |
|---|---|
| `ICE` | **Information Content Entity** — IAO_0000030, [Information Artifact Ontology](http://www.obofoundry.org/ontology/iao.html), OBO Foundry. Public RDF/OWL ontology built atop BFO 2.0. |

### Path components (all generic English / standard ontological partitions)

`SENSITIVE`, `NONSENSITIVE`, `PID`, `CONTACT`, `IDENTITY`, `NAME`, `DOB`,
`GOVID`, `FINANCIAL`, `PAYMENT`, `CARD`, `ACCOUNT`, `TECHNICAL`,
`METADATA`, `TIMESTAMP`, `RECID`, `STATUS` — universally understood
English ontological partitions; equivalents appear across NIST SP 800-122,
ISO/IEC 29100, GDPR Art. 4(1), and standard data-governance literature.

### Leaf abbreviations (publicly grounded)

| Abbrev | Code | Public source |
|---|---|---|
| `EMAIL` | `ICE.SENSITIVE.PID.CONTACT.EMAIL` | RFC 5321 §4.1.2 (Mailbox / Local-part@Domain); universally used English abbreviation. |
| `PHONE` | `ICE.SENSITIVE.PID.CONTACT.PHONE` | ITU-T E.164 numbering; universal English abbreviation. |
| `ADDR` | `ICE.SENSITIVE.PID.CONTACT.ADDRESS` | Universal English abbreviation for postal address; used in USPS, ISO 19160-1. |
| `FULLNAME` | `ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME` | Universal English compound; standard form field convention (HTML `autocomplete="name"`, schema.org `Person.name`). |
| `DOB` | `ICE.SENSITIVE.PID.IDENTITY.DOB` | Universal English abbreviation for Date of Birth; appears across NIST SP 800-122, HIPAA Safe Harbor 18-element list, and standard form-field conventions. |
| `SSN` | `ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN` | US Social Security Administration; HIPAA Safe Harbor §164.514(b)(2)(i)(F); universal abbreviation. |
| `PAN` | `ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN` | **Primary Account Number** — PCI DSS v4.0 Glossary; ISO/IEC 7812-1; ANSI X9.59. |
| `TXNAMT` | `ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT` | Payment-industry standard abbreviation for Transaction Amount; appears across ISO 8583 (Field 4 — Amount, Transaction), ISO 20022 message types, and major card-network APIs. |
| `BAN` | `ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN` | **Bank Account Number** — ISO 13616-1 (IBAN family root); ISO 20022 BankAccount notation; universal banking-industry abbreviation. |
| `IPADDR` | `ICE.SENSITIVE.TECHNICAL.IPADDR` | RFC 791 (IPv4) and RFC 8200 (IPv6); universal abbreviation. |
| `DEVID` | `ICE.SENSITIVE.TECHNICAL.DEVID` | NIST SP 1800-21 (mobile device identity); universal compound abbreviation. |
| `URL` | `ICE.SENSITIVE.TECHNICAL.URL` | RFC 3986 — Uniform Resource Identifier: Generic Syntax. |
| `TSTAMP` | `ICE.METADATA.TIMESTAMP` | ISO 8601; universal English abbreviation. |
| `RECID` | `ICE.METADATA.RECID` | Universal database-engineering abbreviation for Record Identifier; standard primary-key convention. |
| `STATUS` | `ICE.METADATA.STATUS` | Universal English term; standard form-field and HTTP-response convention (RFC 9110 §15). |

