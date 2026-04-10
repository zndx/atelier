# Detect Patterns

Run regex-based pattern detectors across column sample values to identify PII types and structured formats. Pattern matches provide independent evidence for classification.

## Pattern Detectors

| Pattern | Regex Target | Example Match |
|---------|-------------|---------------|
| `email_pattern` | RFC 5322 local@domain | `user@example.com` |
| `phone_pattern` | 7-20 digit phone formats | `+1 (555) 123-4567` |
| `ssn_pattern` | US Social Security Number | `123-45-6789` |
| `ipv4_pattern` | Dotted-quad IPv4 address | `192.168.1.1` |
| `uuid_pattern` | RFC 4122 UUID v1-v5 | `550e8400-e29b-41d4-a716-446655440000` |
| `date_iso_pattern` | ISO 8601 date prefix | `2024-03-15T10:30:00` |
| `url_pattern` | HTTP/HTTPS URL | `https://example.com/path` |
| `credit_card_pattern` | 13-19 digit card number | `4111111111111111` |

## Procedure

1. For each column, sample non-null values
2. Test every value against all 8 regex patterns
3. Compute match ratio per pattern: `matches / total_values`
4. A column is flagged for a pattern if match ratio exceeds threshold (default: 0.5)
5. Return matched patterns as `pattern_signals` feature

## Input
- Column sample values (raw strings)
- Match ratio threshold (default: 0.5)

## Output
JSON per column:
```json
{"column": "...", "patterns": {"email_pattern": 0.92, "phone_pattern": 0.0}, "dominant_pattern": "email_pattern"}
```

## Notes
- Pattern detection is the fastest evidence source — pure regex, no ML models
- High-confidence pattern matches (ratio > 0.9) are strong evidence for PII categories
- Pattern signals feed into both feature extraction and `pattern_to_mass` in DST fusion
- Credit card pattern uses Luhn check as a secondary validation when ratio is ambiguous
