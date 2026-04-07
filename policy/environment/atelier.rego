# Atelier Environment Policy
#
# Validates materialized config (build/config/atelier.json).
# Run with: conftest test build/config/atelier.json --policy policy/environment/ --all-namespaces
#
# deny  = blocking errors (service cannot start safely)
# warn  = advisory (service starts with degraded functionality)

package environment.atelier

import rego.v1

# ── Deny (blocking) ─────────────────────────────────────────────

deny contains msg if {
	not input.ATELIER_GRPC_PORT
	msg := "gRPC port not configured. Check grpc.port in config/base.conf."
}

deny contains msg if {
	not input.ATELIER_GATEWAY_PORT
	msg := "Gateway port not configured. Check gateway.port in config/base.conf."
}

deny contains msg if {
	not input.ATELIER_DB_URL
	msg := "Database URL not configured. Set ATELIER_DB_URL or check db.url in config/base.conf."
}

deny contains msg if {
	not input.QDRANT_HOST
	msg := "Qdrant host not configured. Set QDRANT_HOST or check qdrant.host in config/base.conf."
}

# ── Warn (advisory) ─────────────────────────────────────────────

warn contains msg if {
	not input.has_anthropic
	not input.has_bedrock
	msg := "No LLM credentials configured. Agent features will be unavailable."
}

warn contains msg if {
	input.is_cml
	not input.has_bedrock
	msg := "CML deployment without Bedrock credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
}

warn contains msg if {
	input.is_cml
	not input.has_anthropic
	msg := "CML deployment without Anthropic API key. Overwatch/bootstrap capability unavailable."
}
