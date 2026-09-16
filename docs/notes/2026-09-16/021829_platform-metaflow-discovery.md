# Platform Metaflow discovery (PR1)

Atelier resolves Signals Metaflow from Engine/Status + peer-contract.
Production modules do not pin NodePorts. Host URLs come from
`endpoints.metaflow_service` / `rustfs_s3`; in-cluster uses
`*_INTERNAL_URL` from `config/metaflow/platform.json`.

Fail-closed: Status not `project=signals`, missing/unhealthy
`scheduler`/`metaflow`, missing contract URL, ping miss, local datastore.

Tests: `tests/flows/test_platform_metaflow.py`.
