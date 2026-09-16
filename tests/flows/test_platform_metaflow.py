"""Discover Signals Metaflow from Status + contract — no lab port defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atelier.flows.lattice import (
    GURU_NOCAP,
    assert_complete_capability,
    require_signals_metaflow,
)
from atelier.flows.platform_metaflow import (
    GURU_NOCONTRACT,
    GURU_NOPLATFORM,
    GURU_NOSCHED,
    PlatformMetaflowError,
    metaflow_child_env,
    resolve_metaflow_mode,
)


def _contract(tmp_path: Path, **extra: object) -> Path:
    doc = {
        "engine_grpc_lattice": {"signals": 1, "atelier": 2},
        "endpoints": {
            "metaflow_service": {
                "url": "http://mf.example.test:9",
                "ping": "http://mf.example.test:9/ping",
            },
            "rustfs_s3": {"url": "http://s3.example.test:8"},
            "airflow": {"url": "http://af.example.test:7"},
            "signals_ui": {"url": "http://ui.example.test:6"},
        },
    }
    doc.update(extra)
    path = tmp_path / "peer-contract.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _env(contract: Path, **more: str) -> dict[str, str]:
    env = {
        "SIGNALS_PEER_CONTRACT": str(contract),
        "SIGNALS_ENGINE_GRPC": "signals.example.test:1",
    }
    env.update(more)
    return env


def test_status_not_signals_fails(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    with pytest.raises(PlatformMetaflowError, match=GURU_NOSCHED):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: ("gaius", [("scheduler", True)]),
            ping_probe=lambda url, t: True,
        )


def test_signals_unreachable_fails(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    with pytest.raises(PlatformMetaflowError, match=GURU_NOSCHED):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: None,
            ping_probe=lambda url, t: True,
        )


def test_scheduler_unhealthy_fails(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    with pytest.raises(PlatformMetaflowError, match=GURU_NOSCHED):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: ("signals", [("scheduler", False)]),
            ping_probe=lambda url, t: True,
        )


def test_metaflow_capability_unhealthy_fails(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: (
                "signals",
                [("scheduler", True), ("metaflow", False)],
            ),
            ping_probe=lambda url, t: True,
        )


def test_ping_uses_contract_metaflow_url(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    seen: list[str] = []

    def ping(url: str, t: float) -> bool:
        seen.append(url)
        return True

    d = resolve_metaflow_mode(
        environ=env,
        status_probe=lambda addr, t: ("signals", [("scheduler", True)]),
        ping_probe=ping,
    )
    assert d.mode == "platform"
    assert d.signals_project == "signals"
    assert d.metaflow_service_url == "http://mf.example.test:9"
    assert d.rustfs_s3_url == "http://s3.example.test:8"
    assert d.airflow_url == "http://af.example.test:7"
    assert d.signals_engine == "signals.example.test:1"
    assert seen == ["http://mf.example.test:9"]


def test_prefers_metaflow_capability_when_advertised(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    d = resolve_metaflow_mode(
        environ=env,
        status_probe=lambda addr, t: (
            "signals",
            [("scheduler", True), ("metaflow", True)],
        ),
        ping_probe=lambda url, t: True,
    )
    assert d.mode == "platform"
    assert "metaflow" in d.reason


def test_missing_contract_metaflow_url_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps({"engine_grpc_lattice": {"signals": 1}}),
        encoding="utf-8",
    )
    env = _env(path)
    env.pop("METAFLOW_SERVICE_URL", None)
    with pytest.raises(PlatformMetaflowError, match=GURU_NOCONTRACT):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: ("signals", [("scheduler", True)]),
            ping_probe=lambda url, t: True,
        )


def test_in_cluster_prefers_internal_url(tmp_path: Path) -> None:
    internal = "http://metaflow-service.metaflow.svc.cluster.local:8080"
    env = _env(
        _contract(tmp_path),
        KUBERNETES_SERVICE_HOST="10.0.0.1",
        METAFLOW_SERVICE_INTERNAL_URL=internal,
        METAFLOW_S3_ENDPOINT_INTERNAL_URL="http://signals-rustfs.metaflow.svc.cluster.local:9010",
    )
    seen: list[str] = []
    d = resolve_metaflow_mode(
        environ=env,
        status_probe=lambda addr, t: ("signals", [("scheduler", True)]),
        ping_probe=lambda url, t: seen.append(url) or True,
    )
    assert d.in_cluster
    assert d.metaflow_service_url == internal
    assert seen == [internal]


def test_federated_ping_down_fails(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path))
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: ("signals", [("scheduler", True)]),
            ping_probe=lambda url, t: False,
        )


def test_override_platform_still_requires_ping(tmp_path: Path) -> None:
    env = _env(_contract(tmp_path), ATELIER_METAFLOW_MODE="platform")
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        resolve_metaflow_mode(
            environ=env,
            status_probe=lambda addr, t: None,
            ping_probe=lambda url, t: False,
        )


def test_child_env_forces_discovered_platform_datastore(tmp_path: Path) -> None:
    env = _env(
        _contract(tmp_path),
        ATELIER_METAFLOW_MODE="platform",
        ATELIER_METAFLOW_SKIP_PING="1",
        METAFLOW_DEFAULT_DATASTORE="local",
    )
    child = metaflow_child_env(env)
    assert child["ATELIER_METAFLOW_MODE"] == "platform"
    assert child["METAFLOW_DEFAULT_DATASTORE"] == "s3"
    assert child["METAFLOW_SERVICE_URL"] == "http://mf.example.test:9"
    assert child["METAFLOW_S3_ENDPOINT_URL"] == "http://s3.example.test:8"
    assert child["METAFLOW_DATASTORE_SYSROOT_S3"].startswith("s3://metaflow/")


def test_require_signals_metaflow_refuses_local() -> None:
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        require_signals_metaflow({"METAFLOW_DEFAULT_DATASTORE": "local"})
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        require_signals_metaflow({})


def test_require_signals_metaflow_refuses_tilt_sysroot() -> None:
    with pytest.raises(PlatformMetaflowError, match=GURU_NOPLATFORM):
        require_signals_metaflow(
            {
                "METAFLOW_DEFAULT_DATASTORE": "s3",
                "METAFLOW_DATASTORE_SYSROOT_S3": "s3://metaflow-artifacts/x",
                "METAFLOW_S3_ENDPOINT_URL": "http://devenv-rustfs:9000",
            }
        )


def test_require_signals_metaflow_accepts_platform_sysroot() -> None:
    require_signals_metaflow(
        {
            "METAFLOW_DEFAULT_DATASTORE": "s3",
            "METAFLOW_DATASTORE_SYSROOT_S3": "s3://metaflow/metaflow",
            "METAFLOW_S3_ENDPOINT_URL": "http://s3.example.test:8",
        }
    )


def test_complete_capability_is_thinking_or_instruct() -> None:
    assert assert_complete_capability("thinking") == "thinking"
    assert assert_complete_capability("instruct") == "instruct"
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        assert_complete_capability("referee")
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        assert_complete_capability("anthropic")


def test_engine_addr_comes_from_contract_not_a_module_default(tmp_path: Path) -> None:
    path = _contract(tmp_path)
    env = {
        "SIGNALS_PEER_CONTRACT": str(path),
        "SIGNALS_LATTICE_HOST": "signals.example.test",
    }
    seen_addr: list[str] = []

    def status(addr: str, t: float):
        seen_addr.append(addr)
        return ("signals", [("scheduler", True)])

    d = resolve_metaflow_mode(
        environ=env,
        status_probe=status,
        ping_probe=lambda url, t: True,
    )
    assert seen_addr == ["signals.example.test:1"]
    assert d.signals_engine == "signals.example.test:1"
