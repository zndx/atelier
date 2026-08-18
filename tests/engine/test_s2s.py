"""Status.surfaces and ServerQuery remotes / peers / surfaces."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from atelier.engine.s2s import (
    advertise_host,
    advertised_head,
    collect_peer_surfaces,
    configured_peers,
    is_loopback_host,
    list_named_remotes,
    live_vite_port,
    local_primary_ui,
    local_response,
    local_surfaces,
    rebase_items_for_request,
    rewrite_public_url,
)
from atelier.engine.server import ZndxEngineServicer
from zndx.engine.v1 import engine_pb2 as zpb


def _native():
    from types import SimpleNamespace
    from atelier.engine.config import EngineConfig

    return SimpleNamespace(cfg=EngineConfig(), mgr=SimpleNamespace(status=lambda: []))


def test_list_named_remotes_from_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:zndx/atelier.git"],
        cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "upstream", "git@github.com:example/atelier.git"],
        cwd=tmp_path, check=True, capture_output=True)
    remotes = dict(list_named_remotes(tmp_path))
    assert remotes["origin"] == "git@github.com:zndx/atelier.git"
    assert remotes["upstream"] == "git@github.com:example/atelier.git"


def test_list_named_remotes_empty_when_not_a_repo(tmp_path: Path) -> None:
    assert list_named_remotes(tmp_path) == []
    assert advertised_head(tmp_path) == ""


def test_advertise_host_never_loopback(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    assert advertise_host() == "tinybox.dev.vista.zndx.org"
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "127.0.0.1")
    monkeypatch.setenv("SIGNALS_KRB_HOST", "tinybox.dev.vista.zndx.org")
    assert advertise_host() == "tinybox.dev.vista.zndx.org"
    assert is_loopback_host("localhost")
    assert not is_loopback_host("tinybox.dev.vista.zndx.org")


def test_short_hostname_promotes_to_zt_fqdn(monkeypatch) -> None:
    monkeypatch.delenv("ATELIER_ADVERTISE_HOST", raising=False)
    monkeypatch.delenv("ATELIER_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_ADVERTISE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_KRB_HOST", raising=False)
    monkeypatch.delenv("GAIUS_ADVERTISE_HOST", raising=False)
    import atelier.engine.s2s as s2s
    monkeypatch.setattr(s2s.socket, "getfqdn", lambda: "tinybox")
    monkeypatch.setattr(s2s.socket, "gethostname", lambda: "tinybox")
    monkeypatch.setattr(
        s2s.socket, "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("10.0.0.1", 0))],
    )
    assert advertise_host() == "tinybox.dev.vista.zndx.org"


def test_primary_ui_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.setenv("ATELIER_PRIMARY_UI", "http://127.0.0.1:3000")
    assert local_primary_ui() == "http://tinybox.dev.vista.zndx.org:3000"
    monkeypatch.delenv("ATELIER_PRIMARY_UI")
    monkeypatch.setenv("ATELIER_UI_BIND", "0.0.0.0:13000")
    monkeypatch.setattr("atelier.engine.s2s.live_vite_port", lambda: "")
    assert local_primary_ui() == "http://tinybox.dev.vista.zndx.org:13000"
    assert "127.0.0.1" not in local_primary_ui()
    assert "localhost" not in local_primary_ui()


def test_rebase_items_lan_host_rewrites_this_host_only(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    items = [
        {"project": "atelier", "primary_ui": "http://tinybox.dev.vista.zndx.org:3300"},
        {"project": "gaius", "primary_ui": "http://tinybox.dev.vista.zndx.org:9890"},
        {"project": "other", "primary_ui": "http://other.lan:80/"},
    ]
    out = rebase_items_for_request(items, "192.168.1.55:3300")
    assert out[0]["primary_ui"] == "http://192.168.1.55:3300"
    assert out[1]["primary_ui"] == "http://192.168.1.55:9890"
    assert out[2]["primary_ui"] == "http://other.lan:80/"
    same = rebase_items_for_request(items, "tinybox.dev.vista.zndx.org:3300")
    assert same[0]["primary_ui"] == items[0]["primary_ui"]


def test_rewrite_public_url_leaves_lan_alone(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    assert rewrite_public_url("http://gaius.lan:9890/") == "http://gaius.lan:9890/"


def test_status_advertises_primary_surface(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("ATELIER_PRIMARY_UI", raising=False)
    monkeypatch.delenv("ATELIER_UI_URL", raising=False)
    monkeypatch.delenv("ATELIER_UI_BIND", raising=False)
    monkeypatch.delenv("ATELIER_VITE_PORT", raising=False)
    monkeypatch.setattr("atelier.engine.s2s.live_vite_port", lambda: "")
    resp = ZndxEngineServicer(_native()).Status(zpb.StatusRequest(), None)
    assert resp.project == "atelier"
    assert [(s.kind, s.url) for s in resp.surfaces] == [
        ("primary", "http://tinybox.dev.vista.zndx.org:3300"),
    ]


def test_primary_ui_follows_live_vite_not_metaflow(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("ATELIER_PRIMARY_UI", raising=False)
    monkeypatch.delenv("ATELIER_UI_URL", raising=False)
    monkeypatch.delenv("ATELIER_UI_BIND", raising=False)
    monkeypatch.setattr("atelier.engine.s2s.live_vite_port", lambda: "3001")
    assert local_primary_ui() == "http://tinybox.dev.vista.zndx.org:3001"


def test_server_query_remotes_and_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@example.com:atelier.git"],
        cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "t"],
        cwd=tmp_path, check=True, capture_output=True)
    resp = local_response(zpb.SERVER_QUERY_KIND_REMOTES, root=tmp_path)
    assert resp.project == "atelier"
    assert [(r.name, r.url) for r in resp.remotes] == [
        ("origin", "git@example.com:atelier.git"),
    ]
    assert resp.head == advertised_head(tmp_path)
    assert len(resp.head) == 40


def test_server_query_surfaces_matches_status(monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.setattr("atelier.engine.s2s.live_vite_port", lambda: "")
    q = local_response(zpb.SERVER_QUERY_KIND_SURFACES)
    assert [(s.kind, s.url) for s in q.surfaces] == [
        (s.kind, s.url) for s in local_surfaces()
    ]
    assert all("127.0.0.1" not in s.url for s in q.surfaces)


def test_server_query_peers_same_host_as_signals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ATELIER_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("ATELIER_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_LATTICE_HOST", raising=False)
    contract = tmp_path / "peer-contract.json"
    contract.write_text(
        json.dumps({
            "engine_grpc_lattice": {
                "gaius": 50051,
                "aegir": 50151,
                "atelier": 50251,
                "signals": 50551,
                "service": "zndx.engine.v1.Engine",
            }
        }),
        encoding="utf-8",
    )
    peers = configured_peers(contract)
    assert ("gaius", "tinybox.dev.vista.zndx.org:50051") in peers
    assert ("signals", "tinybox.dev.vista.zndx.org:50551") in peers
    assert all(p[0] != "atelier" for p in peers)
    q = local_response(zpb.SERVER_QUERY_KIND_PEERS, contract=contract)
    assert {p.project for p in q.peers} == {"gaius", "aegir", "signals"}
    assert {p.target.split(":")[0] for p in q.peers} == {"tinybox.dev.vista.zndx.org"}


def test_collect_discovers_one_hop_joiner(monkeypatch) -> None:
    """Offline contract peers are skipped; a PEERS hint is Status'd next."""
    monkeypatch.setattr(
        "atelier.engine.s2s.configured_peers",
        lambda contract=None: [("gaius", "tinybox:50051")],
    )
    monkeypatch.setattr("atelier.engine.s2s.directory_seeds", lambda: [])
    monkeypatch.setattr(
        "atelier.engine.s2s.local_primary_ui",
        lambda: "http://tinybox.dev.vista.zndx.org:3300",
    )
    monkeypatch.setattr("atelier.engine.s2s.advertise_host", lambda: "tinybox.dev.vista.zndx.org")

    def fake_status(addr, timeout=4.0):
        if addr.endswith(":50051"):
            return zpb.StatusResponse(
                project="gaius",
                surfaces=[zpb.Surface(kind="primary", url="http://tinybox.dev.vista.zndx.org:9890", healthy=True)],
            )
        if addr.endswith(":50351"):
            return zpb.StatusResponse(
                project="synth",
                surfaces=[zpb.Surface(kind="primary", url="http://tinybox.dev.vista.zndx.org:3030", healthy=True)],
            )
        return None

    def fake_query(addr, *, kind=0, timeout=10.0):
        if addr.endswith(":50051") and kind == zpb.SERVER_QUERY_KIND_PEERS:
            return zpb.ServerQueryResponse(
                project="gaius",
                peers=[zpb.PeerHint(project="synth", target="tinybox:50351")],
            )
        return zpb.ServerQueryResponse(project="x")

    monkeypatch.setattr("atelier.engine.s2s.status_peer", fake_status)
    monkeypatch.setattr("atelier.engine.s2s.query_peer", fake_query)
    projects = {r["project"] for r in collect_peer_surfaces()}
    assert projects == {"atelier", "gaius", "synth"}


def test_servicer_server_query_and_yield() -> None:
    svc = ZndxEngineServicer(_native())
    resp = svc.ServerQuery(
        zpb.ServerQueryRequest(kind=zpb.SERVER_QUERY_KIND_REMOTES), None)
    assert resp.project == "atelier"
    names = {r.name for r in resp.remotes}
    assert "origin" in names
    assert all(r.url for r in resp.remotes)
    y = svc.Yield(zpb.YieldRequest(workload_id="none"), None)
    assert y.ok is True
    assert y.process_ended is False
