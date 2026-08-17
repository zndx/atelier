"""Server-to-server query helpers (signals-protocol Engine/ServerQuery).

Pairwise snapshot — not gossip. Do not invent remotes, peers, or peer UI URLs.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from zndx.engine.v1 import engine_pb2 as zpb

logger = logging.getLogger(__name__)

PROJECT = "atelier"
LATTICE_SKIP = frozenset(
    {"service", "proto", "server_reflection", "first_party_clients"}
)
_LOOPBACK = frozenset(
    {"", "localhost", "127.0.0.1", "0.0.0.0", "::1", "::", "[::1]", "[::]"}
)
_LAB_CONTRACT = Path.home() / "local/src/wxs/signals/config/platform/peer-contract.json"
# Vite default — not :3000 (Gaius Tilt / Metaflow kubectl forward).
DEFAULT_VITE_PORT = "3300"


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().strip("[]")
    if not h:
        return True
    if h.lower() in _LOOPBACK or h.startswith("127."):
        return True
    return False


def advertise_host() -> str:
    """LAN hostname or IP. Never loopback — empty is honest."""
    for key in (
        "ATELIER_ADVERTISE_HOST",
        "ATELIER_LATTICE_HOST",
        "SIGNALS_ADVERTISE_HOST",
        "SIGNALS_LATTICE_HOST",
        "SIGNALS_KRB_HOST",
        "GAIUS_ADVERTISE_HOST",
    ):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        host = raw.split("/")[-1].split(":")[0].strip("[]")
        if host and not is_loopback_host(host):
            return host
    try:
        fqdn = (socket.getfqdn() or "").strip()
        if fqdn and not is_loopback_host(fqdn) and "." in fqdn:
            return fqdn
        hn = (socket.gethostname() or "").strip()
        if hn and not is_loopback_host(hn) and "." in hn:
            return hn
        if hn and not is_loopback_host(hn):
            zt = f"{hn}.dev.vista.zndx.org"
            try:
                socket.getaddrinfo(zt, None)
                return zt
            except OSError:
                return hn
        for info in socket.getaddrinfo(hn or "localhost", None, socket.AF_INET):
            ip = info[4][0]
            if ip and not is_loopback_host(ip):
                return ip
    except OSError:
        pass
    return ""


def rewrite_public_url(url: str) -> str:
    """Replace a loopback URL host with advertise_host(). Non-loopback unchanged."""
    host = advertise_host()
    if not url or not host:
        return url
    parsed = urlparse(url)
    if not parsed.hostname or not is_loopback_host(parsed.hostname):
        return url
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunparse(
        (parsed.scheme or "http", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def repo_root() -> Path:
    raw = (os.environ.get("ATELIER_REPO_ROOT") or os.environ.get("DEVENV_ROOT") or "").strip()
    if raw:
        return Path(raw)
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".git").exists() and (cand / "src" / "atelier").exists():
            return cand
    return Path.cwd()


def list_named_remotes(root: Path | None = None) -> list[tuple[str, str]]:
    """Unique (name, fetch_url) from `git remote -v`. Do not invent remotes."""
    checkout = root or repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "remote", "-v"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    seen: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        if "(push)" in line and name in seen:
            continue
        if name not in seen:
            seen[name] = url
    return [(name, seen[name]) for name in seen]


def advertised_head(root: Path | None = None) -> str:
    checkout = root or repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _listener_pids(port: int) -> list[int]:
    try:
        proc = subprocess.run(
            ["ss", "-ltnpH"], check=False, capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids: list[int] = []
    needle = f":{port}"
    for line in proc.stdout.splitlines():
        if needle not in line:
            continue
        for token in line.replace(",", " ").split():
            if token.startswith("pid="):
                try:
                    pids.append(int(token.split("=", 1)[1]))
                except ValueError:
                    continue
    return pids


def live_vite_port() -> str:
    """Port of this checkout's Vite, if listening. Empty if none.

    Never treats kubectl/tilt (Metaflow on :3000) as Atelier.
    """
    root = str(repo_root().resolve())
    configured = (os.environ.get("ATELIER_VITE_PORT") or DEFAULT_VITE_PORT).strip()
    candidates: list[int] = []
    for raw in (configured, "3300", "3001", "3000"):
        if raw.isdigit():
            n = int(raw)
            if n not in candidates:
                candidates.append(n)
    for port in candidates:
        for pid in _listener_pids(port):
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
                cwd = os.readlink(f"/proc/{pid}/cwd")
            except OSError:
                continue
            if "kubectl" in cmd or "tilt" in cmd:
                continue
            if "vite" not in cmd and "node" not in cmd:
                continue
            if cwd.startswith(root):
                return str(port)
    return ""


def local_primary_ui() -> str:
    """This engine's product UI as a hostname URL. Never loopback.

    Prefer the live Vite listener (this checkout). Default pin is :3300 —
    :3000 is Gaius Tilt / Metaflow on the lab host. Override with
    ATELIER_PRIMARY_UI or ATELIER_UI_BIND. The waffle rebases this-host
    URLs onto the browser Host when the client arrived on a LAN IP.
    """
    raw = (os.environ.get("ATELIER_PRIMARY_UI") or os.environ.get("ATELIER_UI_URL") or "").strip()
    if raw:
        rewritten = rewrite_public_url(raw)
        parsed = urlparse(rewritten)
        if parsed.hostname and not is_loopback_host(parsed.hostname):
            return rewritten
    host = advertise_host()
    if not host:
        return ""
    port = live_vite_port()
    if not port:
        bind = (os.environ.get("ATELIER_UI_BIND") or "").strip()
        if bind:
            maybe = bind.rsplit(":", 1)[-1]
            if maybe.isdigit():
                port = maybe
        else:
            port = (os.environ.get("ATELIER_VITE_PORT") or DEFAULT_VITE_PORT).strip()
    return f"http://{host}:{port}"


def local_surfaces() -> list[zpb.Surface]:
    url = local_primary_ui()
    if not url:
        return []
    return [zpb.Surface(kind="primary", url=url, healthy=True)]


def surface_title(project: str) -> str:
    raw = (project or "").strip()
    known = {
        "gaius": "Gaius",
        "signals": "Signals",
        "aegir": "Ægir",
        "atelier": "Atelier",
        "metabase": "Metabase",
        "synth": "Synth",
    }
    if raw.lower() in known:
        return known[raw.lower()]
    if not raw:
        return "Peer"
    return raw.replace("-", " ").replace("_", " ").title()


def peer_contract_path() -> Path | None:
    for key in ("ATELIER_PEER_CONTRACT", "SIGNALS_PEER_CONTRACT"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_file() else None
    if _LAB_CONTRACT.is_file():
        return _LAB_CONTRACT
    return None


def configured_peers(contract: Path | None = None) -> list[tuple[str, str]]:
    """Lattice Engine targets from peer-contract. Skip self. Empty is honest.

    Every peer is advertised on the same host as Signals (advertise_host).
    """
    path = contract if contract is not None else peer_contract_path()
    if path is None or not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lattice = doc.get("engine_grpc_lattice") or {}
    host = (
        os.environ.get("ATELIER_LATTICE_HOST")
        or os.environ.get("SIGNALS_LATTICE_HOST")
        or advertise_host()
    ).strip()
    if ":" in host and not host.startswith("["):
        host = host.rsplit(":", 1)[0]
    if not host or is_loopback_host(host):
        return []
    out: list[tuple[str, str]] = []
    for name, port in lattice.items():
        if name in LATTICE_SKIP or name == PROJECT:
            continue
        if not isinstance(port, int):
            continue
        out.append((str(name), f"{host}:{port}"))
    return out


def directory_seeds() -> list[tuple[str, str]]:
    """Engine targets to probe. Not a UI roster."""
    hub = (os.environ.get("SIGNALS_ENGINE_TARGET") or "").strip()
    if not hub:
        return []
    return [("", hub.replace("grpc://", ""))]


def local_response(
    kind: int,
    *,
    root: Path | None = None,
    contract: Path | None = None,
) -> zpb.ServerQueryResponse:
    """Answer ServerQuery. Unknown kind → project only (honest empty payload)."""
    resp = zpb.ServerQueryResponse(project=PROJECT)
    if kind in (
        zpb.SERVER_QUERY_KIND_UNSPECIFIED,
        zpb.SERVER_QUERY_KIND_REMOTES,
    ):
        resp.remotes.extend(
            zpb.GitRemote(name=n, url=u) for n, u in list_named_remotes(root)
        )
        resp.head = advertised_head(root)
    if kind == zpb.SERVER_QUERY_KIND_PEERS:
        resp.peers.extend(
            zpb.PeerHint(project=pid, target=tgt)
            for pid, tgt in configured_peers(contract)
        )
    if kind == zpb.SERVER_QUERY_KIND_SURFACES:
        resp.surfaces.extend(local_surfaces())
    return resp


def primary_ui_of(status: zpb.StatusResponse) -> str:
    for surf in status.surfaces:
        if (surf.kind or "primary") == "primary" and (surf.url or "").strip():
            return surf.url.strip()
    return ""


def status_peer(target: str, timeout: float = 4.0) -> zpb.StatusResponse | None:
    import grpc
    from zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = grpc.insecure_channel(addr)
    try:
        stub = zpb_grpc.EngineStub(channel)
        return stub.Status(zpb.StatusRequest(), timeout=timeout)
    except grpc.RpcError as e:
        logger.info("Status failed at %s: %s", addr, e.code())
        return None
    finally:
        channel.close()


def query_peer(
    target: str,
    *,
    kind: int = zpb.SERVER_QUERY_KIND_REMOTES,
    timeout: float = 10.0,
) -> zpb.ServerQueryResponse | None:
    """Ask a lattice peer ServerQuery. UNIMPLEMENTED → None."""
    import grpc
    from zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = grpc.insecure_channel(addr)
    try:
        stub = zpb_grpc.EngineStub(channel)
        return stub.ServerQuery(
            zpb.ServerQueryRequest(kind=kind, origin_project=PROJECT),
            timeout=timeout,
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            logger.info("ServerQuery UNIMPLEMENTED at %s — peer has not adopted S2S yet", addr)
            return None
        logger.warning("ServerQuery failed at %s: %s %s", addr, e.code(), e.details())
        return None
    finally:
        channel.close()


def collect_peer_surfaces(*, skip_project: str = PROJECT) -> list[dict[str, str]]:
    """S2S waffle roster: self + PEERS + Status.surfaces. Only advertised primary UIs."""
    queue: list[tuple[str, str]] = list(configured_peers())
    queue.extend(directory_seeds())
    seen: set[str] = set()
    found: list[dict[str, str]] = []
    self_ui = local_primary_ui()
    if self_ui:
        host = advertise_host() or "localhost"
        found.append({
            "project": PROJECT,
            "title": surface_title(PROJECT),
            "engine_target": f"{host}:50251",
            "primary_ui": self_ui,
        })
    while queue:
        hint_project, target = queue.pop(0)
        addr = target.replace("grpc://", "").strip()
        if not addr or addr in seen:
            continue
        seen.add(addr)
        status = status_peer(addr)
        if status is None:
            continue
        project = (status.project or hint_project or "").strip()
        if project and project == skip_project:
            continue
        ui = primary_ui_of(status)
        if ui:
            if any(row["project"] == (project or addr) for row in found):
                continue
            found.append({
                "project": project or addr,
                "title": surface_title(project or ""),
                "engine_target": addr,
                "primary_ui": ui,
            })
        peers = query_peer(addr, kind=zpb.SERVER_QUERY_KIND_PEERS)
        if peers is None:
            continue
        for peer in peers.peers:
            tgt = (peer.target or "").strip()
            if tgt:
                queue.append((peer.project or "", tgt))
    found.sort(key=lambda row: row["project"])
    return found
