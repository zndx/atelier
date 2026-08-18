"""Server-to-server query helpers (signals-protocol Engine/ServerQuery).

Pairwise snapshot — not gossip. Do not invent remotes, peers, or peer UI URLs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
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


_IP4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def is_ip_host(host: str) -> bool:
    h = (host or "").strip().strip("[]")
    if not h:
        return False
    if ":" in h:
        return True
    return bool(_IP4.match(h))


def rebase_items_for_request(items: list[dict[str, str]], request_host: str) -> list[dict[str, str]]:
    """LAN IP Host → rewrite this-host primary_ui to that IP. Foreign peers unchanged."""
    host = (request_host or "").split(",")[0].strip()
    if ":" in host and not host.startswith("["):
        # host:port — keep IPv4 host
        name, _, maybe_port = host.rpartition(":")
        if name and maybe_port.isdigit():
            host = name
    host = host.strip("[]")
    if not is_ip_host(host):
        return items
    mine = advertise_host()
    out: list[dict[str, str]] = []
    for it in items:
        row = dict(it)
        url = (row.get("primary_ui") or "").strip()
        parsed = urlparse(url)
        if parsed.hostname and (
            parsed.hostname == mine or is_loopback_host(parsed.hostname)
        ):
            netloc = f"{host}:{parsed.port}" if parsed.port else host
            row["primary_ui"] = urlunparse((
                parsed.scheme or "http", netloc, parsed.path,
                parsed.params, parsed.query, parsed.fragment,
            ))
        out.append(row)
    return out


# Waffle probes must fail fast — a down contract port (synth/metabase)
# is skip-this-round, not a reason to hold the HTTP handler. Signals'
# tonic connect() returns on RST; grpc-python needs a short deadline
# and no retries to feel the same.
_STATUS_TIMEOUT = 1.0
_QUERY_TIMEOUT = 2.0
_GRPC_OPTIONS = (
    ("grpc.enable_retries", 0),
    ("grpc.keepalive_timeout_ms", 1000),
    ("grpc.max_reconnect_backoff_ms", 200),
    ("grpc.initial_reconnect_backoff_ms", 50),
)


def _grpc_channel(addr: str):
    import grpc

    return grpc.insecure_channel(addr, options=_GRPC_OPTIONS)


def status_peer(target: str, timeout: float = _STATUS_TIMEOUT) -> zpb.StatusResponse | None:
    from zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = _grpc_channel(addr)
    try:
        stub = zpb_grpc.EngineStub(channel)
        return stub.Status(zpb.StatusRequest(), timeout=timeout)
    except Exception as e:  # noqa: BLE001 — offline this round
        import grpc

        code = e.code() if isinstance(e, grpc.RpcError) else type(e).__name__
        logger.info("Status failed at %s: %s", addr, code)
        return None
    finally:
        channel.close()


def query_peer(
    target: str,
    *,
    kind: int = zpb.SERVER_QUERY_KIND_REMOTES,
    timeout: float = _QUERY_TIMEOUT,
) -> zpb.ServerQueryResponse | None:
    """Ask a lattice peer ServerQuery. UNIMPLEMENTED → None."""
    import grpc
    from zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = _grpc_channel(addr)
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
    except Exception as e:  # noqa: BLE001 — treat as down this round
        logger.info("ServerQuery failed at %s: %s", addr, type(e).__name__)
        return None
    finally:
        channel.close()


def _url_is_loopback(url: str) -> bool:
    return is_loopback_host(urlparse(url).hostname or "")


def _record_surface(
    by_project: dict[str, dict[str, str]],
    *,
    project: str,
    addr: str,
    ui: str,
) -> None:
    key = (project or addr).lower()
    prev = by_project.get(key)
    if prev is None or _url_is_loopback(prev.get("primary_ui") or ""):
        by_project[key] = {
            "project": project or addr,
            "title": surface_title(project or ""),
            "engine_target": addr,
            "primary_ui": ui,
        }


def collect_peer_surfaces(*, skip_project: str = PROJECT) -> list[dict[str, str]]:
    """Signals-style waffle roster.

    Directory is this engine's PEERS list (peer-contract + optional
    SIGNALS_ENGINE_TARGET) — the same payload ServerQuery PEERS would
    return locally. Status every hint in parallel; a down target is
    skipped this round (Signals `continue`s on Status fail). One-hop
    ServerQuery PEERS runs only against engines that just answered
    Status, so a joiner appears without blocking on offline synth /
    metabase. Do not invent URLs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seeds: list[tuple[str, str]] = list(configured_peers())
    seeds.extend(directory_seeds())
    seen_addr: set[str] = set()
    by_project: dict[str, dict[str, str]] = {}
    self_ui = local_primary_ui()
    if self_ui:
        host = advertise_host() or "localhost"
        by_project[PROJECT] = {
            "project": PROJECT,
            "title": surface_title(PROJECT),
            "engine_target": f"{host}:50251",
            "primary_ui": self_ui,
        }

    def take_unique(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for hint, target in pairs:
            addr = target.replace("grpc://", "").strip()
            if not addr or addr in seen_addr:
                continue
            seen_addr.add(addr)
            out.append((hint, addr))
        return out

    def status_one(item: tuple[str, str]) -> tuple[str, str, zpb.StatusResponse | None]:
        hint, addr = item
        return hint, addr, status_peer(addr, timeout=_STATUS_TIMEOUT)

    def apply_status(
        hint: str, addr: str, status: zpb.StatusResponse | None
    ) -> str | None:
        if status is None:
            return None
        project = (status.project or hint or "").strip()
        if project and project == skip_project:
            return None
        ui = primary_ui_of(status)
        if ui:
            _record_surface(by_project, project=project or addr, addr=addr, ui=ui)
        return addr

    wave = take_unique(seeds)
    live: list[str] = []
    if wave:
        workers = min(8, len(wave))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(status_one, item) for item in wave]
            for fut in as_completed(futs):
                hint, addr, status = fut.result()
                live_addr = apply_status(hint, addr, status)
                if live_addr:
                    live.append(live_addr)

    new_hints: list[tuple[str, str]] = []
    if live:
        workers = min(8, len(live))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    query_peer, addr, kind=zpb.SERVER_QUERY_KIND_PEERS, timeout=_QUERY_TIMEOUT
                )
                for addr in live
            ]
            for fut in as_completed(futs):
                peers = fut.result()
                if peers is None:
                    continue
                for peer in peers.peers:
                    tgt = (peer.target or "").strip()
                    if tgt:
                        new_hints.append((peer.project or "", tgt))

    wave2 = take_unique(new_hints)
    if wave2:
        workers = min(8, len(wave2))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(status_one, item) for item in wave2]
            for fut in as_completed(futs):
                hint, addr, status = fut.result()
                apply_status(hint, addr, status)

    return sorted(by_project.values(), key=lambda row: row["project"])


# Last good waffle roster. Ghostty WASM / engine connect storms make a
# fresh walk miss or hang; the HTTP handler must not return empty while
# a previous probe already found peers.
_ROSTER_TTL_S = 8.0
_roster_lock = threading.Lock()
_roster_at = 0.0
_roster_items: list[dict[str, str]] = []


def reset_roster_cache() -> None:
    """Tests only."""
    global _roster_at, _roster_items
    with _roster_lock:
        _roster_at = 0.0
        _roster_items = []


def collect_peer_surfaces_cached(*, ttl: float = _ROSTER_TTL_S) -> list[dict[str, str]]:
    """Return a fresh walk, a TTL hit, or the last non-empty roster.

    A failed or empty walk during engine connect must not wipe peers the
    operator already saw.
    """
    global _roster_at, _roster_items
    now = time.monotonic()
    with _roster_lock:
        if _roster_items and (now - _roster_at) < ttl:
            return [dict(row) for row in _roster_items]
    try:
        items = collect_peer_surfaces()
    except Exception:
        logger.info("collect_peer_surfaces failed; serving last roster", exc_info=True)
        with _roster_lock:
            return [dict(row) for row in _roster_items]
    if items:
        with _roster_lock:
            _roster_items = [dict(row) for row in items]
            _roster_at = time.monotonic()
        return items
    with _roster_lock:
        if _roster_items:
            return [dict(row) for row in _roster_items]
    return items
