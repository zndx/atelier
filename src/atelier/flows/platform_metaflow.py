"""Discover Signals platform Metaflow — no lab port literals.

Signals is the K8s management instance. Metadata, Airflow, YuniKorn, and
RustFS are its surfaces. Atelier resolves them at runtime from:

1. ``SIGNALS_ENGINE_GRPC`` or peer-contract ``engine_grpc_lattice.signals``
2. Engine/Status (must be ``project=signals`` with healthy ``scheduler``,
   preferring ``metaflow`` when advertised)
3. peer-contract ``endpoints.metaflow_service`` / ``rustfs_s3`` / ``airflow``
4. In-cluster: platform profile ``*_INTERNAL_URL`` (K8s service DNS)

Fail-closed when Status is not Signals, scheduler/metaflow is unhealthy,
the contract has no Metaflow URL, or ping fails. There is no Atelier-local
Tilt Metaflow SoR.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

PLATFORM_CAPABILITIES = ("metaflow", "scheduler")

GURU_NOSCHED = "#MF.00000005.NOSCHED"
GURU_NOPLATFORM = "#MF.00000006.NOPLATFORM"
GURU_NOPROFILE = "#MF.00000007.NOPROFILE"
GURU_NOCONTRACT = "#MF.00000008.NOCONTRACT"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROFILE_REL = Path("config") / "metaflow" / "platform.json"
_TILT_FRAGMENTS = ("metaflow-artifacts", "devenv-rustfs", "minioadmin")
_PLATFORM_SYSROOT = "s3://metaflow/metaflow"


class PlatformMetaflowError(RuntimeError):
    """Fail-fast when federation is present but platform Metaflow is not usable."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(
            f"{code} {detail}\n"
            "  Try: just metaflow-platform   # in the Signals checkout\n"
            "  Or:  just signals-ready\n"
            "  Discover Metaflow from Signals Status/contract; do not start an engine-local metadata service"
        )


@dataclass(frozen=True)
class PlatformMetaflowDecision:
    mode: str
    reason: str
    signals_project: str | None
    capabilities: tuple[tuple[str, bool], ...]
    metaflow_ping_ok: bool
    signals_engine: str = ""
    metaflow_service_url: str = ""
    rustfs_s3_url: str = ""
    airflow_url: str = ""
    in_cluster: bool = False


StatusProbe = Callable[[str, float], tuple[str, list[tuple[str, bool]]] | None]
PingProbe = Callable[[str, float], bool]


def in_cluster(environ: Mapping[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    return bool((env.get("KUBERNETES_SERVICE_HOST") or "").strip())


def _cap_map(caps: Iterable[tuple[str, bool]]) -> dict[str, bool]:
    return {name.lower(): healthy for name, healthy in caps if name}


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes"}


def contract_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """Peer-contract path. Env wins; else the s2s resolver (lab file if present)."""
    env = environ if environ is not None else os.environ
    for key in ("ATELIER_PEER_CONTRACT", "SIGNALS_PEER_CONTRACT"):
        raw = (env.get(key) or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_file() else None
    try:
        from atelier.engine.s2s import peer_contract_path

        return peer_contract_path()
    except Exception:
        return None


def load_contract(
    environ: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    loc = path if path is not None else contract_path(env)
    if loc is None or not loc.is_file():
        return {}
    try:
        doc = json.loads(loc.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PlatformMetaflowError(
            GURU_NOCONTRACT, f"peer-contract unreadable at {loc}: {e}"
        ) from e
    if not isinstance(doc, dict):
        raise PlatformMetaflowError(
            GURU_NOCONTRACT, f"peer-contract at {loc} is not a JSON object"
        )
    return doc


def load_profile_template() -> dict[str, str]:
    """K8s identity + datastore keys. Host URLs are not stored here."""
    path = _REPO_ROOT / _PROFILE_REL
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if not str(k).startswith("_")}


def _endpoint(doc: Mapping[str, Any], name: str, key: str = "url") -> str:
    block = (doc.get("endpoints") or {}).get(name) or {}
    if not isinstance(block, dict):
        return ""
    return str(block.get(key) or "").strip()


def _hostname(url: str) -> str:
    host = (urlparse(url).hostname or "").strip()
    return host.strip("[]")


def signals_engine_addr(
    doc: Mapping[str, Any],
    environ: Mapping[str, str],
) -> str:
    raw = (environ.get("SIGNALS_ENGINE_GRPC") or "").strip()
    if raw:
        return raw
    lattice = doc.get("engine_grpc_lattice") or {}
    port = lattice.get("signals")
    if not isinstance(port, int) or port <= 0:
        raise PlatformMetaflowError(
            GURU_NOCONTRACT,
            "peer-contract engine_grpc_lattice.signals is missing; "
            "set SIGNALS_ENGINE_GRPC or point SIGNALS_PEER_CONTRACT at the Signals contract",
        )
    host = (
        (environ.get("SIGNALS_LATTICE_HOST") or "").strip()
        or (environ.get("ATELIER_LATTICE_HOST") or "").strip()
        or _hostname(_endpoint(doc, "metaflow_service"))
        or _hostname(_endpoint(doc, "signals_ui"))
        or _hostname(_endpoint(doc, "rustfs_s3"))
    )
    if ":" in host and not host.startswith("["):
        host = host.rsplit(":", 1)[0]
    if not host:
        raise PlatformMetaflowError(
            GURU_NOCONTRACT,
            "cannot derive Signals engine host from contract endpoints or SIGNALS_LATTICE_HOST",
        )
    return f"{host}:{port}"


def metaflow_service_url(
    doc: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    cluster: bool,
    profile: Mapping[str, str],
) -> str:
    if cluster:
        internal = (
            (environ.get("METAFLOW_SERVICE_INTERNAL_URL") or "").strip()
            or (profile.get("METAFLOW_SERVICE_INTERNAL_URL") or "").strip()
        )
        if internal:
            return internal.rstrip("/")
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            "in-cluster Metaflow requires METAFLOW_SERVICE_INTERNAL_URL "
            "(Signals platform profile / pod env)",
        )
    env_url = (environ.get("METAFLOW_SERVICE_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    url = _endpoint(doc, "metaflow_service", "url")
    if not url:
        raise PlatformMetaflowError(
            GURU_NOCONTRACT,
            "peer-contract endpoints.metaflow_service.url is missing; "
            "set METAFLOW_SERVICE_URL or SIGNALS_PEER_CONTRACT",
        )
    return url.rstrip("/")


def rustfs_s3_url(
    doc: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    cluster: bool,
    profile: Mapping[str, str],
) -> str:
    if cluster:
        internal = (
            (environ.get("METAFLOW_S3_ENDPOINT_INTERNAL_URL") or "").strip()
            or (profile.get("METAFLOW_S3_ENDPOINT_INTERNAL_URL") or "").strip()
        )
        if internal:
            return internal.rstrip("/")
    env_url = (environ.get("METAFLOW_S3_ENDPOINT_URL") or "").strip()
    if env_url and not cluster:
        return env_url.rstrip("/")
    url = _endpoint(doc, "rustfs_s3", "url")
    if url:
        return url.rstrip("/")
    if cluster:
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            "in-cluster RustFS requires METAFLOW_S3_ENDPOINT_INTERNAL_URL",
        )
    raise PlatformMetaflowError(
        GURU_NOCONTRACT,
        "peer-contract endpoints.rustfs_s3.url is missing",
    )


def airflow_url(doc: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    env_url = (environ.get("AIRFLOW_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    return _endpoint(doc, "airflow", "url").rstrip("/")


def _probe_signals_status(addr: str, timeout: float) -> tuple[str, list[tuple[str, bool]]] | None:
    try:
        import grpc

        from zndx.engine.v1 import engine_pb2
        from zndx.engine.v1 import engine_pb2_grpc
    except ImportError:
        return None

    channel = grpc.insecure_channel(addr)
    try:
        stub = engine_pb2_grpc.EngineStub(channel)
        resp = stub.Status(engine_pb2.StatusRequest(), timeout=timeout)
        caps = [(ep.capability, bool(ep.healthy)) for ep in resp.endpoints]
        return resp.project, caps
    except Exception:
        return None
    finally:
        channel.close()


def _ping_metaflow(base_url: str, timeout: float) -> bool:
    ping = base_url.rstrip("/") + "/ping"
    try:
        with urlopen(ping, timeout=timeout) as resp:
            ok = 200 <= getattr(resp, "status", 200) < 300
            body = resp.read().decode("utf-8", "replace").strip().lower()
        return ok and ("pong" in body or body == "ok")
    except (URLError, OSError, TimeoutError, ValueError):
        return False


def resolve_metaflow_mode(
    *,
    environ: Mapping[str, str] | None = None,
    status_probe: StatusProbe | None = None,
    ping_probe: PingProbe | None = None,
    contract: Mapping[str, Any] | None = None,
    timeout: float = 2.0,
) -> PlatformMetaflowDecision:
    """Resolve platform Metaflow from Signals Status + discovered contract URLs.

    ``ATELIER_METAFLOW_MODE=platform|local`` is tests / break-glass.
    Forced ``platform`` still requires a ping unless ``ATELIER_METAFLOW_SKIP_PING=1``.
    """
    env = dict(environ if environ is not None else os.environ)
    override = (env.get("ATELIER_METAFLOW_MODE") or "").strip().lower()
    skip_ping = _truthy(env.get("ATELIER_METAFLOW_SKIP_PING") or "")
    cluster = in_cluster(env)
    profile = load_profile_template()
    doc = dict(contract) if contract is not None else load_contract(env)
    probe = status_probe or _probe_signals_status
    ping = ping_probe or _ping_metaflow

    if override == "local":
        return PlatformMetaflowDecision(
            mode="local",
            reason="ATELIER_METAFLOW_MODE=local",
            signals_project=None,
            capabilities=(),
            metaflow_ping_ok=False,
            in_cluster=cluster,
        )

    addr = signals_engine_addr(doc, env)
    service_url = metaflow_service_url(doc, env, cluster=cluster, profile=profile)
    rustfs = rustfs_s3_url(doc, env, cluster=cluster, profile=profile)
    af = airflow_url(doc, env)

    if override == "platform":
        ping_ok = True if skip_ping else ping(service_url, timeout)
        if not ping_ok:
            raise PlatformMetaflowError(
                GURU_NOPLATFORM,
                f"ATELIER_METAFLOW_MODE=platform but Metaflow ping failed at {service_url}/ping",
            )
        return PlatformMetaflowDecision(
            mode="platform",
            reason="ATELIER_METAFLOW_MODE=platform",
            signals_project=None,
            capabilities=(),
            metaflow_ping_ok=True,
            signals_engine=addr,
            metaflow_service_url=service_url,
            rustfs_s3_url=rustfs,
            airflow_url=af,
            in_cluster=cluster,
        )

    probed = probe(addr, timeout)
    if probed is None:
        raise PlatformMetaflowError(
            GURU_NOSCHED,
            f"Signals Engine/Status unreachable at {addr}",
        )
    project, caps = probed
    if (project or "").lower() != "signals":
        raise PlatformMetaflowError(
            GURU_NOSCHED,
            f"Engine at {addr} project={project!r} is not signals",
        )
    cmap = _cap_map(caps)
    platform_caps = {k: cmap[k] for k in PLATFORM_CAPABILITIES if k in cmap}
    if not platform_caps:
        raise PlatformMetaflowError(
            GURU_NOSCHED,
            f"Signals Status at {addr} advertises no metaflow/scheduler capability",
        )
    if "metaflow" in platform_caps and not platform_caps["metaflow"]:
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            f"Signals Status capability=metaflow is unhealthy at {addr}",
        )
    if "scheduler" in platform_caps and not platform_caps["scheduler"]:
        raise PlatformMetaflowError(
            GURU_NOSCHED,
            f"Signals Status capability=scheduler is unhealthy at {addr} "
            "(YuniKorn is the Metaflow compute admitter)",
        )

    ping_ok = True if skip_ping else ping(service_url, timeout)
    if not ping_ok:
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            f"Signals scheduler is healthy but Metaflow metadata ping failed at {service_url}/ping",
        )

    chosen = "metaflow" if platform_caps.get("metaflow") else "scheduler"
    return PlatformMetaflowDecision(
        mode="platform",
        reason=f"Signals Engine/Status {chosen} healthy; Metaflow {service_url} ready",
        signals_project=project,
        capabilities=tuple(caps),
        metaflow_ping_ok=True,
        signals_engine=addr,
        metaflow_service_url=service_url,
        rustfs_s3_url=rustfs,
        airflow_url=af,
        in_cluster=cluster,
    )


def require_signals_metaflow(environ: Mapping[str, str] | None = None) -> None:
    """Fail if this process is about to use a local / Tilt datastore."""
    env = environ if environ is not None else os.environ
    ds = (env.get("METAFLOW_DEFAULT_DATASTORE") or "").strip().lower()
    if ds in {"local", ""}:
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            "Classification uses the Signals Metaflow datastore, not a local one. "
            f"METAFLOW_DEFAULT_DATASTORE={ds!r}.",
        )
    sysroot = env.get("METAFLOW_DATASTORE_SYSROOT_S3") or ""
    if not (
        sysroot.startswith("s3://metaflow/")
        or sysroot.startswith("s3://signals-dataproducts/")
    ):
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            f"METAFLOW_DATASTORE_SYSROOT_S3={sysroot!r} is not the Signals Metaflow/RustFS sysroot.",
        )
    endpoint = env.get("METAFLOW_S3_ENDPOINT_URL") or ""
    if "metaflow-artifacts" in sysroot or "devenv-rustfs" in endpoint:
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            "Tilt/devenv object store is not Signals RustFS.",
        )


def _libpq_libdir() -> Path | None:
    """libpq for Metaflow step subprocesses (psycopg) when not in devenv shell.

    Follow devenv's postgresql package to the matching ``-lib`` output.
    Do not call ``nix-store`` (glibc-mixed PATH breaks it on this host).
    """
    wr = _REPO_ROOT / ".devenv" / "profile" / "lib" / "libpqwalreceiver.so"
    if not wr.exists():
        return None
    pkg = wr.resolve().parent.parent  # .../<hash>-postgresql-16.13
    ver = re.search(r"postgresql-[\d.]+", pkg.name)
    globs = [pkg / "lib"]
    if ver:
        globs.extend(pkg.parent.glob(f"*-{ver.group(0)}-lib/lib"))
    for cand in globs:
        if (cand / "libpq.so.5").exists() or (cand / "libpq.so").exists():
            return cand
    return None


def _ensure_libpq(env: dict[str, str]) -> None:
    libdir = _libpq_libdir()
    if libdir is None:
        return
    cur = (env.get("LD_LIBRARY_PATH") or "").split(":")
    s = str(libdir)
    if s in cur:
        return
    env["LD_LIBRARY_PATH"] = ":".join([s, *[p for p in cur if p]])


def _drop_tilt_bindings(env: dict[str, str]) -> None:
    """Remove Tilt/MinIO leftovers so ~/.metaflowconfig cannot win."""
    for key, value in list(env.items()):
        if any(frag in str(value) for frag in _TILT_FRAGMENTS):
            env.pop(key, None)


def _rustfs_keys(env: Mapping[str, str]) -> tuple[str, str]:
    """Signals RustFS keys. Never keep ambient minioadmin from ~/.metaflowconfig."""
    ak = (
        (env.get("ATELIER_RUSTFS_ACCESS_KEY") or env.get("RUSTFS_ACCESS_KEY") or "")
        .strip()
    )
    sk = (
        (env.get("ATELIER_RUSTFS_SECRET_KEY") or env.get("RUSTFS_SECRET_KEY") or "")
        .strip()
    )
    return ak or "rustfsadmin", sk or "rustfsadmin"


def _write_isolated_home(env: dict[str, str], resolved: PlatformMetaflowDecision) -> Path:
    """METAFLOW_HOME that does not load ~/.metaflowconfig (Tilt artifacts bucket)."""
    raw = (env.get("ATELIER_METAFLOW_HOME") or "").strip()
    home = Path(raw) if raw else _REPO_ROOT / "build" / "metaflow-home"
    home.mkdir(parents=True, exist_ok=True)
    cfg = {
        "METAFLOW_DEFAULT_METADATA": "service",
        "METAFLOW_SERVICE_URL": resolved.metaflow_service_url,
        "METAFLOW_DEFAULT_DATASTORE": "s3",
        "METAFLOW_DATASTORE_SYSROOT_S3": _PLATFORM_SYSROOT,
        "METAFLOW_S3_ENDPOINT_URL": resolved.rustfs_s3_url,
    }
    (home / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return home


def metaflow_child_env(
    base: Mapping[str, str] | None = None,
    *,
    mode: str | None = None,
    decision: PlatformMetaflowDecision | None = None,
) -> dict[str, str]:
    """Subprocess env with discovered platform URLs applied.

    Isolates METAFLOW_HOME so a host ``~/.metaflowconfig`` pointing at
    ``s3://metaflow-artifacts`` (Tilt/MinIO) cannot be the datastore.
    """
    env = dict(base or os.environ)
    if mode == "local" or (env.get("ATELIER_METAFLOW_MODE") or "").strip().lower() == "local":
        env["ATELIER_METAFLOW_MODE"] = "local"
        env["METAFLOW_DEFAULT_DATASTORE"] = "local"
        return env

    resolved = decision
    if resolved is None:
        resolved = resolve_metaflow_mode(environ=env)
    if resolved.mode != "platform":
        raise PlatformMetaflowError(
            GURU_NOPLATFORM,
            f"expected platform Metaflow, got mode={resolved.mode!r} ({resolved.reason})",
        )

    profile = load_profile_template()
    if not profile.get("METAFLOW_DATASTORE_SYSROOT_S3"):
        raise PlatformMetaflowError(
            GURU_NOPROFILE,
            f"platform profile missing at {_REPO_ROOT / _PROFILE_REL}",
        )
    _drop_tilt_bindings(env)
    for key, value in profile.items():
        if key.endswith("_INTERNAL_URL") and not in_cluster(env):
            continue
        env[key] = str(value)
    env["ATELIER_METAFLOW_MODE"] = "platform"
    env["METAFLOW_DEFAULT_METADATA"] = "service"
    env["METAFLOW_DEFAULT_DATASTORE"] = "s3"
    env["METAFLOW_DATASTORE_SYSROOT_S3"] = _PLATFORM_SYSROOT
    env["METAFLOW_SERVICE_URL"] = resolved.metaflow_service_url
    if resolved.rustfs_s3_url:
        env["METAFLOW_S3_ENDPOINT_URL"] = resolved.rustfs_s3_url
        env["AWS_ENDPOINT_URL_S3"] = resolved.rustfs_s3_url
    ak, sk = _rustfs_keys(env)
    env["AWS_ACCESS_KEY_ID"] = ak
    env["AWS_SECRET_ACCESS_KEY"] = sk
    env.pop("METAFLOW_CARD_S3ROOT", None)
    _ensure_libpq(env)
    home = _write_isolated_home(env, resolved)
    env["METAFLOW_HOME"] = str(home)
    if resolved.signals_engine:
        env.setdefault("SIGNALS_ENGINE_GRPC", resolved.signals_engine)
    require_signals_metaflow(env)
    return env


def apply_metaflow_config(mode: str | None = None) -> PlatformMetaflowDecision:
    """Apply discovered Metaflow config to this process environment."""
    if mode == "local":
        os.environ["ATELIER_METAFLOW_MODE"] = "local"
        return PlatformMetaflowDecision(
            mode="local",
            reason="apply_metaflow_config(mode=local)",
            signals_project=None,
            capabilities=(),
            metaflow_ping_ok=False,
        )
    decision = resolve_metaflow_mode(environ=os.environ)
    os.environ.update(metaflow_child_env(os.environ, decision=decision))
    return decision
