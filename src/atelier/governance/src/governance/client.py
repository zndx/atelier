"""Base HTTP client with CDP-aware URL detection and session management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceEndpoint:
    """Resolved service endpoint after URL normalisation."""

    base_url: str
    api_prefix: str

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.api_prefix}".rstrip("/")


class CDPUrlResolver:
    """Detects CDP proxy, standard, and custom Atlas/Ranger URL formats.

    Supported patterns (Atlas example — Ranger works identically):
        CDP proxy:  https://host/cdp-proxy-api/atlas   → .../atlas/api/atlas/v2
        Standard:   https://host:21000                  → .../api/atlas/v2
        Explicit:   https://host:21000/api/atlas/v2     → kept as-is
    """

    ATLAS_API_PREFIX = "api/atlas/v2"
    RANGER_API_PREFIX = "service/public/v2/api"

    @classmethod
    def resolve_atlas(cls, url: str) -> ServiceEndpoint:
        return cls._resolve(url, cls.ATLAS_API_PREFIX, "/atlas")

    @classmethod
    def resolve_ranger(cls, url: str) -> ServiceEndpoint:
        return cls._resolve(url, cls.RANGER_API_PREFIX, "/ranger")

    @classmethod
    def _resolve(cls, raw_url: str, api_prefix: str, proxy_suffix: str) -> ServiceEndpoint:
        url = raw_url.rstrip("/")
        if api_prefix in url:
            # Already fully qualified
            base = url[: url.index(api_prefix)].rstrip("/")
            return ServiceEndpoint(base_url=base, api_prefix=api_prefix)
        if url.endswith(proxy_suffix):
            # CDP proxy format — prefix sits under the proxy path
            return ServiceEndpoint(base_url=url, api_prefix=api_prefix)
        # Bare host:port — append the standard prefix
        return ServiceEndpoint(base_url=url, api_prefix=api_prefix)


@dataclass
class ClientConfig:
    """Connection parameters shared by Atlas and Ranger clients."""

    url: str
    username: str
    password: str
    verify_ssl: bool = False
    timeout: float = 30.0
    cluster_name: str = "cm"
    extra_headers: dict[str, str] = field(default_factory=dict)


class BaseClient:
    """Thin wrapper around :class:`requests.Session` with auth, SSL, and logging."""

    def __init__(self, endpoint: ServiceEndpoint, config: ClientConfig) -> None:
        self._endpoint = endpoint
        self._config = config
        self._session = requests.Session()
        self._session.auth = HTTPBasicAuth(config.username, config.password)
        self._session.verify = config.verify_ssl
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if config.extra_headers:
            self._session.headers.update(config.extra_headers)

    @property
    def endpoint(self) -> ServiceEndpoint:
        return self._endpoint

    @property
    def base_api_url(self) -> str:
        return self._endpoint.url

    # -- HTTP verbs --------------------------------------------------------

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("DELETE", path, **kwargs)

    # -- internals ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_api_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", self._config.timeout)
        log.debug("%s %s", method, url)
        resp = self._session.request(method, url, **kwargs)
        if not resp.ok:
            log.warning(
                "%s %s → %d: %s",
                method,
                url,
                resp.status_code,
                resp.text[:300],
            )
        return resp


class GovernanceClient:
    """Convenience facade that exposes both Atlas and Ranger via one config.

    Usage::

        from governance import GovernanceClient

        gc = GovernanceClient(
            atlas_url="https://cdp-host/cdp-proxy-api/atlas",
            ranger_url="https://cdp-host:6182",
            username="admin",
            password="...",
        )
        gc.atlas.get_classifications()
        gc.ranger.get_roles()
    """

    def __init__(
        self,
        atlas_url: str | None = None,
        ranger_url: str | None = None,
        username: str = "admin",
        password: str = "",
        verify_ssl: bool = False,
        timeout: float = 30.0,
        cluster_name: str = "cm",
    ) -> None:
        from governance.atlas import AtlasClient
        from governance.ranger import RangerClient

        shared = dict(
            username=username,
            password=password,
            verify_ssl=verify_ssl,
            timeout=timeout,
            cluster_name=cluster_name,
        )

        self._atlas: AtlasClient | None = None
        self._ranger: RangerClient | None = None

        if atlas_url:
            cfg = ClientConfig(url=atlas_url, **shared)
            self._atlas = AtlasClient(cfg)
        if ranger_url:
            cfg = ClientConfig(url=ranger_url, **shared)
            self._ranger = RangerClient(cfg)

    @property
    def atlas(self) -> "AtlasClient":
        if self._atlas is None:
            raise RuntimeError("AtlasClient not configured — pass atlas_url to GovernanceClient")
        return self._atlas

    @property
    def ranger(self) -> "RangerClient":
        if self._ranger is None:
            raise RuntimeError("RangerClient not configured — pass ranger_url to GovernanceClient")
        return self._ranger
