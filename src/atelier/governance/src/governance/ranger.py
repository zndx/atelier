"""Apache Ranger v2 REST client.

Covers role CRUD, tag-based and resource-based policy management,
user management, and service definition queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from governance.client import BaseClient, CDPUrlResolver, ClientConfig

log = logging.getLogger(__name__)


# -- Data structures -------------------------------------------------------


@dataclass
class RangerRole:
    """Minimal role representation for create/update operations."""

    name: str
    description: str = ""
    users: list[dict[str, Any]] = field(default_factory=list)
    groups: list[dict[str, Any]] = field(default_factory=list)
    roles: list[dict[str, Any]] = field(default_factory=list)
    id: int | None = None

    def add_user(self, username: str, is_admin: bool = False) -> None:
        if not any(u["name"] == username for u in self.users):
            self.users.append({"name": username, "isAdmin": is_admin})

    def remove_user(self, username: str) -> None:
        self.users = [u for u in self.users if u["name"] != username]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "users": self.users,
            "groups": self.groups,
            "roles": self.roles,
        }
        if self.id is not None:
            payload["id"] = self.id
        return payload

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> RangerRole:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            users=data.get("users", []),
            groups=data.get("groups", []),
            roles=data.get("roles", []),
            id=data.get("id"),
        )


@dataclass
class PolicyItem:
    """A single access or masking entry within a policy."""

    users: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    accesses: list[dict[str, Any]] = field(default_factory=list)
    delegate_admin: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "users": self.users,
            "groups": self.groups,
            "roles": self.roles,
            "accesses": self.accesses,
            "delegateAdmin": self.delegate_admin,
        }


@dataclass
class PolicyResource:
    """Resource definition for a Ranger policy (database, table, column)."""

    values: list[str]
    is_excludes: bool = False
    is_recursive: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "isExcludes": self.is_excludes,
            "isRecursive": self.is_recursive,
        }


# -- Client ----------------------------------------------------------------


class RangerClient:
    """Apache Ranger v2 REST client.

    Usage::

        from governance import RangerClient
        from governance.client import ClientConfig

        client = RangerClient(ClientConfig(
            url="https://host:6182",
            username="admin",
            password="secret",
        ))

        # Roles
        roles = client.get_roles("hive")
        client.create_role("hive", RangerRole(name="DATA_READERS"))

        # Policies
        policies = client.find_policies(service_name="cm_hive")
    """

    def __init__(self, config: ClientConfig) -> None:
        endpoint = CDPUrlResolver.resolve_ranger(config.url)
        self._http = BaseClient(endpoint, config)
        self._config = config

    @property
    def base_api_url(self) -> str:
        return self._http.base_api_url

    # -- Service definitions -----------------------------------------------

    def find_service_defs(self) -> list[dict[str, Any]]:
        """GET /servicedef — list all service definitions."""
        resp = self._http.get("servicedef")
        resp.raise_for_status()
        return resp.json().get("serviceDefs", [])

    def get_service_def_by_name(self, name: str) -> dict[str, Any] | None:
        """GET /servicedef/name/{name}."""
        resp = self._http.get(f"servicedef/name/{name}")
        if resp.ok:
            return resp.json()
        return None

    # -- Roles -------------------------------------------------------------

    def get_roles(self, service_name: str = "hive") -> list[dict[str, Any]]:
        """GET /roles — list roles for a service."""
        resp = self._http.get("roles", params={"serviceName": service_name})
        if resp.ok:
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("vXRangerRoles", [])
        return []

    def get_role(
        self,
        role_name: str,
        exec_user: str = "admin",
        service_name: str = "hive",
    ) -> RangerRole | None:
        """GET /roles/name/{roleName} — fetch a single role."""
        resp = self._http.get(
            f"roles/name/{role_name}",
            params={"execUser": exec_user, "serviceName": service_name},
        )
        if resp.ok:
            return RangerRole.from_response(resp.json())
        return None

    def create_role(
        self,
        service_name: str,
        role: RangerRole,
    ) -> dict[str, Any] | None:
        """POST /roles — create a new role."""
        resp = self._http.post(
            "roles",
            params={"serviceName": service_name},
            json=role.to_payload(),
        )
        if resp.ok:
            log.info("Created role: %s", role.name)
            return resp.json()
        log.error("Failed to create role %s: %s", role.name, resp.text[:300])
        return None

    def update_role(self, role_id: int, role: RangerRole) -> dict[str, Any] | None:
        """PUT /roles/{id} — update an existing role."""
        resp = self._http.put(f"roles/{role_id}", json=role.to_payload())
        if resp.ok:
            log.info("Updated role: %s (id=%d)", role.name, role_id)
            return resp.json()
        log.error("Failed to update role %s: %s", role.name, resp.text[:300])
        return None

    def delete_role(
        self,
        role_name: str,
        exec_user: str = "admin",
        service_name: str = "hive",
    ) -> bool:
        """DELETE /roles/name/{roleName}."""
        resp = self._http.delete(
            f"roles/name/{role_name}",
            params={"execUser": exec_user, "serviceName": service_name},
        )
        return resp.ok

    # -- Policies ----------------------------------------------------------

    def find_policies(self, service_name: str = "cm_hive") -> list[dict[str, Any]]:
        """GET /policy — list policies for a service."""
        resp = self._http.get("policy", params={"serviceName": service_name})
        if resp.ok:
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("policies", [])
        return []

    def get_policy(self, policy_id: int) -> dict[str, Any] | None:
        """GET /policy/{id}."""
        resp = self._http.get(f"policy/{policy_id}")
        if resp.ok:
            return resp.json()
        return None

    def create_policy(self, policy: dict[str, Any]) -> dict[str, Any] | None:
        """POST /policy — create a new policy."""
        resp = self._http.post("policy", json=policy)
        if resp.ok:
            log.info("Created policy: %s", policy.get("name", "unnamed"))
            return resp.json()
        log.error("Failed to create policy: %s", resp.text[:300])
        return None

    def update_policy(self, policy_id: int, policy: dict[str, Any]) -> dict[str, Any] | None:
        """PUT /policy/{id}."""
        resp = self._http.put(f"policy/{policy_id}", json=policy)
        if resp.ok:
            return resp.json()
        log.error("Failed to update policy %d: %s", policy_id, resp.text[:300])
        return None

    def delete_policy(self, policy_id: int) -> bool:
        """DELETE /policy/{id}."""
        resp = self._http.delete(f"policy/{policy_id}")
        return resp.ok

    # -- Policy builders ---------------------------------------------------

    def build_tag_access_policy(
        self,
        name: str,
        service_name: str,
        tag: str,
        items: list[PolicyItem],
    ) -> dict[str, Any]:
        """Build a tag-based access policy payload."""
        return {
            "policyType": 0,  # Access
            "name": name,
            "service": service_name,
            "resources": {"tag": PolicyResource(values=[tag]).to_payload()},
            "policyItems": [i.to_payload() for i in items],
        }

    def build_tag_masking_policy(
        self,
        name: str,
        service_name: str,
        tag: str,
        mask_type: str,
        items: list[PolicyItem],
    ) -> dict[str, Any]:
        """Build a tag-based masking policy payload."""
        masking_items = []
        for item in items:
            d = item.to_payload()
            d["dataMaskInfo"] = {"dataMaskType": mask_type}
            masking_items.append(d)
        return {
            "policyType": 1,  # Masking
            "name": name,
            "service": service_name,
            "resources": {"tag": PolicyResource(values=[tag]).to_payload()},
            "dataMaskPolicyItems": masking_items,
        }

    def build_resource_access_policy(
        self,
        name: str,
        service_name: str,
        resources: dict[str, PolicyResource],
        items: list[PolicyItem],
    ) -> dict[str, Any]:
        """Build a resource-based (database/table/column) access policy."""
        return {
            "policyType": 0,
            "name": name,
            "service": service_name,
            "resources": {k: v.to_payload() for k, v in resources.items()},
            "policyItems": [i.to_payload() for i in items],
        }

    def build_resource_masking_policy(
        self,
        name: str,
        service_name: str,
        resources: dict[str, PolicyResource],
        mask_type: str,
        items: list[PolicyItem],
    ) -> dict[str, Any]:
        """Build a resource-based masking policy."""
        masking_items = []
        for item in items:
            d = item.to_payload()
            d["dataMaskInfo"] = {"dataMaskType": mask_type}
            masking_items.append(d)
        return {
            "policyType": 1,
            "name": name,
            "service": service_name,
            "resources": {k: v.to_payload() for k, v in resources.items()},
            "dataMaskPolicyItems": masking_items,
        }

    # -- Users -------------------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str = "",
        first_name: str = "",
        email: str = "",
    ) -> dict[str, Any] | None:
        """POST /xusers/secure/users — create a user via the secure endpoint."""
        payload = {
            "name": username,
            "password": password or username,
            "firstName": first_name or username,
            "emailAddress": email,
            "userRoleList": ["ROLE_USER"],
            "groupIdList": [],
            "status": 1,
            "isVisible": 1,
            "userSource": 0,
        }
        resp = self._http.post(
            "../xusers/secure/users",  # relative to service/public/v2/api
            json=payload,
        )
        if resp.ok:
            log.info("Created user: %s", username)
            return resp.json()
        log.error("Failed to create user %s: %s", username, resp.text[:300])
        return None

    def create_users(self, usernames: list[str]) -> list[dict[str, Any]]:
        """Create multiple users, returning results for each."""
        results = []
        for name in usernames:
            result = self.create_user(name)
            if result:
                results.append(result)
        return results
