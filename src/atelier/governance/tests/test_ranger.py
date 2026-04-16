"""Tests for Ranger v2 client with mocked HTTP responses."""

import json

import pytest
import responses

from governance.ranger import RangerClient, RangerRole, PolicyItem, PolicyResource
from governance.client import ClientConfig

RANGER_URL = "https://ranger-host:6182"
API_BASE = f"{RANGER_URL}/service/public/v2/api"


@pytest.fixture
def ranger():
    cfg = ClientConfig(url=RANGER_URL, username="admin", password="secret")
    return RangerClient(cfg)


class TestRangerRole:

    def test_add_user(self):
        role = RangerRole(name="test")
        role.add_user("alice")
        role.add_user("alice")  # duplicate
        assert len(role.users) == 1
        assert role.users[0]["name"] == "alice"

    def test_remove_user(self):
        role = RangerRole(name="test", users=[{"name": "alice", "isAdmin": False}])
        role.remove_user("alice")
        assert len(role.users) == 0

    def test_to_payload(self):
        role = RangerRole(name="readers", description="Read-only", id=42)
        role.add_user("bob")
        p = role.to_payload()
        assert p["name"] == "readers"
        assert p["id"] == 42
        assert len(p["users"]) == 1

    def test_from_response(self):
        data = {
            "name": "writers",
            "description": "Write access",
            "users": [{"name": "carol", "isAdmin": True}],
            "groups": [],
            "roles": [],
            "id": 7,
        }
        role = RangerRole.from_response(data)
        assert role.name == "writers"
        assert role.id == 7
        assert role.users[0]["name"] == "carol"


class TestPolicyBuilders:

    def test_build_tag_access_policy(self, ranger):
        item = PolicyItem(
            roles=["DATA_READERS"],
            accesses=[{"type": "select", "isAllowed": True}],
        )
        policy = ranger.build_tag_access_policy(
            name="pii_access",
            service_name="cm_hive",
            tag="PII",
            items=[item],
        )
        assert policy["policyType"] == 0
        assert policy["resources"]["tag"]["values"] == ["PII"]
        assert policy["policyItems"][0]["roles"] == ["DATA_READERS"]

    def test_build_tag_masking_policy(self, ranger):
        item = PolicyItem(users=["analyst"])
        policy = ranger.build_tag_masking_policy(
            name="pii_mask",
            service_name="cm_hive",
            tag="PII",
            mask_type="MASK_SHOW_LAST_4",
            items=[item],
        )
        assert policy["policyType"] == 1
        assert policy["dataMaskPolicyItems"][0]["dataMaskInfo"]["dataMaskType"] == "MASK_SHOW_LAST_4"

    def test_build_resource_access_policy(self, ranger):
        resources = {
            "database": PolicyResource(values=["mydb"]),
            "table": PolicyResource(values=["*"]),
            "column": PolicyResource(values=["*"]),
        }
        item = PolicyItem(groups=["data_team"], accesses=[{"type": "select", "isAllowed": True}])
        policy = ranger.build_resource_access_policy(
            name="mydb_access",
            service_name="cm_hive",
            resources=resources,
            items=[item],
        )
        assert policy["resources"]["database"]["values"] == ["mydb"]
        assert policy["policyItems"][0]["groups"] == ["data_team"]


class TestRangerClientRoles:

    @responses.activate
    def test_get_role(self, ranger):
        responses.add(
            responses.GET,
            f"{API_BASE}/roles/name/DATA_READERS",
            json={"name": "DATA_READERS", "id": 10, "users": [], "groups": [], "roles": []},
            status=200,
        )
        role = ranger.get_role("DATA_READERS")
        assert role is not None
        assert role.name == "DATA_READERS"
        assert role.id == 10

    @responses.activate
    def test_get_role_not_found(self, ranger):
        responses.add(
            responses.GET,
            f"{API_BASE}/roles/name/MISSING",
            status=404,
        )
        assert ranger.get_role("MISSING") is None

    @responses.activate
    def test_create_role(self, ranger):
        responses.add(
            responses.POST,
            f"{API_BASE}/roles",
            json={"name": "NEW_ROLE", "id": 99},
            status=200,
        )
        role = RangerRole(name="NEW_ROLE")
        role.add_user("alice")
        result = ranger.create_role("hive", role)
        assert result is not None
        assert result["id"] == 99
        body = json.loads(responses.calls[0].request.body)
        assert body["name"] == "NEW_ROLE"


class TestRangerClientPolicies:

    @responses.activate
    def test_create_policy(self, ranger):
        responses.add(
            responses.POST,
            f"{API_BASE}/policy",
            json={"id": 1, "name": "test_policy"},
            status=200,
        )
        result = ranger.create_policy({"name": "test_policy", "service": "cm_hive"})
        assert result is not None
        assert result["name"] == "test_policy"

    @responses.activate
    def test_find_policies(self, ranger):
        responses.add(
            responses.GET,
            f"{API_BASE}/policy",
            json=[{"id": 1, "name": "policy_1"}, {"id": 2, "name": "policy_2"}],
            status=200,
        )
        policies = ranger.find_policies("cm_hive")
        assert len(policies) == 2


class TestRangerClientUsers:

    @responses.activate
    def test_create_user(self, ranger):
        responses.add(
            responses.POST,
            f"{RANGER_URL}/service/public/v2/xusers/secure/users",
            json={"name": "newuser", "id": 5},
            status=200,
        )
        result = ranger.create_user("newuser")
        assert result is not None
        body = json.loads(responses.calls[0].request.body)
        assert body["name"] == "newuser"
        assert body["userRoleList"] == ["ROLE_USER"]
