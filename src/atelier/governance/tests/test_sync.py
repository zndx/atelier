"""Tests for sync orchestrator with mocked Atlas/Ranger clients."""

import json

import pytest
import responses

from governance.atlas import AtlasClient
from governance.client import ClientConfig
from governance.ranger import RangerClient
from governance.sync import (
    ColumnClassification,
    PolicySpec,
    TaxonomyNode,
    derive_ranger_policies,
    sync_classifications_to_atlas,
    sync_entitlements_to_roles,
    sync_taxonomy_to_atlas,
)

ATLAS_URL = "https://atlas-host:21000"
ATLAS_API = f"{ATLAS_URL}/api/atlas/v2"
RANGER_URL = "https://ranger-host:6182"
RANGER_API = f"{RANGER_URL}/service/public/v2/api"


@pytest.fixture
def atlas():
    return AtlasClient(ClientConfig(url=ATLAS_URL, username="a", password="p"))


@pytest.fixture
def ranger():
    return RangerClient(ClientConfig(url=RANGER_URL, username="a", password="p"))


class TestTaxonomyNode:

    def test_super_types_from_parent(self):
        node = TaxonomyNode(code="SENS.PID.CI", label="Contact Info", parent_code="SENS.PID")
        assert node.super_types == ["SENS.PID"]

    def test_root_has_no_super_types(self):
        node = TaxonomyNode(code="SENS", label="Sensitive")
        assert node.super_types == []

    def test_to_classification_tag(self):
        node = TaxonomyNode(
            code="SENS.PID.CI.EMAIL",
            label="Email",
            notation="1.1.1.1",
            parent_code="SENS.PID.CI",
        )
        tag = node.to_classification_tag()
        assert tag.type_name == "SENS.PID.CI.EMAIL"
        assert tag.super_types == ["SENS.PID.CI"]
        assert tag.notation == "1.1.1.1"


class TestSyncTaxonomyDryRun:

    @responses.activate
    def test_dry_run_reports_all_as_created(self, atlas):
        # All lookups return 404 (not found)
        responses.add(responses.GET, f"{ATLAS_API}/types/classificationdef/name/SENS", status=404)
        responses.add(responses.GET, f"{ATLAS_API}/types/classificationdef/name/SENS.PID", status=404)

        nodes = [
            TaxonomyNode(code="SENS", label="Sensitive"),
            TaxonomyNode(code="SENS.PID", label="PID", parent_code="SENS"),
        ]
        report = sync_taxonomy_to_atlas(atlas, nodes, dry_run=True)
        assert report.dry_run is True
        assert report.created == ["SENS", "SENS.PID"]
        assert report.skipped == []

    @responses.activate
    def test_skips_existing(self, atlas):
        responses.add(
            responses.GET,
            f"{ATLAS_API}/types/classificationdef/name/SENS",
            json={"name": "SENS"},
            status=200,
        )
        nodes = [TaxonomyNode(code="SENS", label="Sensitive")]
        report = sync_taxonomy_to_atlas(atlas, nodes, dry_run=True)
        assert report.skipped == ["SENS"]
        assert report.created == []


class TestSyncClassificationsToAtlas:

    @responses.activate
    def test_table_not_found(self, atlas):
        responses.add(
            responses.GET,
            f"{ATLAS_API}/entity/uniqueAttribute/type/hive_table",
            status=404,
        )
        results = sync_classifications_to_atlas(
            atlas, "db.tbl", [ColumnClassification(column_name="c1", tags=["PII"])]
        )
        assert len(results) == 1
        assert results[0].status == "error"
        assert "not found" in results[0].message

    @responses.activate
    def test_column_not_found(self, atlas):
        # Table found
        responses.add(
            responses.GET,
            f"{ATLAS_API}/entity/uniqueAttribute/type/hive_table",
            json={"entity": {"guid": "tbl-1"}},
            status=200,
        )
        # Column not found
        responses.add(
            responses.GET,
            f"{ATLAS_API}/entity/uniqueAttribute/type/hive_column",
            status=404,
        )
        results = sync_classifications_to_atlas(
            atlas, "db.tbl", [ColumnClassification(column_name="missing_col", tags=["PII"])]
        )
        assert results[0].status == "error"
        assert "Column not found" in results[0].message


class TestDeriveRangerPolicies:

    @responses.activate
    def test_dry_run(self, ranger):
        specs = [PolicySpec(tag="PII", access_types=["select"], mask_type="MASK", roles=["READERS"])]
        results = derive_ranger_policies(ranger, "cm_hive", specs, dry_run=True)
        # Should get 2 policies: access + masking
        assert len(results) == 2
        assert all(r.get("_dry_run") for r in results)

    @responses.activate
    def test_creates_access_policy(self, ranger):
        responses.add(responses.POST, f"{RANGER_API}/policy", json={"id": 1, "name": "tag_access_PII"}, status=200)
        specs = [PolicySpec(tag="PII", access_types=["select"], roles=["READERS"])]
        results = derive_ranger_policies(ranger, "cm_hive", specs)
        assert len(results) == 1
        assert results[0]["name"] == "tag_access_PII"


class TestSyncEntitlementsToRoles:

    @responses.activate
    def test_dry_run(self, ranger):
        assignments = {"ROLE_A": ["alice", "bob"], "ROLE_B": ["carol"]}
        results = sync_entitlements_to_roles(ranger, "hive", assignments, dry_run=True)
        assert len(results) == 2
        assert all(r.get("_dry_run") for r in results)

    @responses.activate
    def test_creates_new_role(self, ranger):
        # Role doesn't exist
        responses.add(responses.GET, f"{RANGER_API}/roles/name/NEW_ROLE", status=404)
        # Create
        responses.add(
            responses.POST,
            f"{RANGER_API}/roles",
            json={"name": "NEW_ROLE", "id": 50},
            status=200,
        )
        results = sync_entitlements_to_roles(ranger, "hive", {"NEW_ROLE": ["alice"]})
        assert len(results) == 1
        assert results[0]["name"] == "NEW_ROLE"

    @responses.activate
    def test_updates_existing_role(self, ranger):
        # Role exists
        responses.add(
            responses.GET,
            f"{RANGER_API}/roles/name/EXISTING",
            json={"name": "EXISTING", "id": 10, "users": [{"name": "bob", "isAdmin": False}], "groups": [], "roles": []},
            status=200,
        )
        # Update
        responses.add(
            responses.PUT,
            f"{RANGER_API}/roles/10",
            json={"name": "EXISTING", "id": 10},
            status=200,
        )
        results = sync_entitlements_to_roles(ranger, "hive", {"EXISTING": ["alice"]})
        assert len(results) == 1
        # Verify the PUT payload has both users
        body = json.loads(responses.calls[1].request.body)
        user_names = {u["name"] for u in body["users"]}
        assert "alice" in user_names
        assert "bob" in user_names
