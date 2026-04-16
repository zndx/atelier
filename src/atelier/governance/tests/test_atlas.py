"""Tests for Atlas v2 client with mocked HTTP responses."""

import json

import pytest
import responses

from governance.atlas import AtlasClient, ClassificationTag, QualifiedName
from governance.client import ClientConfig

ATLAS_URL = "https://atlas-host:21000"
API_BASE = f"{ATLAS_URL}/api/atlas/v2"


@pytest.fixture
def atlas():
    cfg = ClientConfig(url=ATLAS_URL, username="admin", password="secret")
    return AtlasClient(cfg)


class TestQualifiedName:

    def test_from_table_name_with_db(self):
        qn = QualifiedName.from_table_name("mydb.mytable", cluster="prod")
        assert qn.for_table() == "mydb.mytable@prod"
        assert qn.for_column("email") == "mydb.mytable.email@prod"

    def test_from_table_name_default_db(self):
        qn = QualifiedName.from_table_name("mytable")
        assert qn.for_table() == "default.mytable@cm"

    def test_for_column_requires_column(self):
        qn = QualifiedName.from_table_name("db.tbl")
        with pytest.raises(ValueError, match="column is required"):
            qn.for_column()


class TestClassificationTag:

    def test_to_atlas_payload_minimal(self):
        tag = ClassificationTag(type_name="PII")
        assert tag.to_atlas_payload() == {"typeName": "PII"}

    def test_to_atlas_payload_with_attrs(self):
        tag = ClassificationTag(type_name="PII", confidence="high", reason="test", notation="1.1")
        payload = tag.to_atlas_payload()
        assert payload["typeName"] == "PII"
        assert payload["attributes"]["confidence"] == "high"
        assert payload["attributes"]["notation"] == "1.1"

    def test_to_typedef_payload(self):
        tag = ClassificationTag(type_name="SENS.PID", super_types=["SENS"])
        td = tag.to_typedef_payload("Personally identifiable")
        assert td["name"] == "SENS.PID"
        assert td["superTypes"] == ["SENS"]
        assert td["description"] == "Personally identifiable"
        assert len(td["attributeDefs"]) == 3


class TestAtlasClientTypeDefs:

    @responses.activate
    def test_get_classification_def_found(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/types/classificationdef/name/PII",
            json={"name": "PII", "guid": "abc-123"},
            status=200,
        )
        result = atlas.get_classification_def("PII")
        assert result["name"] == "PII"

    @responses.activate
    def test_get_classification_def_not_found(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/types/classificationdef/name/MISSING",
            status=404,
        )
        assert atlas.get_classification_def("MISSING") is None

    @responses.activate
    def test_ensure_classification_already_exists(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/types/classificationdef/name/PII",
            json={"name": "PII"},
            status=200,
        )
        assert atlas.ensure_classification("PII") is True
        assert len(responses.calls) == 1  # No POST needed

    @responses.activate
    def test_ensure_classification_creates(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/types/classificationdef/name/NEW_TAG",
            status=404,
        )
        responses.add(
            responses.POST,
            f"{API_BASE}/types/typedefs",
            json={"classificationDefs": [{"name": "NEW_TAG"}]},
            status=200,
        )
        assert atlas.ensure_classification("NEW_TAG", super_types=["PARENT"]) is True
        # Verify POST payload
        body = json.loads(responses.calls[1].request.body)
        cdef = body["classificationDefs"][0]
        assert cdef["name"] == "NEW_TAG"
        assert cdef["superTypes"] == ["PARENT"]


class TestAtlasClientEntities:

    @responses.activate
    def test_get_entity_by_qualified_name(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/uniqueAttribute/type/hive_table",
            json={"entity": {"guid": "tbl-guid-1", "typeName": "hive_table"}},
            status=200,
        )
        result = atlas.get_entity_by_qualified_name("hive_table", "db.tbl@cm")
        assert result["entity"]["guid"] == "tbl-guid-1"

    @responses.activate
    def test_get_entity_guid(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/uniqueAttribute/type/hive_column",
            json={"entity": {"guid": "col-guid-1"}},
            status=200,
        )
        assert atlas.get_entity_guid("hive_column", "db.tbl.email@cm") == "col-guid-1"

    @responses.activate
    def test_get_entity_classifications(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/guid/guid-1",
            json={"entity": {"classifications": [
                {"typeName": "PII"},
                {"typeName": "SENS"},
            ]}},
            status=200,
        )
        assert atlas.get_entity_classifications("guid-1") == ["PII", "SENS"]


class TestAtlasClientTagging:

    @responses.activate
    def test_tag_entity_skips_existing(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/guid/g1",
            json={"entity": {"classifications": [{"typeName": "PII"}]}},
            status=200,
        )
        results = atlas.tag_entity("g1", [ClassificationTag(type_name="PII")])
        assert len(results) == 1
        assert results[0].status == "skipped"

    @responses.activate
    def test_tag_entity_dry_run(self, atlas):
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/guid/g1",
            json={"entity": {"classifications": []}},
            status=200,
        )
        results = atlas.tag_entity(
            "g1",
            [ClassificationTag(type_name="NEW", confidence="high")],
            dry_run=True,
        )
        assert len(results) == 1
        assert results[0].status == "dry_run"
        assert results[0].confidence == "high"

    @responses.activate
    def test_tag_entity_applies(self, atlas):
        # get_entity_classifications
        responses.add(
            responses.GET,
            f"{API_BASE}/entity/guid/g1",
            json={"entity": {"classifications": []}},
            status=200,
        )
        # ensure_classification (check)
        responses.add(
            responses.GET,
            f"{API_BASE}/types/classificationdef/name/PII",
            json={"name": "PII"},
            status=200,
        )
        # apply
        responses.add(
            responses.POST,
            f"{API_BASE}/entity/guid/g1/classifications",
            status=204,
        )
        results = atlas.tag_entity("g1", [ClassificationTag(type_name="PII")])
        assert len(results) == 1
        assert results[0].status == "success"
