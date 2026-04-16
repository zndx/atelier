"""Apache Atlas v2 REST client.

Covers classification lifecycle, entity search, glossary operations,
and entity tagging with dry-run support and duplicate prevention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from atelier.governance.client import BaseClient, CDPUrlResolver, ClientConfig

log = logging.getLogger(__name__)


# -- Data structures -------------------------------------------------------


@dataclass
class QualifiedName:
    """Builds Atlas qualified names for Hive entities.

    Format: ``database.table@cluster`` or ``database.table.column@cluster``
    """

    database: str
    table: str
    column: str | None = None
    cluster: str = "cm"

    @classmethod
    def from_table_name(cls, table_name: str, cluster: str = "cm") -> QualifiedName:
        """Parse ``database.table`` string."""
        if "." in table_name:
            database, table = table_name.split(".", 1)
        else:
            database, table = "default", table_name
        return cls(database=database, table=table, cluster=cluster)

    def for_table(self) -> str:
        return f"{self.database}.{self.table}@{self.cluster}"

    def for_column(self, column: str | None = None) -> str:
        col = column or self.column
        if col is None:
            raise ValueError("column is required")
        return f"{self.database}.{self.table}.{col}@{self.cluster}"


@dataclass
class ClassificationTag:
    """A tag to apply to an entity, with optional metadata."""

    type_name: str
    confidence: float = 0.0
    reason: str = ""
    notation: str = ""
    super_types: list[str] | None = None

    def to_atlas_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"typeName": self.type_name}
        attrs: dict[str, str] = {}
        if self.confidence > 0:
            attrs["confidence"] = f"{self.confidence:.3f}"
        if self.reason:
            attrs["reason"] = self.reason
        if self.notation:
            attrs["notation"] = self.notation
        if attrs:
            payload["attributes"] = attrs
        return payload

    def to_typedef_payload(self, description: str = "") -> dict[str, Any]:
        """Build a classificationDefs entry for type creation."""
        attr_defs = [
            _attr_def("confidence", "Confidence level (high, medium, low)"),
            _attr_def("reason", "Reason for classification"),
            _attr_def("notation", "SKOS notation code"),
        ]
        return {
            "name": self.type_name,
            "description": description or f"Classification: {self.type_name}",
            "superTypes": self.super_types or [],
            "attributeDefs": attr_defs,
        }


@dataclass
class SyncResult:
    """Outcome of tagging a single entity."""

    entity_name: str
    entity_guid: str = ""
    tags: list[str] | None = None
    status: str = ""  # success | skipped | dry_run | error
    message: str = ""
    existing_tags: list[str] | None = None
    new_tags: list[str] | None = None
    confidence: float = 0.0


# -- Client ----------------------------------------------------------------


class AtlasClient:
    """Apache Atlas v2 REST client.

    Usage::

        from governance import AtlasClient
        from atelier.governance.client import ClientConfig

        client = AtlasClient(ClientConfig(
            url="https://host:21000",
            username="admin",
            password="secret",
        ))

        # Search
        entity = client.get_entity_by_qualified_name("hive_table", "db.tbl@cm")

        # Classify
        client.ensure_classification("PII")
        client.tag_entity(guid, [ClassificationTag("PII", confidence="high")])
    """

    def __init__(self, config: ClientConfig) -> None:
        endpoint = CDPUrlResolver.resolve_atlas(config.url)
        self._http = BaseClient(endpoint, config)
        self._config = config

    @property
    def base_api_url(self) -> str:
        return self._http.base_api_url

    # -- Type definitions --------------------------------------------------

    def get_typedefs(self) -> dict[str, Any]:
        """GET /types/typedefs — all type definitions."""
        resp = self._http.get("types/typedefs")
        resp.raise_for_status()
        return resp.json()

    def get_classification_def(self, name: str) -> dict[str, Any] | None:
        """Fetch a single classification type definition, or None."""
        resp = self._http.get(f"types/classificationdef/name/{name}")
        if resp.ok:
            return resp.json()
        return None

    def ensure_classification(
        self,
        name: str,
        description: str = "",
        super_types: list[str] | None = None,
        notation: str = "",
    ) -> bool:
        """Create a classification type if it doesn't already exist.

        Returns True if the type exists (or was created successfully).
        """
        if self.get_classification_def(name) is not None:
            return True

        tag = ClassificationTag(
            type_name=name,
            notation=notation,
            super_types=super_types,
        )
        payload = {
            "classificationDefs": [tag.to_typedef_payload(description)],
            "enumDefs": [],
            "structDefs": [],
            "entityDefs": [],
        }
        resp = self._http.post("types/typedefs", json=payload)
        if resp.ok:
            log.info("Created classification type: %s", name)
            return True
        log.error("Failed to create classification %s: %s", name, resp.text[:300])
        return False

    def ensure_classification_hierarchy(
        self,
        tags: list[ClassificationTag],
    ) -> list[str]:
        """Create multiple classification types respecting superType ordering.

        Returns list of type names that failed to create.
        """
        failures: list[str] = []
        for tag in tags:
            ok = self.ensure_classification(
                tag.type_name,
                super_types=tag.super_types,
                notation=tag.notation,
            )
            if not ok:
                failures.append(tag.type_name)
        return failures

    # -- Entity operations -------------------------------------------------

    def get_entity_by_qualified_name(
        self, type_name: str, qualified_name: str
    ) -> dict[str, Any] | None:
        """Fetch entity by its qualified name."""
        resp = self._http.get(
            f"entity/uniqueAttribute/type/{type_name}",
            params={"attr:qualifiedName": qualified_name},
        )
        if resp.ok:
            return resp.json()
        return None

    def get_entity_guid(self, type_name: str, qualified_name: str) -> str | None:
        """Convenience: return just the GUID, or None."""
        data = self.get_entity_by_qualified_name(type_name, qualified_name)
        if data:
            return data.get("entity", {}).get("guid")
        return None

    def get_entity_classifications(self, guid: str) -> list[str]:
        """Return classification type names currently on an entity."""
        resp = self._http.get(f"entity/guid/{guid}")
        if not resp.ok:
            return []
        entity = resp.json().get("entity", {})
        return [c["typeName"] for c in entity.get("classifications", [])]

    def tag_entity(
        self,
        guid: str,
        tags: list[ClassificationTag],
        *,
        dry_run: bool = False,
        skip_existing: bool = True,
    ) -> list[SyncResult]:
        """Apply classification tags to an entity.

        Args:
            guid: Entity GUID.
            tags: Tags to apply.
            dry_run: If True, return what *would* happen without writing.
            skip_existing: Skip tags already present on the entity.

        Returns:
            Per-tag SyncResult list.
        """
        existing = self.get_entity_classifications(guid) if skip_existing else []
        results: list[SyncResult] = []

        new_tags = [t for t in tags if t.type_name not in existing]
        skipped = [t for t in tags if t.type_name in existing]

        for t in skipped:
            results.append(SyncResult(
                entity_name=guid, entity_guid=guid,
                tags=[t.type_name], status="skipped",
                message="Already present", existing_tags=existing,
            ))

        if not new_tags:
            return results

        if dry_run:
            for t in new_tags:
                results.append(SyncResult(
                    entity_name=guid, entity_guid=guid,
                    tags=[t.type_name], status="dry_run",
                    message="Would apply (dry run)",
                    existing_tags=existing, new_tags=[t.type_name],
                    confidence=t.confidence,
                ))
            return results

        # Ensure type definitions exist
        for t in new_tags:
            self.ensure_classification(
                t.type_name,
                super_types=t.super_types,
                notation=t.notation,
            )

        # Apply all new tags in one request
        payload = [t.to_atlas_payload() for t in new_tags]
        resp = self._http.post(f"entity/guid/{guid}/classifications", json=payload)

        if resp.status_code in (200, 204):
            for t in new_tags:
                results.append(SyncResult(
                    entity_name=guid, entity_guid=guid,
                    tags=[t.type_name], status="success",
                    message="Applied", existing_tags=existing,
                    new_tags=[t.type_name], confidence=t.confidence,
                ))
        else:
            for t in new_tags:
                results.append(SyncResult(
                    entity_name=guid, entity_guid=guid,
                    tags=[t.type_name], status="error",
                    message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                ))

        return results

    # -- Search / Glossary -------------------------------------------------

    def basic_search(
        self,
        type_name: str,
        query: str = "",
        classification: str = "",
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """POST /search/basic."""
        body: dict[str, Any] = {
            "typeName": type_name,
            "limit": limit,
            "offset": offset,
        }
        if query:
            body["query"] = query
        if classification:
            body["classification"] = classification
        resp = self._http.post("search/basic", json=body)
        resp.raise_for_status()
        return resp.json()

    def get_all_glossaries(self) -> list[dict[str, Any]]:
        """GET /glossary — list all glossaries."""
        resp = self._http.get("glossary")
        if resp.ok:
            return resp.json()
        return []

    # -- Health / discovery ------------------------------------------------

    def ping(self) -> dict[str, Any]:
        """Lightweight health probe. Returns version info or error."""
        try:
            resp = self._http.get("types/typedefs/headers")
            if resp.ok:
                headers = resp.json()
                return {
                    "ok": True,
                    "classification_count": len([
                        h for h in headers if h.get("category") == "CLASSIFICATION"
                    ]),
                    "entity_type_count": len([
                        h for h in headers if h.get("category") == "ENTITY"
                    ]),
                }
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # -- Batch operations --------------------------------------------------

    def tag_entities_bulk(
        self,
        entity_guids: list[str],
        tags: list[ClassificationTag],
        *,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        """Apply the same classification tags to multiple entities at once.

        Uses the Atlas bulk classification endpoint for efficiency.
        """
        if dry_run:
            return [
                SyncResult(
                    entity_name=guid,
                    entity_guid=guid,
                    tags=[t.type_name for t in tags],
                    status="dry_run",
                    message=f"Would apply {len(tags)} tags",
                )
                for guid in entity_guids
            ]

        # Ensure type definitions exist
        for t in tags:
            self.ensure_classification(
                t.type_name,
                super_types=t.super_types,
                notation=t.notation,
            )

        # POST /entity/bulk/classifications
        payload = {
            "classification": tags[0].to_atlas_payload() if len(tags) == 1 else None,
            "classifications": [t.to_atlas_payload() for t in tags] if len(tags) > 1 else None,
            "entityGuids": entity_guids,
        }
        # Clean None keys
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = self._http.post("entity/bulk/classifications", json=payload)
        if resp.status_code in (200, 204):
            return [
                SyncResult(
                    entity_name=guid,
                    entity_guid=guid,
                    tags=[t.type_name for t in tags],
                    status="success",
                    message="Applied (bulk)",
                )
                for guid in entity_guids
            ]
        return [
            SyncResult(
                entity_name=guid,
                entity_guid=guid,
                tags=[t.type_name for t in tags],
                status="error",
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
            for guid in entity_guids
        ]


# -- helpers ---------------------------------------------------------------


def _attr_def(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "typeName": "string",
        "isOptional": True,
        "cardinality": "SINGLE",
        "valuesMinCount": 0,
        "valuesMaxCount": 1,
        "isUnique": False,
        "isIndexable": True,
        "includeInNotification": False,
        "defaultValue": None,
        "description": description,
    }
