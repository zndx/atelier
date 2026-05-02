# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Sync orchestrator: taxonomy → Atlas classifications, results → entity tags,
classifications → Ranger role/policy derivation.

This module bridges Atelier's classification pipeline output with Atlas and
Ranger, handling the full lifecycle:

1. **Taxonomy sync** — push a hierarchical vocabulary as Atlas classification
   types with superType inheritance (Option D from the BFO reorientation doc).
2. **Entity tagging** — apply classification results to Hive columns in Atlas,
   with dry-run, deduplication, and batch support.
3. **Policy derivation** — generate Ranger tag-based access/masking policies
   from classification types applied to Atlas entities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from atelier.governance.atlas import AtlasClient, ClassificationTag, QualifiedName, SyncResult
from atelier.governance.ranger import RangerClient, RangerRole, PolicyItem, PolicyResource

log = logging.getLogger(__name__)


# -- Taxonomy → Atlas classifications --------------------------------------


@dataclass
class TaxonomyNode:
    """A node in a hierarchical vocabulary, e.g. ``SENS.PID.CI.EMAIL``.

    Maps to Atlas classification types with ``superTypes`` expressing hierarchy.
    The ``notation`` field carries the SKOS-style numeric code (e.g. ``1.1.1.1``).
    """

    code: str  # mnemonic dot-path, e.g. "SENS.PID.CI.EMAIL"
    label: str  # human-readable, e.g. "Email Address"
    notation: str = ""  # SKOS numeric code, e.g. "1.1.1.1"
    parent_code: str = ""  # e.g. "SENS.PID.CI"
    description: str = ""  # long-form text for Atlas typedef description; falls back to label

    @property
    def super_types(self) -> list[str]:
        return [self.parent_code] if self.parent_code else []

    def to_classification_tag(self) -> ClassificationTag:
        return ClassificationTag(
            type_name=self.code,
            notation=self.notation,
            super_types=self.super_types,
        )


def sync_taxonomy_to_atlas(
    atlas: AtlasClient,
    nodes: list[TaxonomyNode],
    *,
    dry_run: bool = False,
) -> TaxonomySyncReport:
    """Push a hierarchical vocabulary to Atlas as classification types.

    Nodes must be sorted parents-first (ancestors before descendants) so that
    superType references resolve. Returns a report of what was created/skipped.
    """
    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for node in nodes:
        existing = atlas.get_classification_def(node.code)
        if existing is not None:
            skipped.append(node.code)
            continue

        if dry_run:
            created.append(node.code)
            continue

        ok = atlas.ensure_classification(
            node.code,
            description=node.description or node.label,
            super_types=node.super_types,
            notation=node.notation,
        )
        if ok:
            created.append(node.code)
        else:
            failed.append(node.code)

    return TaxonomySyncReport(
        dry_run=dry_run,
        created=created,
        skipped=skipped,
        failed=failed,
    )


@dataclass
class TaxonomySyncReport:
    dry_run: bool
    created: list[str]
    skipped: list[str]
    failed: list[str]

    @property
    def total(self) -> int:
        return len(self.created) + len(self.skipped) + len(self.failed)


# -- Classification results → Atlas entity tags ----------------------------


@dataclass
class ColumnClassification:
    """A classification result for a single column."""

    column_name: str
    tags: list[str]
    confidence: float = 0.0
    reason: str = ""


def sync_classifications_to_atlas(
    atlas: AtlasClient,
    table_name: str,
    results: list[ColumnClassification],
    *,
    cluster_name: str = "cm",
    dry_run: bool = False,
) -> list[SyncResult]:
    """Apply classification results to Hive column entities in Atlas.

    Args:
        atlas: Configured AtlasClient.
        table_name: Fully qualified ``database.table`` name.
        results: Per-column classification outputs.
        cluster_name: Atlas cluster (default ``cm``).
        dry_run: Preview without writing.

    Returns:
        Per-column SyncResult list.
    """
    qn = QualifiedName.from_table_name(table_name, cluster=cluster_name)

    # Verify table exists
    table_guid = atlas.get_entity_guid("hive_table", qn.for_table())
    if table_guid is None:
        return [
            SyncResult(
                entity_name=table_name,
                tags=[],
                status="error",
                message=f"Table {table_name} not found in Atlas",
            )
        ]

    all_results: list[SyncResult] = []

    for cr in results:
        col_qn = qn.for_column(cr.column_name)
        col_guid = atlas.get_entity_guid("hive_column", col_qn)

        if col_guid is None:
            all_results.append(SyncResult(
                entity_name=cr.column_name,
                tags=cr.tags,
                status="error",
                message="Column not found in Atlas",
            ))
            continue

        tags = [
            ClassificationTag(
                type_name=t,
                confidence=cr.confidence,
                reason=cr.reason,
            )
            for t in cr.tags
        ]

        tag_results = atlas.tag_entity(col_guid, tags, dry_run=dry_run)
        all_results.extend(tag_results)

    return all_results


# -- Classifications → Ranger policies ------------------------------------


@dataclass
class PolicySpec:
    """Describes a Ranger policy to create from a classification tag."""

    tag: str
    access_types: list[str] = field(default_factory=lambda: ["select"])
    mask_type: str = ""  # e.g. "MASK", "SHUFFLE", "MASK_SHOW_LAST_4"
    roles: list[str] = field(default_factory=list)
    users: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


def derive_ranger_policies(
    ranger: RangerClient,
    service_name: str,
    specs: list[PolicySpec],
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Generate Ranger tag-based policies from classification specs.

    For each spec, creates:
    - A tag-based access policy (if access_types provided)
    - A tag-based masking policy (if mask_type provided)

    Returns list of created (or would-create) policy payloads.
    """
    created: list[dict[str, Any]] = []

    for spec in specs:
        item = PolicyItem(
            users=spec.users,
            groups=spec.groups,
            roles=spec.roles,
            accesses=[{"type": a, "isAllowed": True} for a in spec.access_types],
        )

        # Access policy
        if spec.access_types:
            policy = ranger.build_tag_access_policy(
                name=f"tag_access_{spec.tag}",
                service_name=service_name,
                tag=spec.tag,
                items=[item],
            )
            if dry_run:
                policy["_dry_run"] = True
                created.append(policy)
            else:
                result = ranger.create_policy(policy)
                if result:
                    created.append(result)

        # Masking policy
        if spec.mask_type:
            policy = ranger.build_tag_masking_policy(
                name=f"tag_mask_{spec.tag}",
                service_name=service_name,
                tag=spec.tag,
                mask_type=spec.mask_type,
                items=[item],
            )
            if dry_run:
                policy["_dry_run"] = True
                created.append(policy)
            else:
                result = ranger.create_policy(policy)
                if result:
                    created.append(result)

    return created


# -- Role derivation from entitlements ------------------------------------


def sync_entitlements_to_roles(
    ranger: RangerClient,
    service_name: str,
    role_assignments: dict[str, list[str]],
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Create or update Ranger roles from entitlement-derived assignments.

    Args:
        ranger: Configured RangerClient.
        service_name: Ranger service (e.g. ``hive``).
        role_assignments: Mapping of role_name → list of usernames.
        dry_run: Preview without writing.

    Returns:
        List of role payloads that were created/updated (or would be).
    """
    results: list[dict[str, Any]] = []

    for role_name, usernames in role_assignments.items():
        role = RangerRole(
            name=role_name,
            description=role_name,
        )
        for username in usernames:
            role.add_user(username)

        if dry_run:
            payload = role.to_payload()
            payload["_dry_run"] = True
            results.append(payload)
            continue

        # Check if role exists
        existing = ranger.get_role(role_name, service_name=service_name)
        if existing and existing.id is not None:
            # Merge users
            for username in usernames:
                existing.add_user(username)
            result = ranger.update_role(existing.id, existing)
        else:
            result = ranger.create_role(service_name, role)

        if result:
            results.append(result)

    return results
