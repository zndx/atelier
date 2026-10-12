"""Relational generators for published SKOS IRIs.

The coding harness writes GENERATORS. Each callable returns one table.
insitu_corpus applies an entry only when a projected column is assigned
that IRI.
"""

import random

GENERATORS = {
    "https://signals.zndx.org/sdg#DISPOSITION_CORPORATE_AUDIT_FUNCTION": lambda rng: {
        "table": f"audit_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "audit_function",
                "values": [rng.choice(["financial_statements", "internal_controls", "regulatory_filing"])],
                "iri": "https://signals.zndx.org/sdg#DISPOSITION_CORPORATE_AUDIT_FUNCTION",
            },
            {
                "name": "review_started_on",
                "values": [f"202{rng.randint(0, 5)}-03-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DISPOSITION_OBSERVATION_FUNCTION": lambda rng: {
        "table": f"observation_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "observation_mode",
                "values": [rng.choice(["visible", "monitored", "recorded"])],
                "iri": "https://signals.zndx.org/sdg#DISPOSITION_OBSERVATION_FUNCTION",
            },
            {
                "name": "observed_at",
                "values": [f"202{rng.randint(0, 5)}-06-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DISPOSITION_PRIM_VULNERABILITYSTATE": lambda rng: {
        "table": f"vulnerability_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "vulnerability_state",
                "values": [rng.choice(["exposed", "mitigated", "unassessed"])],
                "iri": "https://signals.zndx.org/sdg#DISPOSITION_PRIM_VULNERABILITYSTATE",
            },
            {
                "name": "recorded_on",
                "values": [f"202{rng.randint(0, 5)}-09-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DISPOSITION_SIMULATION_SOFTWARE": lambda rng: {
        "table": f"simulation_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "simulation_model",
                "values": [rng.choice(["fluid_dynamics", "load_stress", "market_impulse"])],
                "iri": "https://signals.zndx.org/sdg#DISPOSITION_SIMULATION_SOFTWARE",
            },
            {
                "name": "run_started_on",
                "values": [f"202{rng.randint(0, 5)}-02-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DISPOSITION_TEAM_FUNCTION": lambda rng: {
        "table": f"crew_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "team_function",
                "values": [rng.choice(["navigation", "logistics", "medical_response"])],
                "iri": "https://signals.zndx.org/sdg#DISPOSITION_TEAM_FUNCTION",
            },
            {
                "name": "shift_started_on",
                "values": [f"202{rng.randint(0, 5)}-05-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DOM_AGENT": lambda rng: {
        "table": f"agent_directory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "agent",
                "values": [rng.choice(["ag-4471", "ag-9023", "ag-1288"])],
                "iri": "https://signals.zndx.org/sdg#DOM_AGENT",
            },
            {
                "name": "registered_on",
                "values": [f"201{rng.randint(0, 9)}-08-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DOM_SCHOOL_SECURITY_POLICY": lambda rng: {
        "table": f"policy_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "school_security_policy",
                "values": [rng.choice(["access_badging", "lockdown_drill", "visitor_log"])],
                "iri": "https://signals.zndx.org/sdg#DOM_SCHOOL_SECURITY_POLICY",
            },
            {
                "name": "effective_on",
                "values": [f"202{rng.randint(0, 5)}-09-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#DOM_SURVEY_RESPONSE_RECORD": lambda rng: {
        "table": f"survey_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "survey_response_record",
                "values": [rng.choice(["r-88213", "r-90347", "r-76102"])],
                "iri": "https://signals.zndx.org/sdg#DOM_SURVEY_RESPONSE_RECORD",
            },
            {
                "name": "captured_on",
                "values": [f"202{rng.randint(0, 5)}-11-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_ACADEMIC_DEGREE": lambda rng: {
        "table": f"degree_award_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "degree_level",
                "values": [rng.choice(["bachelors", "masters", "doctoral"])],
                "iri": "https://signals.zndx.org/sdg#GDC_ACADEMIC_DEGREE",
            },
            {
                "name": "conferred_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_BOUNDED_METRIC_SERIES": lambda rng: {
        "table": f"engagement_series_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "metric_series",
                "values": [rng.choice(["daily_active", "retention_week", "conversion_rate"])],
                "iri": "https://signals.zndx.org/sdg#GDC_BOUNDED_METRIC_SERIES",
            },
            {
                "name": "decimal_places",
                "values": [str(rng.randint(2, 12))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_CALENDAR_PRODUCT": lambda rng: {
        "table": f"calendar_release_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "calendar_system",
                "values": [rng.choice(["gregorian", "iso_8601", "coptic"])],
                "iri": "https://signals.zndx.org/sdg#GDC_CALENDAR_PRODUCT",
            },
            {
                "name": "reference_month",
                "values": [f"20{rng.randint(0, 5)}-{rng.randint(1, 9):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_CONSTRUCTED_PREDICATE": lambda rng: {
        "table": f"predicate_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "predicate",
                "values": [rng.choice(["excludes", "requires", "must_precede"])],
                "iri": "https://signals.zndx.org/sdg#GDC_CONSTRUCTED_PREDICATE",
            },
            {
                "name": "defined_on",
                "values": [f"20{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_COURSE": lambda rng: {
        "table": f"course_offering_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "course",
                "values": [rng.choice(["cs-210", "bio-104", "eng-302"])],
                "iri": "https://signals.zndx.org/sdg#GDC_COURSE",
            },
            {
                "name": "offered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_COURSE_SELECTION_SET": lambda rng: {
        "table": f"selection_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "selection_set",
                "values": [rng.choice(["core_track_set", "elective_track_set", "capstone_track_set"])],
                "iri": "https://signals.zndx.org/sdg#GDC_COURSE_SELECTION_SET",
            },
            {
                "name": "planned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_CURRICULUM": lambda rng: {
        "table": f"curriculum_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "curriculum",
                "values": [rng.choice(["cs_core_v3", "data_sc_v2", "nurs_prog_v1"])],
                "iri": "https://signals.zndx.org/sdg#GDC_CURRICULUM",
            },
            {
                "name": "effective_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_HOUSING_POLICY_TOPIC": lambda rng: {
        "table": f"policy_topic_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "policy_topic",
                "values": [rng.choice(["lease_renewal", "deposit_refund", "tenant_maintenance"])],
                "iri": "https://signals.zndx.org/sdg#GDC_HOUSING_POLICY_TOPIC",
            },
            {
                "name": "published_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_MEASURED_SERIES_READING": lambda rng: {
        "table": f"series_reading_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "measured_series_reading",
                "values": [f"{rng.randint(1, 99)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#GDC_MEASURED_SERIES_READING",
            },
            {
                "name": "captured_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_PRIM_SCHOOLSECURITYPOLICY": lambda rng: {
        "table": f"security_policy_issue_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "security_policy_revision",
                "values": [rng.choice(["r-2021-04", "r-2022-11", "r-2024-07"])],
                "iri": "https://signals.zndx.org/sdg#GDC_PRIM_SCHOOLSECURITYPOLICY",
            },
            {
                "name": "adopted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_PRIM_SERVICELEVELAGREEMENT": lambda rng: {
        "table": f"sla_contract_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "sla_tier",
                "values": [rng.choice(["platinum", "gold", "standard"])],
                "iri": "https://signals.zndx.org/sdg#GDC_PRIM_SERVICELEVELAGREEMENT",
            },
            {
                "name": "target_uptime_pct",
                "values": [rng.choice(["99.5", "99.9", "99.95"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_SCORED_ASSESSMENT_BOUNDS": lambda rng: {
        "table": f"assessment_score_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "confidence_score",
                "values": [f"{rng.randint(0, 9)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#GDC_SCORED_ASSESSMENT_BOUNDS",
            },
            {
                "name": "scored_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_SERVICE_LEVEL_AGREEMENT": lambda rng: {
        "table": f"sla_terms_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "qos_class",
                "values": [rng.choice(["latency_bounded", "throughput_guaranteed", "loss_limited"])],
                "iri": "https://signals.zndx.org/sdg#GDC_SERVICE_LEVEL_AGREEMENT",
            },
            {
                "name": "reviewed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_SOFTWARE_INTERFACE": lambda rng: {
        "table": f"data_export_link_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "export_target",
                "values": [rng.choice(["sql_dump", "csv_batch", "json_stream"])],
                "iri": "https://signals.zndx.org/sdg#GDC_SOFTWARE_INTERFACE",
            },
            {
                "name": "batch_interval",
                "values": [rng.choice(["5m", "15m", "1h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GDC_VERSIONED_FORM_LINEAGE": lambda rng: {
        "table": f"intake_form_revision_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "form_version",
                "values": [rng.choice(["v2_1", "v2_2", "v3_0"])],
                "iri": "https://signals.zndx.org/sdg#GDC_VERSIONED_FORM_LINEAGE",
            },
            {
                "name": "revised_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ASSESSMENT_PARTICIPANT": lambda rng: {
        "table": f"assessment_panel_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "panel_membership",
                "values": [rng.choice(["cross_functional", "peer_review", "expert_panel"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ASSESSMENT_PARTICIPANT",
            },
            {
                "name": "convened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ATHLETIC_MONITORING_ACTIVITY": lambda rng: {
        "table": f"training_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "monitoring_activity",
                "values": [rng.choice(["sprint_review", "lift_technique_check", "match_metrics_track"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ATHLETIC_MONITORING_ACTIVITY",
            },
            {
                "name": "camera_id",
                "values": [f"cam-{rng.randint(1, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_BUILDING_FLOOR_STRUCTURE": lambda rng: {
        "table": f"floor_assembly_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "floor_structure",
                "values": [rng.choice(["cast_slab", "joist_deck", "steel_deck"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_BUILDING_FLOOR_STRUCTURE",
            },
            {
                "name": "material_grade",
                "values": [rng.choice(["c25", "c30", "s275"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CERTIFICATION_TIME": lambda rng: {
        "table": f"certificate_issue_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "certification_time",
                "values": [f"{rng.randint(6, 22):02d}:00"],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CERTIFICATION_TIME",
            },
            {
                "name": "certificate_id",
                "values": [f"ct-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CLINICAL_PERFORMANCE_DATA": lambda rng: {
        "table": f"clinical_quality_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "performance_data",
                "values": [f"{rng.randint(60, 99)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CLINICAL_PERFORMANCE_DATA",
            },
            {
                "name": "unit_code",
                "values": [rng.choice(["icu-2", "med-b", "onc-1"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_COLLEGE_OFFICIAL": lambda rng: {
        "table": f"staff_directory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "official_position",
                "values": [rng.choice(["dean_of_studies", "academic_vice_president", "registrar"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_COLLEGE_OFFICIAL",
            },
            {
                "name": "appointed_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CORPORATE_ACTIVITY": lambda rng: {
        "table": f"activity_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "activity_type",
                "values": [rng.choice(["product_launch", "market_expansion", "supplier_negotiation"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CORPORATE_ACTIVITY",
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CORPORATE_REPUTATION_METRIC": lambda rng: {
        "table": f"reputation_study_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "metric_kind",
                "values": [rng.choice(["sentiment_score", "brand_equity_index", "stakeholder_trust_index"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CORPORATE_REPUTATION_METRIC",
            },
            {
                "name": "reported_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CRIMINAL_JUSTICE_PROCESS": lambda rng: {
        "table": f"case_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "process_stage",
                "values": [rng.choice(["arraignment", "pretrial_conference", "sentencing"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CRIMINAL_JUSTICE_PROCESS",
            },
            {
                "name": "filed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CRIMINAL_JUSTICE_SYSTEM": lambda rng: {
        "table": f"justice_unit_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "justice_system_unit",
                "values": [rng.choice(["police_agency", "prosecutors_office", "corrections_facility"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CRIMINAL_JUSTICE_SYSTEM",
            },
            {
                "name": "jurisdiction",
                "values": [rng.choice(["federal", "state", "county"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CULTURAL_ATTRIBUTE": lambda rng: {
        "table": f"culture_assessment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "cultural_attribute",
                "values": [rng.choice(["psychological_safety", "risk_tolerance", "collaboration"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CULTURAL_ATTRIBUTE",
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_CYBER_SECURITY_INCIDENT": lambda rng: {
        "table": f"incident_response_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "incident_class",
                "values": [rng.choice(["unauthorized_intrusion", "malware_propagation", "data_exfiltration"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_CYBER_SECURITY_INCIDENT",
            },
            {
                "name": "detected_at",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}T{rng.randint(0, 23):02d}:00:00Z"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DATA_INPUT": lambda rng: {
        "table": f"intake_batch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "input_mode",
                "values": [rng.choice(["form_submission", "kiosk_entry", "paper_scan"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DATA_INPUT",
            },
            {
                "name": "target_store",
                "values": [rng.choice(["crm", "erp", "claims_db"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DECISION_ACTION": lambda rng: {
        "table": f"decision_action_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "decision_action",
                "values": [rng.choice(["rank_alternatives", "select_option", "record_choice"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DECISION_ACTION",
            },
            {
                "name": "executed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DECISION_MAKING_PROCESS": lambda rng: {
        "table": f"decision_process_instance_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "decision_process",
                "values": [rng.choice(["budget_review_cycle", "vendor_selection_round", "go_no_go_board"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DECISION_MAKING_PROCESS",
            },
            {
                "name": "participant_ref",
                "values": [f"pr-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DECISION_MAKING_PROCESS_DEFINITION": lambda rng: {
        "table": f"process_definition_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "process_definition",
                "values": [rng.choice(["planning_act_with_parts", "choice_transition_sequence", "governed_plan_sequence"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DECISION_MAKING_PROCESS_DEFINITION",
            },
            {
                "name": "applies_to",
                "values": [rng.choice(["governance", "project_management", "operational_planning"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DECISION_TRANSITION_ACTION": lambda rng: {
        "table": f"decision_transition_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "transition_action",
                "values": [rng.choice(["handoff_to_review", "advance_to_signoff", "return_to_draft"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DECISION_TRANSITION_ACTION",
            },
            {
                "name": "occurred_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DEGRADED_SYSTEM_CONDITION": lambda rng: {
        "table": f"system_state_action_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "degraded_system_condition",
                "values": [rng.choice(["reduced_throughput", "partial_capacity_loss", "impaired_response_time"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DEGRADED_SYSTEM_CONDITION",
            },
            {
                "name": "reported_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_DOMESTIC_VIOLENCE_VICTIM_CLIENT": lambda rng: {
        "table": f"advocacy_casework_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "victim_client",
                "values": [f"cl-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#GENERIC_DOMESTIC_VIOLENCE_VICTIM_CLIENT",
            },
            {
                "name": "case_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ECONOMIC_GOODS_AND_SERVICES": lambda rng: {
        "table": f"trade_schedule_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "economic_goods_and_services",
                "values": [rng.choice(["telecom_rental", "freight_handling", "software_license"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ECONOMIC_GOODS_AND_SERVICES",
            },
            {
                "name": "tariff_class",
                "values": [rng.choice(["duty_free", "standard_duty", "concessional_duty"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_EXECUTIVE_EXECUTION_FUNCTION": lambda rng: {
        "table": f"execution_wave_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "execution_function",
                "values": [rng.choice(["rollout_phase", "reorg_implementation", "strategy_decomposition"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_EXECUTIVE_EXECUTION_FUNCTION",
            },
            {
                "name": "directive_ref",
                "values": [f"dx-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_EXECUTIVE_LEADERSHIP_ROLE": lambda rng: {
        "table": f"leadership_seat_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "leadership_role",
                "values": [rng.choice(["ceo_seat", "cfo_seat", "coo_seat"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_EXECUTIVE_LEADERSHIP_ROLE",
            },
            {
                "name": "mandate_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_INSTRUMENT": lambda rng: {
        "table": f"instrument_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "instrument_class",
                "values": [rng.choice(["equity", "bond", "swap"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_INSTRUMENT",
            },
            {
                "name": "clearing_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_MANAGEMENT_PROCESS": lambda rng: {
        "table": f"finance_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "management_activity",
                "values": [rng.choice(["budget_formulation", "cash_reconciliation", "forecast_renewal"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_MANAGEMENT_PROCESS",
            },
            {
                "name": "cycle_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_STRATEGY_OVERSIGHT": lambda rng: {
        "table": f"oversight_round_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "oversight_stage",
                "values": [rng.choice(["capital_allocation", "risk_limit_review", "liquidity_review"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_FINANCIAL_STRATEGY_OVERSIGHT",
            },
            {
                "name": "reviewed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_IDENTIFIER_BEARER_UNION": lambda rng: {
        "table": f"registry_entry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "registration_form",
                "values": [rng.choice(["sole_proprietorship", "limited_liability_company", "general_partnership"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_IDENTIFIER_BEARER_UNION",
            },
            {
                "name": "registered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_INFORMATION_CAPABILITY": lambda rng: {
        "table": f"capability_assessment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "information_capability",
                "values": [rng.choice(["message_broadcast", "record_processing", "report_distribution"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_INFORMATION_CAPABILITY",
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_INVESTOR_RELATIONS_ACTIVITY": lambda rng: {
        "table": f"ir_calendar_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "relations_activity",
                "values": [rng.choice(["earnings_call", "quarterly_report_distro", "shareholder_meeting"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_INVESTOR_RELATIONS_ACTIVITY",
            },
            {
                "name": "scheduled_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_LEGAL_PROFESSIONAL": lambda rng: {
        "table": f"counsel_retention_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advocate_role",
                "values": [rng.choice(["litigation_counsel", "in_house_advisor", "contract_drafting"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_LEGAL_PROFESSIONAL",
            },
            {
                "name": "admitted_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_LOGIN_CREDENTIALS": lambda rng: {
        "table": f"credential_store_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "credential_record",
                "values": [f"lc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#GENERIC_LOGIN_CREDENTIALS",
            },
            {
                "name": "access_scoped_to",
                "values": [rng.choice(["admin_console", "batch_pipeline", "support_portal"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_MONITORING_PROTOCOL": lambda rng: {
        "table": f"sensor_monitoring_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "monitoring_protocol",
                "values": [rng.choice(["vibration_sweep", "thermal_baseline", "pressure_drift"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_MONITORING_PROTOCOL",
            },
            {
                "name": "monitored_sensor_id",
                "values": [f"sn-{rng.randint(1, 999):03d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_OFFLINE_CHANNEL": lambda rng: {
        "table": f"dissemination_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "offline_channel",
                "values": [rng.choice(["tv_broadcast", "radio_spot", "print_insert"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_OFFLINE_CHANNEL",
            },
            {
                "name": "market_region",
                "values": [rng.choice(["metro_north", "rural_west", "port_cities"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ONLINE_CHANNEL": lambda rng: {
        "table": f"campaign_delivery_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "online_channel",
                "values": [rng.choice(["website_banner", "social_feed_post", "sms_broadcast"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ONLINE_CHANNEL",
            },
            {
                "name": "launched_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_OPERATIONAL_PROTOCOL": lambda rng: {
        "table": f"protocol_brief_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "protocol_scope",
                "values": [rng.choice(["loading_dock_handling", "night_shift_rounds", "chemical_storage_access"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_OPERATIONAL_PROTOCOL",
            },
            {
                "name": "in_effect_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_CULTURE": lambda rng: {
        "table": f"culture_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "culture_dimension",
                "values": [rng.choice(["feedback_norm", "decision_style", "accountability_style"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_CULTURE",
            },
            {
                "name": "surveyed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_HEALTH_METRIC": lambda rng: {
        "table": f"workforce_pulse_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "health_indicator",
                "values": [rng.choice(["engagement_index", "near_miss_rate", "wellbeing_pulse_score"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_HEALTH_METRIC",
            },
            {
                "name": "collected_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_LEADERSHIP_PROCESS": lambda rng: {
        "table": f"leadership_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "leadership_process",
                "values": [rng.choice(["succession_nominations", "board_orientation", "strategy_handoff"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_LEADERSHIP_PROCESS",
            },
            {
                "name": "role_ref",
                "values": [f"lr-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_LEADERSHIP_ROLE": lambda rng: {
        "table": f"board_seat_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "organizational_leadership_role",
                "values": [rng.choice(["vp_field_operations", "chief_data_officer", "executive_sponsor"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_ORGANIZATIONAL_LEADERSHIP_ROLE",
            },
            {
                "name": "took_office_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_PRACTICE_MANAGEMENT_TOPIC": lambda rng: {
        "table": f"practice_training_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "practice_management_topic",
                "values": [rng.choice(["scheduling_standards", "billing_procedures", "licensure_renewal"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_PRACTICE_MANAGEMENT_TOPIC",
            },
            {
                "name": "session_delivered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_PREVALENCE_RATE": lambda rng: {
        "table": f"cohort_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "prevalence_rate",
                "values": [f"0.{rng.randint(1, 49):02d}"],
                "iri": "https://signals.zndx.org/sdg#GENERIC_PREVALENCE_RATE",
            },
            {
                "name": "population_cohort",
                "values": [rng.choice(["adult_cohort", "pediatric_cohort", "senior_cohort"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_PRIM_RISKMANAGEMENTACTIVITY": lambda rng: {
        "table": f"risk_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "risk_management_activity",
                "values": [rng.choice(["hazard_identification", "impact_assessment", "mitigation_planning"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_PRIM_RISKMANAGEMENTACTIVITY",
            },
            {
                "name": "owner_unit",
                "values": [rng.choice(["compliance_team", "operations_team", "finance_team"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_PRIM_SOCIALNETWORKINGSERVICEUSAGE": lambda rng: {
        "table": f"platform_usage_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_used",
                "values": [rng.choice(["photo_feed_app", "microblog_app", "forum_app"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_PRIM_SOCIALNETWORKINGSERVICEUSAGE",
            },
            {
                "name": "session_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_PRIM_TARIFFSTRUCTURE": lambda rng: {
        "table": f"tariff_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "structure_id",
                "values": [rng.choice(["res_tiered", "com_flat", "ind_block"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_PRIM_TARIFFSTRUCTURE",
            },
            {
                "name": "effective_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_RAFFLE_CONDUCTING_ACTIVITY": lambda rng: {
        "table": f"raffle_draw_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "draw_round",
                "values": [rng.choice(["ticket_sales", "entry_cutoff", "ball_draw"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_RAFFLE_CONDUCTING_ACTIVITY",
            },
            {
                "name": "venue_code",
                "values": [f"vf-{rng.randint(10, 99)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_REFERRAL_PROCESS": lambda rng: {
        "table": f"referral_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "intake_stage",
                "values": [rng.choice(["intake", "service_coordination", "provider_match"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_REFERRAL_PROCESS",
            },
            {
                "name": "referral_id",
                "values": [f"rf-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_RENDERING_ENVIRONMENT": lambda rng: {
        "table": f"rendering_bay_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "rendering_environment",
                "values": [rng.choice(["studio_a", "render_bay_2", "gpu_room_c"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_RENDERING_ENVIRONMENT",
            },
            {
                "name": "rendered_artifact_ref",
                "values": [f"ra-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_RENDERING_ENVIRONMENT_DEFINITION": lambda rng: {
        "table": f"environment_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "environment_definition",
                "values": [rng.choice(["indoor_render_suite", "visualization_room", "review_theater"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_RENDERING_ENVIRONMENT_DEFINITION",
            },
            {
                "name": "lighting_profile",
                "values": [rng.choice(["dimmable_led", "diffused_panel", "neutral_6500k"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_REPORTING_CHANNEL": lambda rng: {
        "table": f"disclosure_route_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "channel_mode",
                "values": [rng.choice(["hotline", "secure_portal", "designated_inbox"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_REPORTING_CHANNEL",
            },
            {
                "name": "acknowledged_within",
                "values": [rng.choice(["24h", "48h", "72h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_REQUESTED_PROVIDER_TYPE": lambda rng: {
        "table": f"provider_change_request_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "requested_service_type",
                "values": [rng.choice(["managed_hosting", "backup_retention", "load_balancing"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_REQUESTED_PROVIDER_TYPE",
            },
            {
                "name": "request_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_RESEARCH_ACTIVITY": lambda rng: {
        "table": f"research_project_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "research_activity",
                "values": [rng.choice(["field_survey", "lab_trial", "data_analysis"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_RESEARCH_ACTIVITY",
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(0, 5)}-03-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_SECRETARY_ROLE": lambda rng: {
        "table": f"secretary_seat_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "secretary_role",
                "values": [rng.choice(["clerk_of_court", "cabinet_secretary", "assembly_chief_of_staff"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_SECRETARY_ROLE",
            },
            {
                "name": "appointed_on",
                "values": [f"202{rng.randint(0, 5)}-05-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_SOCIAL_NETWORKING_SERVICE_TYPE": lambda rng: {
        "table": f"service_taxonomy_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_type",
                "values": [rng.choice(["profile_directory", "message_thread", "group_feed"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_SOCIAL_NETWORKING_SERVICE_TYPE",
            },
            {
                "name": "catalog_version",
                "values": [f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_VEHICLE_POWER_TYPE": lambda rng: {
        "table": f"fleet_inventory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "power_type",
                "values": [rng.choice(["diesel", "non_diesel"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_VEHICLE_POWER_TYPE",
            },
            {
                "name": "inspected_on",
                "values": [f"202{rng.randint(0, 5)}-09-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_WAVELENGTH_VARIANT": lambda rng: {
        "table": f"laser_module_test_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "wavelength_variant",
                "values": [rng.choice(["930nm", "940nm"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_WAVELENGTH_VARIANT",
            },
            {
                "name": "measured_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_WORKPLACE_CULTURE_DIMENSION": lambda rng: {
        "table": f"workplace_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "workplace_culture_dimension",
                "values": [rng.choice(["psychological_safety", "decision_style", "accountability_style"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_WORKPLACE_CULTURE_DIMENSION",
            },
            {
                "name": "team_surveyed",
                "values": [rng.choice(["engineering", "sales", "operations"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#GENERIC_YOUNG_PHYSICIAN_AMBASSADOR_ROLE": lambda rng: {
        "table": f"ambassador_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "young_physician_ambassador",
                "values": [rng.choice(["outreach_ambassador", "mentoring_captain", "conference_liaison"])],
                "iri": "https://signals.zndx.org/sdg#GENERIC_YOUNG_PHYSICIAN_AMBASSADOR_ROLE",
            },
            {
                "name": "term_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_ADOPTION_APPLICATION_FORM": lambda rng: {
        "table": f"adoption_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "adoption_application_form",
                "values": [rng.choice(["initial_filing", "amended_filing", "addendum_filing"])],
                "iri": "https://signals.zndx.org/sdg#ICE_ADOPTION_APPLICATION_FORM",
            },
            {
                "name": "version_number",
                "values": [str(rng.randint(1, 9))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_APPEAL_WAIVER_CERTIFICATION": lambda rng: {
        "table": f"waiver_certification_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "waiver_certification",
                "values": [f"awc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_APPEAL_WAIVER_CERTIFICATION",
            },
            {
                "name": "certifying_board_code",
                "values": [rng.choice(["boe-04", "boe-11", "boe-17"])],
            },
            {
                "name": "certified_appeal_waiver_ref",
                "values": [f"aw-{rng.randint(100, 999)}"],
            },
            {
                "name": "certified_at",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}T{rng.randint(0, 23):02d}:00:00Z"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_BACK_SHELL_DIMENSION_SPEC": lambda rng: {
        "table": f"back_shell_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "back_shell_dimension_spec",
                "values": [f"bds-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_BACK_SHELL_DIMENSION_SPEC",
            },
            {
                "name": "width_measurement_mm",
                "values": [str(rng.randint(320, 960))],
            },
            {
                "name": "height_measurement_mm",
                "values": [str(rng.randint(240, 720))],
            },
            {
                "name": "depth_value",
                "values": [str(rng.randint(40, 220))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_BREEDING_PROGRAM_APPLICATION": lambda rng: {
        "table": f"breeding_application_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "breeding_program_application",
                "values": [f"bpa-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_BREEDING_PROGRAM_APPLICATION",
            },
            {
                "name": "form_section_code",
                "values": [rng.choice(["sec-01", "sec-02", "sec-03"])],
            },
            {
                "name": "horse_identity",
                "values": [f"hi-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "filed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_BUSINESS_PLAN": lambda rng: {
        "table": f"business_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "business_plan",
                "values": [f"bp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_BUSINESS_PLAN",
            },
            {
                "name": "described_scenario_ref",
                "values": [f"scen-{rng.randint(100, 999)}"],
            },
            {
                "name": "targeted_funding_situation",
                "values": [rng.choice(["seed_round", "series_a", "expansion_round"])],
            },
            {
                "name": "drafted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_CAMPAIGN_COORDINATION": lambda rng: {
        "table": f"campaign_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "coordinated_campaign",
                "values": [rng.choice(["flood_relief_push", "autumn_enrollment_wave", "clinic_promo_cycle"])],
                "iri": "https://signals.zndx.org/sdg#ICE_CAMPAIGN_COORDINATION",
            },
            {
                "name": "online_channel_ref",
                "values": [f"onch-{rng.randint(100, 999)}"],
            },
            {
                "name": "offline_channel_ref",
                "values": [f"offch-{rng.randint(10, 99)}"],
            },
            {
                "name": "engagement_metric_value",
                "values": [f"{rng.randint(10, 99)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_CLINICAL_INTELLIGENCE_ARTIFACT": lambda rng: {
        "table": f"clinical_intel_artifact_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "intelligence_artifact",
                "values": [rng.choice(["sepsis_risk_signal", "stewardship_flag", "readmission_alert"])],
                "iri": "https://signals.zndx.org/sdg#ICE_CLINICAL_INTELLIGENCE_ARTIFACT",
            },
            {
                "name": "analytic_method_code",
                "values": [rng.choice(["bayes_net_v3", "gradient_boost_11", "rule_set_2024"])],
            },
            {
                "name": "produced_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_CLINICAL_PERFORMANCE_DATA_EVALUATION": lambda rng: {
        "table": f"performance_evaluation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "evaluation_record",
                "values": [rng.choice(["surgical_volume_eval", "antibiotic_timing_eval", "order_compliance_eval"])],
                "iri": "https://signals.zndx.org/sdg#ICE_CLINICAL_PERFORMANCE_DATA_EVALUATION",
            },
            {
                "name": "clinical_system_code",
                "values": [rng.choice(["emr_orion", "lims_north", "pacs_2"])],
            },
            {
                "name": "clinician_id",
                "values": [f"clin-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "metric_value",
                "values": [f"{rng.randint(55, 99)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_CONFERENCE_PROGRAMME": lambda rng: {
        "table": f"conference_programme_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "programme_issue",
                "values": [rng.choice(["oncology_annual_prog", "biomed_symposium_prog", "fellows_workshop_prog"])],
                "iri": "https://signals.zndx.org/sdg#ICE_CONFERENCE_PROGRAMME",
            },
            {
                "name": "presentation_ref",
                "values": [f"pres-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "session_date",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_CORPORATE_REPUTATION_INFLUENCE_SITUATION": lambda rng: {
        "table": f"reputation_briefing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "reputation_influence_situation",
                "values": [rng.choice(["investor_roadshow_cycle", "stakeholder_briefing_cycle", "community_engagement_cycle"])],
                "iri": "https://signals.zndx.org/sdg#ICE_CORPORATE_REPUTATION_INFLUENCE_SITUATION",
            },
            {
                "name": "decision_reviewed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_COURT_DOCUMENT": lambda rng: {
        "table": f"court_filing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "court_document",
                "values": [rng.choice(["pretrial_motion", "evidence_submission", "final_judgment"])],
                "iri": "https://signals.zndx.org/sdg#ICE_COURT_DOCUMENT",
            },
            {
                "name": "filed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ABUSE_INDICATOR": lambda rng: {
        "table": f"manifestation_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "abuse_indicator",
                "values": [rng.choice(["unexplained_injury_flag", "bruise_pattern_flag", "wound_documentation_flag"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ABUSE_INDICATOR",
            },
            {
                "name": "manifests_as_ref",
                "values": [f"pm-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ACADEMIC_PROGRAM_REQUIREMENT": lambda rng: {
        "table": f"program_requirement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "program_requirement",
                "values": [rng.choice(["core_credit_block", "lab_credit_block", "capstone_credit_block"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ACADEMIC_PROGRAM_REQUIREMENT",
            },
            {
                "name": "min_credit_hours",
                "values": [str(rng.randint(3, 6))],
            },
            {
                "name": "max_credit_hours",
                "values": [str(rng.randint(8, 15))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ASSESSMENT_STANDARD": lambda rng: {
        "table": f"assessment_standard_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "assessment_standard",
                "values": [rng.choice(["clinical_scale_v2", "behavioral_rubric_3", "diagnostic_cutpoint_1"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ASSESSMENT_STANDARD",
            },
            {
                "name": "assessment_method_ref",
                "values": [f"am-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CALENDAR_PRODUCT_SPECIFICATION": lambda rng: {
        "table": f"calendar_product_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "calendar_product_spec",
                "values": [rng.choice(["ics_sync_v2", "room_booking_v3", "holiday_feed_v1"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CALENDAR_PRODUCT_SPECIFICATION",
            },
            {
                "name": "sync_protocol",
                "values": [rng.choice(["icalendar", "webcal", "ics_feed"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CALL_VOLUME_METRIC": lambda rng: {
        "table": f"call_volume_metric_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "call_volume_metric",
                "values": [rng.choice(["calls_per_hour", "calls_per_day", "transfers_per_hour"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CALL_VOLUME_METRIC",
            },
            {
                "name": "count_window",
                "values": [rng.choice(["15m", "1h", "8h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CLOUD_STORAGE_ALLOCATION": lambda rng: {
        "table": f"storage_allocation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "cloud_storage_allocation",
                "values": [f"sa-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_CLOUD_STORAGE_ALLOCATION",
            },
            {
                "name": "allocated_student_ref",
                "values": [f"st-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "storage_volume",
                "values": [rng.choice(["50gb", "100gb", "250gb"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_HEIGHT_MEASUREMENT": lambda rng: {
        "table": f"elevation_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "height_measurement",
                "values": [f"{rng.randint(120, 480)}.{rng.randint(0, 9)}mm"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_HEIGHT_MEASUREMENT",
            },
            {
                "name": "structure_ref",
                "values": [f"st-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "measured_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_LEADERSHIP_EXPERIENCE_REQUIREMENT": lambda rng: {
        "table": f"role_requisition_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "leadership_experience_requirement",
                "values": [f"{rng.randint(2, 12)}.{rng.randint(0, 9)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_LEADERSHIP_EXPERIENCE_REQUIREMENT",
            },
            {
                "name": "requisition_ref",
                "values": [f"req-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "posted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_MEASUREMENT_PARAMETER": lambda rng: {
        "table": f"calibration_protocol_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "measurement_parameter",
                "values": [rng.choice(["ambient_temperature", "axial_load", "surface_roughness"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_MEASUREMENT_PARAMETER",
            },
            {
                "name": "unit_symbol",
                "values": [rng.choice(["degc", "kn", "kpa"])],
            },
            {
                "name": "calibrated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_NETWORK_RESOURCE_MAPPING": lambda rng: {
        "table": f"routing_binding_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_resource_mapping",
                "values": [f"nrm-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_NETWORK_RESOURCE_MAPPING",
            },
            {
                "name": "network_resource_ref",
                "values": [f"nr-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "traffic_path_ref",
                "values": [f"tp-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_NUTRITIONAL_STATUS_INDICATOR": lambda rng: {
        "table": f"nutrition_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "nutritional_status_indicator",
                "values": [rng.choice(["underweight", "normal_range", "overweight"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_NUTRITIONAL_STATUS_INDICATOR",
            },
            {
                "name": "measured_quantity",
                "values": [rng.choice(["bmi", "mid_upper_arm_circ", "weight_for_age"])],
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_OBSERVATORY_MOUNTING_CONFIGURATION": lambda rng: {
        "table": f"mount_allocation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mounting_configuration",
                "values": [rng.choice(["alt_az_pair", "fork_yoke", "dovetail_rack"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_OBSERVATORY_MOUNTING_CONFIGURATION",
            },
            {
                "name": "instrument_ref",
                "values": [f"tel-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "structure_ref",
                "values": [f"mnt-{rng.randint(10, 99)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ORDER_CONSTRAINT": lambda rng: {
        "table": f"campaign_sequence_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "order_constraint",
                "values": [rng.choice(["lead_slot_only", "must_precede_teaser", "final_week_only"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_ORDER_CONSTRAINT",
            },
            {
                "name": "calendar_item_ref",
                "values": [f"promo-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "sequence_position",
                "values": [str(rng.randint(1, 12))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_PAYMENT_METHOD_CATEGORY": lambda rng: {
        "table": f"payment_category_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "payment_method_category",
                "values": [rng.choice(["card_network", "direct_debit", "wallet_balance"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_PAYMENT_METHOD_CATEGORY",
            },
            {
                "name": "funding_source",
                "values": [rng.choice(["operating_fund", "grant_pool", "reserve_fund"])],
            },
            {
                "name": "published_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_RETIREMENT_BENEFIT_OPTION": lambda rng: {
        "table": f"benefit_election_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "retirement_benefit_option",
                "values": [rng.choice(["survivor_annuity", "joint_life", "qualified_pre_retirement"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_RETIREMENT_BENEFIT_OPTION",
            },
            {
                "name": "survivor_percentage",
                "values": [rng.choice(["50", "67", "75"])],
            },
            {
                "name": "guarantee_months",
                "values": [rng.choice(["12", "24", "36"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SEQUENCING_COVERAGE_PARAMETER": lambda rng: {
        "table": f"coverage_parameter_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "sequencing_coverage_parameter",
                "values": [f"{rng.randint(1, 4)}.{rng.randint(0, 9)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SEQUENCING_COVERAGE_PARAMETER",
            },
            {
                "name": "sequence_position",
                "values": [str(rng.randint(1, 12))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SERVICE_CHARGE_RATE": lambda rng: {
        "table": f"fee_schedule_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_charge_rate",
                "values": [f"{rng.randint(5, 40)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SERVICE_CHARGE_RATE",
            },
            {
                "name": "invoice_cycle",
                "values": [rng.choice(["monthly", "quarterly", "annual"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SIGNIFICANCE_LEVEL": lambda rng: {
        "table": f"qualification_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "significance_level",
                "values": [rng.choice(["0.05", "0.01", "0.10"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SIGNIFICANCE_LEVEL",
            },
            {
                "name": "test_type",
                "values": [rng.choice(["t_test", "chi_square", "anova"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SMALL_GROUP_SURCHARGE_RULE": lambda rng: {
        "table": f"surcharge_rule_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "small_group_surcharge_rule",
                "values": [f"sgsr-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_SMALL_GROUP_SURCHARGE_RULE",
            },
            {
                "name": "min_group_size",
                "values": [str(rng.randint(4, 12))],
            },
            {
                "name": "surcharge_amount",
                "values": [f"{rng.randint(10, 99)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_TEST_STATISTIC": lambda rng: {
        "table": f"hypothesis_test_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "test_statistic",
                "values": [f"{rng.randint(0, 9)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_TEST_STATISTIC",
            },
            {
                "name": "test_kind",
                "values": [rng.choice(["t_test", "f_test", "chi_square"])],
            },
            {
                "name": "sample_size",
                "values": [str(rng.randint(40, 900))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_THERMAL_OPERATING_RANGE_CONSTRAINT": lambda rng: {
        "table": f"thermal_range_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "thermal_operating_range_constraint",
                "values": [f"torc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_THERMAL_OPERATING_RANGE_CONSTRAINT",
            },
            {
                "name": "min_temp_c",
                "values": [str(rng.randint(0, 20))],
            },
            {
                "name": "max_temp_c",
                "values": [str(rng.randint(40, 85))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_YOUNG_PHYSICIAN_ELIGIBILITY_CONSTRAINT": lambda rng: {
        "table": f"physician_eligibility_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "young_physician_eligibility_constraint",
                "values": [f"ypec-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_YOUNG_PHYSICIAN_ELIGIBILITY_CONSTRAINT",
            },
            {
                "name": "max_age_years",
                "values": [str(rng.randint(35, 45))],
            },
            {
                "name": "max_practice_years",
                "values": [str(rng.randint(2, 8))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_YOUTH_HEALTH_CARE_SUMMARY": lambda rng: {
        "table": f"youth_health_summary_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "youth_health_care_summary",
                "values": [f"yhc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESCRIPTIVE_YOUTH_HEALTH_CARE_SUMMARY",
            },
            {
                "name": "health_status_band",
                "values": [rng.choice(["stable", "improving", "under_watch"])],
            },
            {
                "name": "data_format",
                "values": [rng.choice(["hl7_cda", "fhir_uscore", "cda_r2"])],
            },
            {
                "name": "documented_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_AGENDA_SUBJECT": lambda rng: {
        "table": f"board_agenda_item_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "agenda_subject",
                "values": [rng.choice(["budget_ratification", "facility_acquisition", "executive_compensation"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_AGENDA_SUBJECT",
            },
            {
                "name": "item_number",
                "values": [str(rng.randint(1, 12))],
            },
            {
                "name": "meeting_date",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_CLIENT_PROFILE": lambda rng: {
        "table": f"client_profile_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "client_profile",
                "values": [f"cp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_CLIENT_PROFILE",
            },
            {
                "name": "service_preference",
                "values": [rng.choice(["auto_renew", "paper_statements", "digital_only"])],
            },
            {
                "name": "opened_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_EXPORT_EXECUTION": lambda rng: {
        "table": f"pdf_export_profile_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "export_execution",
                "values": [f"ex-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_EXPORT_EXECUTION",
            },
            {
                "name": "page_range",
                "values": [rng.choice(["all_pages", "first_ten", "selected_sections"])],
            },
            {
                "name": "rendered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_GOVERNANCE_FUNCTION": lambda rng: {
        "table": f"governance_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "governance_function",
                "values": [rng.choice(["board_ratification", "policy_directive", "oversight_mandate"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_GOVERNANCE_FUNCTION",
            },
            {
                "name": "governing_body_ref",
                "values": [f"org-{rng.randint(100, 999)}"],
            },
            {
                "name": "ratified_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_HORSE_IDENTITY": lambda rng: {
        "table": f"stud_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "horse_identity",
                "values": [f"hi-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_HORSE_IDENTITY",
            },
            {
                "name": "registered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_MATHEMATICAL_CONCEPT": lambda rng: {
        "table": f"syllabus_topic_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mathematical_concept",
                "values": [rng.choice(["group_axioms", "limit_convergence", "eigenvalue_spectral"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_MATHEMATICAL_CONCEPT",
            },
            {
                "name": "course_ref",
                "values": [f"mth-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_SOCIAL_NETWORKING_SERVICE_PLATFORM": lambda rng: {
        "table": f"platform_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_platform",
                "values": [rng.choice(["circlefeed", "linkhub", "postloop"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_SOCIAL_NETWORKING_SERVICE_PLATFORM",
            },
            {
                "name": "designated_service_type",
                "values": [rng.choice(["photo_feed", "microblog", "forum_thread"])],
            },
            {
                "name": "listed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_SUBJECT_CONCEPT": lambda rng: {
        "table": f"denotation_link_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "subject_concept",
                "values": [rng.choice(["physical_object", "digital_record", "abstract_entity"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DESIGNATIVE_SUBJECT_CONCEPT",
            },
            {
                "name": "referencing_artifact_ref",
                "values": [f"art-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "denoted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DEVICE_INVENTORY_RECORD": lambda rng: {
        "table": f"device_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "device_inventory_record",
                "values": [f"dinr-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DEVICE_INVENTORY_RECORD",
            },
            {
                "name": "enumerated_device_ref",
                "values": [f"md-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "rostered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_AUDIT_OBJECTIVE": lambda rng: {
        "table": f"audit_mandate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "audit_objective",
                "values": [rng.choice(["control_effectiveness_check", "filing_accuracy_check", "exception_trend_review"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_AUDIT_OBJECTIVE",
            },
            {
                "name": "audit_plan_ref",
                "values": [f"ap-{rng.randint(100, 999)}"],
            },
            {
                "name": "objective_set_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_CARE_QUALITY_ASSESSMENT": lambda rng: {
        "table": f"quality_assessment_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_quality_assessment",
                "values": [rng.choice(["outcome_benchmark_round", "intervention_adherence_round", "service_effectiveness_round"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_CARE_QUALITY_ASSESSMENT",
            },
            {
                "name": "benchmark_set_ref",
                "values": [f"bs-{rng.randint(100, 999)}"],
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_EDUCATION_HEALTH_CARE_PLAN": lambda rng: {
        "table": f"care_plan_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_plan_file",
                "values": [rng.choice(["annual_review_file", "accommodation_review_file", "transition_review_file"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_EDUCATION_HEALTH_CARE_PLAN",
            },
            {
                "name": "student_profile_ref",
                "values": [f"st-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_ETHICAL_REQUIREMENT": lambda rng: {
        "table": f"ethics_mandate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "requirement_clause",
                "values": [rng.choice(["consent_obtained", "data_minimized", "risk_disclosed"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_ETHICAL_REQUIREMENT",
            },
            {
                "name": "review_board_ref",
                "values": [f"irb-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_ETHICAL_STANDARD": lambda rng: {
        "table": f"ethics_norm_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "norm_revision",
                "values": [rng.choice(["v2_1", "v2_2", "v3_0"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_ETHICAL_STANDARD",
            },
            {
                "name": "scoped_to",
                "values": [rng.choice(["human_subjects", "field_research", "clinical_trials"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_EVACUATION_INSTRUCTION": lambda rng: {
        "table": f"shelter_order_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "instruction_zone",
                "values": [rng.choice(["coastal_zone", "floodway_zone", "wildfire_zone"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_EVACUATION_INSTRUCTION",
            },
            {
                "name": "hazard_event_ref",
                "values": [f"he-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_HEALTH_STANDARD_COMPLIANCE": lambda rng: {
        "table": f"compliance_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "standard_compliance",
                "values": [rng.choice(["infection_control", "sterile_handling", "radiation_safety"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_HEALTH_STANDARD_COMPLIANCE",
            },
            {
                "name": "certified_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
            {
                "name": "auditor_ref",
                "values": [f"au-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_HOUSING_POLICY_INFORMATION": lambda rng: {
        "table": f"policy_notice_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "housing_policy_information",
                "values": [rng.choice(["rent_stabilization", "veteran_exemption", "eviction_notice_filing"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_HOUSING_POLICY_INFORMATION",
            },
            {
                "name": "document_id",
                "values": [f"hp-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "issued_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_MACHINE_LEARNING_MODEL": lambda rng: {
        "table": f"model_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "learning_model",
                "values": [rng.choice(["gradient_boost_churn", "tree_forest_credit", "linear_demand_fit"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_MACHINE_LEARNING_MODEL",
            },
            {
                "name": "trained_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
            {
                "name": "validation_accuracy",
                "values": [f"{rng.randint(80, 99)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_MARKETING_PLAN": lambda rng: {
        "table": f"promo_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "marketing_plan",
                "values": [rng.choice(["autumn_launch_push", "holiday_promo_cycle", "regional_rollout_wave"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_MARKETING_PLAN",
            },
            {
                "name": "revenue_target_pct",
                "values": [str(rng.randint(5, 25))],
            },
            {
                "name": "launched_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_QUALITY_IMPROVEMENT_FRAMEWORK": lambda rng: {
        "table": f"quality_improvement_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "quality_improvement_framework",
                "values": [rng.choice(["pdca_v3", "lean_round_v2", "dmaic_v1"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_QUALITY_IMPROVEMENT_FRAMEWORK",
            },
            {
                "name": "initiative_ref",
                "values": [f"qi-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "adopted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_SERVICE_PERFORMANCE_METRIC": lambda rng: {
        "table": f"service_metric_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_performance_metric",
                "values": [rng.choice(["p95_latency_ms", "requests_per_min", "availability_pct"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_SERVICE_PERFORMANCE_METRIC",
            },
            {
                "name": "target_value",
                "values": [f"{rng.randint(10, 99)}.{rng.randint(0, 99):02d}"],
            },
            {
                "name": "reviewed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DIRECTIVE_SLAVIC_LINGUISTIC_ROOT": lambda rng: {
        "table": f"etymon_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "slavic_linguistic_root",
                "values": [rng.choice(["*grad-", "*vod-", "*morj-"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DIRECTIVE_SLAVIC_LINGUISTIC_ROOT",
            },
            {
                "name": "attested_language",
                "values": [rng.choice(["russian", "czech", "polish"])],
            },
            {
                "name": "documented_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DONATION_TRANSACTION_RECORD": lambda rng: {
        "table": f"gift_ledger_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "donation_transaction_record",
                "values": [f"dtr-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_DONATION_TRANSACTION_RECORD",
            },
            {
                "name": "donor_ref",
                "values": [f"dn-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "allocation_purpose",
                "values": [rng.choice(["emergency_fund", "research_grant", "capital_campaign"])],
            },
            {
                "name": "monetary_value",
                "values": [f"{rng.randint(25, 5000)}.{rng.randint(0, 99):02d}"],
            },
            {
                "name": "posted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DONOR_PROFILE": lambda rng: {
        "table": f"donor_contact_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "donor_profile",
                "values": [rng.choice(["contact_block", "reach_card", "identity_sheet"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DONOR_PROFILE",
            },
            {
                "name": "contact_channel",
                "values": [rng.choice(["email", "phone", "post"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_DUPLICATING_REQUEST": lambda rng: {
        "table": f"copy_request_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "duplicating_request",
                "values": [rng.choice(["certified_copy_order", "legal_copy_order", "file_copy_order"])],
                "iri": "https://signals.zndx.org/sdg#ICE_DUPLICATING_REQUEST",
            },
            {
                "name": "turnaround_class",
                "values": [rng.choice(["same_day", "two_days", "one_week"])],
            },
            {
                "name": "requested_document_type",
                "values": [rng.choice(["paper_record", "photo_record", "film_record"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_EDUCATIONAL_COURSE": lambda rng: {
        "table": f"course_bundle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "educational_course",
                "values": [rng.choice(["lecture_pack", "lab_module", "seminar_series"])],
                "iri": "https://signals.zndx.org/sdg#ICE_EDUCATIONAL_COURSE",
            },
            {
                "name": "assessment_standard_ref",
                "values": [f"std-{rng.randint(100, 999)}"],
            },
            {
                "name": "cataloged_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_ETYMOLOGICAL_DECIPHERING": lambda rng: {
        "table": f"etymon_link_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "etymological_deciphering",
                "values": [rng.choice(["cognate_mapping", "sound_correspondence", "borrowing_note"])],
                "iri": "https://signals.zndx.org/sdg#ICE_ETYMOLOGICAL_DECIPHERING",
            },
            {
                "name": "gaulish_tribe_ref",
                "values": [rng.choice(["belovaci", "harudesii", "ambiani"])],
            },
            {
                "name": "slavic_root_ref",
                "values": [rng.choice(["*gost-", "*dver-", "*volk-"])],
            },
            {
                "name": "confidence_score",
                "values": [f"{rng.randint(1, 9)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_FINANCIAL_REPORT": lambda rng: {
        "table": f"report_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "financial_report",
                "values": [rng.choice(["q1_income_statement", "annual_cash_flow", "year_end_balance_sheet"])],
                "iri": "https://signals.zndx.org/sdg#ICE_FINANCIAL_REPORT",
            },
            {
                "name": "reporting_currency",
                "values": [rng.choice(["usd", "eur", "gbp"])],
            },
            {
                "name": "filed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_FINANCIAL_REPORT_ARTIFACT": lambda rng: {
        "table": f"planning_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "financial_report_artifact",
                "values": [rng.choice(["status_snapshot", "strategy_appendix", "reconciliation_brief"])],
                "iri": "https://signals.zndx.org/sdg#ICE_FINANCIAL_REPORT_ARTIFACT",
            },
            {
                "name": "status_band",
                "values": [rng.choice(["healthy", "strained", "watchlist"])],
            },
            {
                "name": "strategy_ref",
                "values": [f"fs-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_FINE_ASSESSMENT_EVENT": lambda rng: {
        "table": f"penalty_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "fine_assessment_event",
                "values": [rng.choice(["parking_penalty", "code_violation_fine", "zoning_infraction_fine"])],
                "iri": "https://signals.zndx.org/sdg#ICE_FINE_ASSESSMENT_EVENT",
            },
            {
                "name": "property_owner_ref",
                "values": [f"po-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "violation_ref",
                "values": [f"pv-{rng.randint(100, 999)}"],
            },
            {
                "name": "assessment_amount",
                "values": [f"{rng.randint(25, 900)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_HEALTH_CARE_SUMMARY": lambda rng: {
        "table": f"patient_record_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "health_care_summary",
                "values": [rng.choice(["discharge_overview", "annual_review_brief", "episode_care_brief"])],
                "iri": "https://signals.zndx.org/sdg#ICE_HEALTH_CARE_SUMMARY",
            },
            {
                "name": "patient_ref",
                "values": [f"pt-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "summarized_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_HSSSEQ_ISSUE_DOCUMENT": lambda rng: {
        "table": f"hssseq_issue_document_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "issue_document",
                "values": [rng.choice(["situation_briefing", "regulatory_filing", "case_memorandum"])],
                "iri": "https://signals.zndx.org/sdg#ICE_HSSSEQ_ISSUE_DOCUMENT",
            },
            {
                "name": "described_situation_ref",
                "values": [f"hs-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "regulatory_context",
                "values": [rng.choice(["state_agency_rule", "federal_statute", "local_ordinance"])],
            },
            {
                "name": "issued_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_INFORMATION_ITEM": lambda rng: {
        "table": f"tagging_metadata_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "information_item",
                "values": [rng.choice(["tag_assignment", "tagger_profile", "tagged_resource_link"])],
                "iri": "https://signals.zndx.org/sdg#ICE_INFORMATION_ITEM",
            },
            {
                "name": "tagged_resource_ref",
                "values": [f"res-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "tagged_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_INSPECTION_REPORT": lambda rng: {
        "table": f"inspection_finding_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "inspection_report",
                "values": [rng.choice(["site_visit_finding", "compliance_check_finding", "equipment_walk_finding"])],
                "iri": "https://signals.zndx.org/sdg#ICE_INSPECTION_REPORT",
            },
            {
                "name": "documented_event_ref",
                "values": [f"ev-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "fielded_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_INTERNAL_AUDIT_STRATEGY": lambda rng: {
        "table": f"audit_strategy_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "audit_strategy",
                "values": [rng.choice(["risk_based_cycle", "compliance_focus_cycle", "controls_coverage_cycle"])],
                "iri": "https://signals.zndx.org/sdg#ICE_INTERNAL_AUDIT_STRATEGY",
            },
            {
                "name": "target_objective_ref",
                "values": [f"ao-{rng.randint(100, 999)}"],
            },
            {
                "name": "approved_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_MARITAL_STATUS_CERTIFICATION": lambda rng: {
        "table": f"marital_cert_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "marital_status_certification",
                "values": [f"msc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_MARITAL_STATUS_CERTIFICATION",
            },
            {
                "name": "certified_marital_state",
                "values": [rng.choice(["married", "separated", "divorced"])],
            },
            {
                "name": "spousal_consent_reason",
                "values": [rng.choice(["medical_care", "financial_agreement", "housing_arrangement"])],
            },
            {
                "name": "certified_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_MATCHED_SME_DATASET": lambda rng: {
        "table": f"sme_dataset_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "matched_sme_dataset",
                "values": [f"smed-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_MATCHED_SME_DATASET",
            },
            {
                "name": "survey_response_record_ref",
                "values": [f"srr-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "longitudinal_waves",
                "values": [str(rng.randint(2, 9))],
            },
            {
                "name": "collected_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_MEASUREMENT_CAPABILITY": lambda rng: {
        "table": f"capability_spec_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "measurement_capability",
                "values": [f"mc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_MEASUREMENT_CAPABILITY",
            },
            {
                "name": "measurement_parameter",
                "values": [rng.choice(["ambient_temperature", "axial_load", "surface_roughness"])],
            },
            {
                "name": "measurement_range",
                "values": [f"{rng.randint(0, 80)}-{rng.randint(81, 160)}"],
            },
            {
                "name": "expanded_uncertainty",
                "values": [f"0.{rng.randint(1, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_MEASUREMENT_CAPABILITY_PROFILE": lambda rng: {
        "table": f"capability_profile_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "measurement_capability_profile",
                "values": [f"mcp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_MEASUREMENT_CAPABILITY_PROFILE",
            },
            {
                "name": "measurement_parameter",
                "values": [rng.choice(["vibration_level", "flow_rate", "pressure_drop"])],
            },
            {
                "name": "measurement_range",
                "values": [f"{rng.randint(1, 50)}-{rng.randint(51, 120)}"],
            },
            {
                "name": "expanded_uncertainty_value",
                "values": [f"0.{rng.randint(1, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_MEMBERSHIP_TRANSFER_REQUEST": lambda rng: {
        "table": f"membership_transfer_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "membership_transfer_request",
                "values": [rng.choice(["benefit_port", "enrollment_move", "sponsor_handover"])],
                "iri": "https://signals.zndx.org/sdg#ICE_MEMBERSHIP_TRANSFER_REQUEST",
            },
            {
                "name": "submitting_dept_code",
                "values": [rng.choice(["ssd-north", "ssd-central", "ssd-harbor"])],
            },
            {
                "name": "effective_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_ORIGINAL_DOCUMENT": lambda rng: {
        "table": f"document_provenance_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "original_document",
                "values": [f"od-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_ORIGINAL_DOCUMENT",
            },
            {
                "name": "copied_document_ref",
                "values": [f"cd-{rng.randint(100, 999)}"],
            },
            {
                "name": "preserved_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PATIENT_EDUCATIONAL_CONTENT_PACKAGE": lambda rng: {
        "table": f"education_package_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "patient_education_package",
                "values": [rng.choice(["anticoagulant_factsheet_set", "insulin_start_set", "statin_adherence_set"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PATIENT_EDUCATIONAL_CONTENT_PACKAGE",
            },
            {
                "name": "sheet_count",
                "values": [str(rng.randint(2, 9))],
            },
            {
                "name": "released_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PATIENT_FEEDBACK_RECORD": lambda rng: {
        "table": f"feedback_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "patient_feedback_record",
                "values": [f"pfr-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_PATIENT_FEEDBACK_RECORD",
            },
            {
                "name": "care_quality_band",
                "values": [rng.choice(["above_target", "at_target", "below_target"])],
            },
            {
                "name": "captured_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PDF_DOCUMENT": lambda rng: {
        "table": f"document_file_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "pdf_document",
                "values": [f"pdf-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_PDF_DOCUMENT",
            },
            {
                "name": "stored_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PERSONALLY_IDENTIFIABLE_INFORMATION": lambda rng: {
        "table": f"data_element_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "personally_identifiable_information",
                "values": [rng.choice(["postal_code", "date_of_birth", "national_id_number"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PERSONALLY_IDENTIFIABLE_INFORMATION",
            },
            {
                "name": "collected_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PERSONAL_DATA_RECORD": lambda rng: {
        "table": f"record_file_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "personal_data_record",
                "values": [rng.choice(["voter_registration", "credit_application", "enrollment_form"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PERSONAL_DATA_RECORD",
            },
            {
                "name": "retained_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PERSONAL_HEALTH_PROFILE": lambda rng: {
        "table": f"health_profile_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "personal_health_profile",
                "values": [f"php-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_PERSONAL_HEALTH_PROFILE",
            },
            {
                "name": "genomic_dataset_ref",
                "values": [f"gd-{rng.randint(100, 999)}"],
            },
            {
                "name": "imaging_series_ref",
                "values": [f"im-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PERSONAL_HEALTH_RECORD": lambda rng: {
        "table": f"health_record_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "personal_health_record",
                "values": [rng.choice(["clinical_episode_record", "discharge_record", "follow_up_record"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PERSONAL_HEALTH_RECORD",
            },
            {
                "name": "stated_condition",
                "values": [rng.choice(["asthma", "hypertension", "type_2_diabetes"])],
            },
            {
                "name": "profile_ref",
                "values": [f"cp-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PREDICATE_CONCEPT": lambda rng: {
        "table": f"predicate_sheet_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "predicate_concept",
                "values": [rng.choice(["part_of", "located_in", "requires_approval"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PREDICATE_CONCEPT",
            },
            {
                "name": "knowledge_system_ref",
                "values": [f"kb-{rng.randint(100, 999)}"],
            },
            {
                "name": "defined_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PRODUCT_DISCLOSURE_STATEMENT": lambda rng: {
        "table": f"fund_disclosure_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "product_disclosure_statement",
                "values": [rng.choice(["initial_issuance", "quarterly_supplement", "material_change_notice"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PRODUCT_DISCLOSURE_STATEMENT",
            },
            {
                "name": "fund_ref",
                "values": [f"ff-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "released_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_PROPERTY_ASSESSMENT_NOTICE": lambda rng: {
        "table": f"assessment_notice_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "property_assessment_notice",
                "values": [rng.choice(["rate_review_notice", "parcel_reassessment_notice", "appeal_filing_notice"])],
                "iri": "https://signals.zndx.org/sdg#ICE_PROPERTY_ASSESSMENT_NOTICE",
            },
            {
                "name": "recommendation_ref",
                "values": [f"rec-{rng.randint(100, 999)}"],
            },
            {
                "name": "submitted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_QUALIFICATION_LEVEL": lambda rng: {
        "table": f"credential_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "qualification_level",
                "values": [rng.choice(["level_1", "level_2", "level_3"])],
                "iri": "https://signals.zndx.org/sdg#ICE_QUALIFICATION_LEVEL",
            },
            {
                "name": "issued_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_QUALITY_IMPROVEMENT_PRESENTATION": lambda rng: {
        "table": f"improvement_briefing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "quality_improvement_presentation",
                "values": [rng.choice(["ward_round_brief", "board_metrics_deck", "safety_audit_brief"])],
                "iri": "https://signals.zndx.org/sdg#ICE_QUALITY_IMPROVEMENT_PRESENTATION",
            },
            {
                "name": "evidence_basis_ref",
                "values": [f"ebm-{rng.randint(100, 999)}"],
            },
            {
                "name": "framework_ref",
                "values": [f"qif-{rng.randint(100, 999)}"],
            },
            {
                "name": "presented_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_RESPONSIBLE_INVESTMENT_FUND_COLLECTION": lambda rng: {
        "table": f"ethics_fund_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "responsible_investment_fund_collection",
                "values": [rng.choice(["annual_fund_catalog", "sustainable_book_review", "values_aligned_digest"])],
                "iri": "https://signals.zndx.org/sdg#ICE_RESPONSIBLE_INVESTMENT_FUND_COLLECTION",
            },
            {
                "name": "ethical_asset_ref",
                "values": [f"eia-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "published_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_ROLE_PERCEPTION": lambda rng: {
        "table": f"cognition_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "role_perception",
                "values": [rng.choice(["mentor_view", "guardian_view", "advisor_view"])],
                "iri": "https://signals.zndx.org/sdg#ICE_ROLE_PERCEPTION",
            },
            {
                "name": "role_concept_ref",
                "values": [f"rc-{rng.randint(100, 999)}"],
            },
            {
                "name": "recorded_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_SOCIAL_TAG": lambda rng: {
        "table": f"social_tag_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "social_tag",
                "values": [rng.choice(["#climatewatch", "#cityliving", "#gamedayrecap"])],
                "iri": "https://signals.zndx.org/sdg#ICE_SOCIAL_TAG",
            },
            {
                "name": "tagged_content_ref",
                "values": [f"post-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "captured_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_STAKEHOLDER_NEED": lambda rng: {
        "table": f"stakeholder_need_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "stakeholder_need",
                "values": [rng.choice(["data_privacy_concern", "uptime_complaint", "training_gap_report"])],
                "iri": "https://signals.zndx.org/sdg#ICE_STAKEHOLDER_NEED",
            },
            {
                "name": "stakeholder_role_ref",
                "values": [f"sr-{rng.randint(100, 999)}"],
            },
            {
                "name": "expressed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_SUPERVISED_TEST_UNIT_CATALOG": lambda rng: {
        "table": f"emission_test_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "supervised_test_unit_catalog",
                "values": [f"stuc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_SUPERVISED_TEST_UNIT_CATALOG",
            },
            {
                "name": "listed_emission_source_ref",
                "values": [f"ies-{rng.randint(100, 999)}"],
            },
            {
                "name": "unit_count",
                "values": [str(rng.randint(1, 12))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_SUPPLIER_LANDSCAPE_SURVEY": lambda rng: {
        "table": f"supplier_survey_batch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "supplier_landscape_survey",
                "values": [f"sfs-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_SUPPLIER_LANDSCAPE_SURVEY",
            },
            {
                "name": "assessed_supplier_ref",
                "values": [f"sup-{rng.randint(100, 999)}"],
            },
            {
                "name": "documented_market_trend",
                "values": [rng.choice(["capacity_expansion", "consolidation_wave", "price_softening"])],
            },
            {
                "name": "market_growth_index",
                "values": [f"{rng.randint(1, 9)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_SURVEY_FINDING": lambda rng: {
        "table": f"substance_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "survey_finding",
                "values": [f"sf-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_SURVEY_FINDING",
            },
            {
                "name": "substance_type",
                "values": [rng.choice(["stimulant_class", "sedative_class", "analgesic_class"])],
            },
            {
                "name": "prevalence_pct",
                "values": [f"{rng.randint(1, 49)}.{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_TELEMARKETING_KPI_CONSTRAINT": lambda rng: {
        "table": f"telemarketing_kpi_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "kpi_constraint",
                "values": [f"tkc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_TELEMARKETING_KPI_CONSTRAINT",
            },
            {
                "name": "applied_role",
                "values": [rng.choice(["sales_admin_queue", "dialer_desk_admin", "outbound_supervisor"])],
            },
            {
                "name": "call_volume_metric",
                "values": [rng.choice(["calls_per_hour", "attempts_per_hour", "talk_seconds_per_hour"])],
            },
            {
                "name": "minimum_calls_per_hour",
                "values": [f"{rng.randint(12, 55)}.{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ICE_WEEKLY_SALES_REPORT": lambda rng: {
        "table": f"telesales_weekly_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "weekly_sales_report",
                "values": [f"wsr-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#ICE_WEEKLY_SALES_REPORT",
            },
            {
                "name": "described_activity",
                "values": [rng.choice(["inbound_qualification", "outbound_dial_wave", "callback_round"])],
            },
            {
                "name": "reporting_period",
                "values": [f"202{rng.randint(0, 5)}-W{rng.randint(1, 52):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_CLINICAL_EFFICACY_MONITORING_SYSTEM": lambda rng: {
        "table": f"efficacy_monitor_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "efficacy_monitoring_system",
                "values": [f"cem-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_CLINICAL_EFFICACY_MONITORING_SYSTEM",
            },
            {
                "name": "realized_protocol",
                "values": [rng.choice(["adverse_event_watch", "endpoint_adjudication", "dosing_compliance_round"])],
            },
            {
                "name": "abstracted_metric",
                "values": [rng.choice(["response_rate_pct", "median_pfs_months", "grade3_adverse_event_rate"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_CONDIMENT_STATION": lambda rng: {
        "table": f"line_station_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "condiment_station",
                "values": [rng.choice(["cs-114", "cs-207", "cs-330"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_CONDIMENT_STATION",
            },
            {
                "name": "station_location",
                "values": [rng.choice(["entrance_north", "service_counter", "line_exit"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_EMERGENCY_SUPPLIES_KIT": lambda rng: {
        "table": f"supply_cache_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "emergency_supplies_kit",
                "values": [rng.choice(["es-201", "es-318", "es-405"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_EMERGENCY_SUPPLIES_KIT",
            },
            {
                "name": "disaster_supply_item",
                "values": [rng.choice(["water_case", "first_aid_case", "flashlight_set"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_ENERGY_EFFICIENT_APPLIANCE": lambda rng: {
        "table": f"appliance_fleet_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "energy_efficient_appliance",
                "values": [rng.choice(["ea-501", "ea-622", "ea-733"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_ENERGY_EFFICIENT_APPLIANCE",
            },
            {
                "name": "saving_component",
                "values": [rng.choice(["inverter_drive", "heat_pump_loop", "ec_fan"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_GEOGRAPHIC_REGION": lambda rng: {
        "table": f"planning_region_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "geographic_region",
                "values": [rng.choice(["rg-041", "rg-077", "rg-112"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_GEOGRAPHIC_REGION",
            },
            {
                "name": "boundary_basis",
                "values": [rng.choice(["political_border", "watershed_limit", "statistical_district"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_JOINT_INFORMATION_ENVIRONMENT": lambda rng: {
        "table": f"secure_infra_zone_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "joint_information_environment",
                "values": [f"jie-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_JOINT_INFORMATION_ENVIRONMENT",
            },
            {
                "name": "hosting_service",
                "values": [rng.choice(["air_gapped_network", "classified_datalink", "encrypted_backbone"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_MARITAL_STATE": lambda rng: {
        "table": f"civil_status_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "marital_state",
                "values": [rng.choice(["married", "civil_union", "single"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_MARITAL_STATE",
            },
            {
                "name": "recorded_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_MENU_COURSE": lambda rng: {
        "table": f"dining_menu_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "menu_course",
                "values": [rng.choice(["first_course", "main_course", "dessert_course"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_MENU_COURSE",
            },
            {
                "name": "dish_ref",
                "values": [f"dh-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_PICK_DROP_LOCATION": lambda rng: {
        "table": f"route_stop_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "pick_drop_location",
                "values": [rng.choice(["bay_1", "bay_2", "bay_3"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_PICK_DROP_LOCATION",
            },
            {
                "name": "route_ref",
                "values": [f"tr-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_RECOVERED_STATE": lambda rng: {
        "table": f"recovery_followup_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "recovered_state",
                "values": [rng.choice(["recovery_confirmed", "baseline_restored", "intake_stable"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_RECOVERED_STATE",
            },
            {
                "name": "followup_date",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_RENDERED_ANIMATION_ARTIFACT": lambda rng: {
        "table": f"render_output_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "rendered_animation_artifact",
                "values": [f"raa-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_RENDERED_ANIMATION_ARTIFACT",
            },
            {
                "name": "render_environment",
                "values": [rng.choice(["farm_node_3", "preview_theater", "rig_suite_b"])],
            },
            {
                "name": "animation_sequence_ref",
                "values": [f"asq-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_SECURITY_INFRASTRUCTURE": lambda rng: {
        "table": f"security_asset_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "security_infrastructure",
                "values": [rng.choice(["si-104", "si-217", "si-329"])],
                "iri": "https://signals.zndx.org/sdg#INDEPENDENT_CONTINUANT_SECURITY_INFRASTRUCTURE",
            },
            {
                "name": "access_control_part",
                "values": [rng.choice(["badge_reader", "keypad_lock", "turnstile_gate"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AUTHORIZED_ENTITY": lambda rng: {
        "table": f"signatory_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "authorized_entity",
                "values": [rng.choice(["signatory_officer", "resource_controller", "entity_representative"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AUTHORIZED_ENTITY",
            },
            {
                "name": "authorization_scope",
                "values": [rng.choice(["contract_signing", "resource_management", "official_representation"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AUTOMATED_SYSTEM": lambda rng: {
        "table": f"control_line_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "control_system",
                "values": [rng.choice(["plc_loop", "scada_node", "dcs_unit"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AUTOMATED_SYSTEM",
            },
            {
                "name": "commissioned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AVAILABLE_TOPPING": lambda rng: {
        "table": f"prep_line_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "topping_portion",
                "values": [rng.choice(["jalapeno_slice", "shredded_cheese", "sliced_avocado"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_AVAILABLE_TOPPING",
            },
            {
                "name": "staged_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_BOARD_OF_EQUALIZATION": lambda rng: {
        "table": f"appeal_panel_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "tax_appeal_board",
                "values": [rng.choice(["county_boe_north", "county_boe_south", "district_boe_central"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_BOARD_OF_EQUALIZATION",
            },
            {
                "name": "convened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_BUSINESS_ENTITY": lambda rng: {
        "table": f"vendor_master_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "business_entity",
                "values": [f"co-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_BUSINESS_ENTITY",
            },
            {
                "name": "trading_started_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CANCER_PATIENT": lambda rng: {
        "table": f"oncology_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "patient",
                "values": [f"pt-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CANCER_PATIENT",
            },
            {
                "name": "treatment_phase",
                "values": [rng.choice(["induction", "consolidation", "maintenance"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CERTIFIED_INTERNAL_AUDITOR": lambda rng: {
        "table": f"audit_staff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "auditor",
                "values": [f"au-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CERTIFIED_INTERNAL_AUDITOR",
            },
            {
                "name": "credential_number",
                "values": [f"ci-{rng.randint(10000, 99999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHEMICAL_INSTALLATION": lambda rng: {
        "table": f"plant_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "installation",
                "values": [f"chem-plant-{rng.randint(1, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHEMICAL_INSTALLATION",
            },
            {
                "name": "process_line",
                "values": [rng.choice(["solvent_recovering", "polymer_synthesis", "nitration_unit"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHILD": lambda rng: {
        "table": f"pediatric_census_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "child",
                "values": [f"ch-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHILD",
            },
            {
                "name": "age_band",
                "values": [rng.choice(["0_2", "3_5", "6_11"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHILD_PROTECTION_TEAM": lambda rng: {
        "table": f"team_caseload_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "protection_team",
                "values": [f"cpt-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CHILD_PROTECTION_TEAM",
            },
            {
                "name": "service_region",
                "values": [rng.choice(["metro_north", "metro_south", "rural_county"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLIENT": lambda rng: {
        "table": f"advice_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advised_client",
                "values": [f"cl-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLIENT",
            },
            {
                "name": "advice_product",
                "values": [rng.choice(["retirement_fund", "bond_ladder", "index_portfolio"])],
            },
            {
                "name": "session_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLINICAL_SYSTEM": lambda rng: {
        "table": f"system_uptime_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "clinical_system",
                "values": [rng.choice(["emr_orion", "lis_north", "pacs_core"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLINICAL_SYSTEM",
            },
            {
                "name": "availability_pct",
                "values": [f"{rng.randint(97, 99)}.{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLOUD_STORAGE_VOLUME": lambda rng: {
        "table": f"volume_provision_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "storage_volume",
                "values": [f"vol-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CLOUD_STORAGE_VOLUME",
            },
            {
                "name": "capacity_gb",
                "values": [str(rng.randint(500, 4000))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COLLEGE_ADMINISTRATION": lambda rng: {
        "table": f"campus_office_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "administration_unit",
                "values": [rng.choice(["registrar_office", "provosts_office", "office_of_facilities"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COLLEGE_ADMINISTRATION",
            },
            {
                "name": "budget_year",
                "values": [f"20{rng.randint(10, 25)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COLLEGE_COMMUNITY_MEMBER": lambda rng: {
        "table": f"campus_directory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "community_member",
                "values": [f"cm-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COLLEGE_COMMUNITY_MEMBER",
            },
            {
                "name": "member_class",
                "values": [rng.choice(["faculty", "staff", "student"])],
            },
            {
                "name": "affiliated_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COMMUNITY_HEALTHCARE_SETTING": lambda rng: {
        "table": f"local_care_site_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_setting",
                "values": [rng.choice(["clinic-northside", "clinic-hillcrest", "clinic-riverside"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COMMUNITY_HEALTHCARE_SETTING",
            },
            {
                "name": "service_line",
                "values": [rng.choice(["primary_care", "pediatrics", "dental"])],
            },
            {
                "name": "opened_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CONTINUOUS_EMISSIONS_MONITORING_SYSTEM": lambda rng: {
        "table": f"stack_monitor_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "monitoring_system",
                "values": [f"cem-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CONTINUOUS_EMISSIONS_MONITORING_SYSTEM",
            },
            {
                "name": "pollutant",
                "values": [rng.choice(["so2", "nox", "co"])],
            },
            {
                "name": "sample_interval",
                "values": [rng.choice(["5m", "15m", "1h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COPIED_DOCUMENT": lambda rng: {
        "table": f"replica_track_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "replica_unit",
                "values": [f"rep-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_COPIED_DOCUMENT",
            },
            {
                "name": "copy_medium",
                "values": [rng.choice(["photocopied", "digital_reproduction", "facsimile"])],
            },
            {
                "name": "source_record_ref",
                "values": [f"src-{rng.randint(100, 999)}"],
            },
            {
                "name": "duplicated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_AGENT": lambda rng: {
        "table": f"entity_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "trading_entity",
                "values": [f"ent-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_AGENT",
            },
            {
                "name": "entity_form",
                "values": [rng.choice(["corporation", "limited_liability_company", "partnership"])],
            },
            {
                "name": "registry_state",
                "values": [rng.choice(["de", "tx", "wa"])],
            },
            {
                "name": "registered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_COALITION": lambda rng: {
        "table": f"alliance_charter_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "partner_group",
                "values": [f"cg-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_COALITION",
            },
            {
                "name": "charter_scope",
                "values": [rng.choice(["shared_procurement", "joint_ventures", "standards_alignment"])],
            },
            {
                "name": "member_count",
                "values": [str(rng.randint(3, 12))],
            },
            {
                "name": "chartered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_EMPLOYEE": lambda rng: {
        "table": f"roster_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "member",
                "values": [f"emp-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CORPORATE_EMPLOYEE",
            },
            {
                "name": "employment_status",
                "values": [rng.choice(["full_time", "part_time", "temporary"])],
            },
            {
                "name": "reporting_unit",
                "values": [rng.choice(["operations", "sales", "engineering"])],
            },
            {
                "name": "hired_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CUSTOMER_DEVICE": lambda rng: {
        "table": f"engagement_device_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "customer_device",
                "values": [rng.choice(["mobile_app", "web_browser", "tablet_kiosk"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CUSTOMER_DEVICE",
            },
            {
                "name": "first_seen_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CUSTOMER_NETWORK": lambda rng: {
        "table": f"engagement_network_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "customer_network",
                "values": [rng.choice(["loyalty_app", "service_portal", "web_portal"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_CUSTOMER_NETWORK",
            },
            {
                "name": "provisioned_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DATABASE_CONTAINER": lambda rng: {
        "table": f"container_placement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "storage_container",
                "values": [f"ctn-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DATABASE_CONTAINER",
            },
            {
                "name": "host_volume",
                "values": [f"vol-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "provisioned_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DEPARTING_EMPLOYEE": lambda rng: {
        "table": f"offboarding_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "departing_employee",
                "values": [f"emp-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DEPARTING_EMPLOYEE",
            },
            {
                "name": "role_held",
                "values": [rng.choice(["engineer", "analyst", "manager"])],
            },
            {
                "name": "last_working_day",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DISH": lambda rng: {
        "table": f"service_portion_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "served_portion",
                "values": [rng.choice(["roast_squash_cup", "braised_beet_cup", "grilled_pepper_crown"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DISH",
            },
            {
                "name": "receiver_ref",
                "values": [f"dv-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DONOR_AGENT": lambda rng: {
        "table": f"donation_event_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "donor_agent",
                "values": [f"da-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_DONOR_AGENT",
            },
            {
                "name": "donation_kind",
                "values": [rng.choice(["blood_draw", "organ_transfer", "cash_gift"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EARTH_OBSERVATION_DATA": lambda rng: {
        "table": f"orbital_pass_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "observation_dataset",
                "values": [f"eo-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EARTH_OBSERVATION_DATA",
            },
            {
                "name": "collecting_satellite",
                "values": [f"orb-{rng.randint(1, 40):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ENERGY_SAVING_COMPONENT": lambda rng: {
        "table": f"heating_line_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "saving_component",
                "values": [rng.choice(["variable_speed_drive", "thermal_regulator", "low_loss_manifold"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ENERGY_SAVING_COMPONENT",
            },
            {
                "name": "installed_in",
                "values": [rng.choice(["boiler_loop", "air_handler", "heat_exchanger_bank"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ETHICAL_INVESTMENT_ASSET": lambda rng: {
        "table": f"stewardship_book_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "asset_position",
                "values": [rng.choice(["green_bond_position", "renewable_equity_position", "clean_energy_debt_position"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ETHICAL_INVESTMENT_ASSET",
            },
            {
                "name": "screening_tier",
                "values": [rng.choice(["impact_first", "exclusion_based", "values_aligned"])],
            },
            {
                "name": "position_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXECUTIVE_TEAM": lambda rng: {
        "table": f"management_forum_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "leadership_group",
                "values": [rng.choice(["c_suite_forum", "strategic_review_body", "board_executives_panel"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXECUTIVE_TEAM",
            },
            {
                "name": "quorum_size",
                "values": [str(rng.randint(4, 12))],
            },
            {
                "name": "term_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXECUTIVE_TEAM_MEMBER": lambda rng: {
        "table": f"senior_appointment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "senior_leader",
                "values": [f"etm-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXECUTIVE_TEAM_MEMBER",
            },
            {
                "name": "seat_portfolio",
                "values": [rng.choice(["revenue_operations", "capital_planning", "customer_growth"])],
            },
            {
                "name": "seated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPEDITION_EQUIPMENT": lambda rng: {
        "table": f"polar_cache_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "expedition_item",
                "values": [f"exp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPEDITION_EQUIPMENT",
            },
            {
                "name": "deploy_class",
                "values": [rng.choice(["cold_weather_transport", "field_science_rig", "basecamp_support"])],
            },
            {
                "name": "deployed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPEDITION_PARTICIPANT": lambda rng: {
        "table": f"crew_manifest_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "expedition_participant",
                "values": [f"crew-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPEDITION_PARTICIPANT",
            },
            {
                "name": "duty_station",
                "values": [rng.choice(["ice_core_drilling", "glacier_survey", "weather_watch"])],
            },
            {
                "name": "joined_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPORTED_DATA_ARTIFACT": lambda rng: {
        "table": f"export_artifact_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "exported_data_artifact",
                "values": [f"eda-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXPORTED_DATA_ARTIFACT",
            },
            {
                "name": "export_medium",
                "values": [rng.choice(["digital_file", "printout", "tape_backup"])],
            },
            {
                "name": "exported_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXTERNAL_AGENCY": lambda rng: {
        "table": f"partner_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "external_agency",
                "values": [f"agc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_EXTERNAL_AGENCY",
            },
            {
                "name": "partnership_scope",
                "values": [rng.choice(["reseller", "consultancy", "logistics_provider"])],
            },
            {
                "name": "agreement_started_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FACILITY_INFRASTRUCTURE": lambda rng: {
        "table": f"site_infrastructure_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "facility_infrastructure",
                "values": [rng.choice(["service_road", "loading_dock", "pedestrian_walkway"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FACILITY_INFRASTRUCTURE",
            },
            {
                "name": "structure_ref",
                "values": [f"inf-{rng.randint(100, 999)}"],
            },
            {
                "name": "maintained_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FINANCIAL_FUND": lambda rng: {
        "table": f"fund_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "fund_identifier",
                "values": [f"fd-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FINANCIAL_FUND",
            },
            {
                "name": "mandate_scope",
                "values": [rng.choice(["capital_reserve", "pension_pool", "grant_disbursement"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FINANCIAL_PRODUCT": lambda rng: {
        "table": f"product_catalog_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "product_code",
                "values": [f"fp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_FINANCIAL_PRODUCT",
            },
            {
                "name": "trading_venue",
                "values": [rng.choice(["primary_exchange", "otc_desk", "clearing_house"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_AGENT": lambda rng: {
        "table": f"care_provider_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_agent_id",
                "values": [f"hg-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_AGENT",
            },
            {
                "name": "care_domain",
                "values": [rng.choice(["outpatient_clinic", "inpatient_ward", "field_response_unit"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_NETWORK_INFRASTRUCTURE": lambda rng: {
        "table": f"network_inventory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_segment",
                "values": [rng.choice(["emr_backbone", "imaging_link", "telehealth_uplink"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_NETWORK_INFRASTRUCTURE",
            },
            {
                "name": "bandwidth_class",
                "values": [rng.choice(["1gbps", "10gbps", "25gbps"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_ORGANIZATION": lambda rng: {
        "table": f"provider_directory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "provider_org",
                "values": [rng.choice(["city_general_system", "rural_health_cooperative", "childrens_care_group"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HEALTHCARE_ORGANIZATION",
            },
            {
                "name": "service_region",
                "values": [rng.choice(["metro_east", "metro_west", "county_line"])],
            },
            {
                "name": "accredited_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HIGHER_EDUCATION_INSTITUTION": lambda rng: {
        "table": f"campus_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "degree_granting_institution",
                "values": [rng.choice(["northgate_university", "coastal_state_college", "harbor_polytechnic"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HIGHER_EDUCATION_INSTITUTION",
            },
            {
                "name": "accreditation_cycle",
                "values": [f"20{rng.randint(1, 6)}"],
            },
            {
                "name": "chartered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HOUSING_CLIENT": lambda rng: {
        "table": f"lease_file_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "tenant_of_record",
                "values": [f"tc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HOUSING_CLIENT",
            },
            {
                "name": "unit_type",
                "values": [rng.choice(["studio", "one_bedroom", "two_bedroom"])],
            },
            {
                "name": "lease_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HUMAN_AGENT": lambda rng: {
        "table": f"field_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "assigned_actor",
                "values": [f"ha-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_HUMAN_AGENT",
            },
            {
                "name": "duty_station",
                "values": [rng.choice(["permit_counter", "intake_desk", "field_unit_2"])],
            },
            {
                "name": "appointed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_IMAGING_EQUIPMENT": lambda rng: {
        "table": f"imaging_fleet_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "imaging_equipment",
                "values": [rng.choice(["mri_scanner_3t", "ct_tomograph_128", "ultrasound_unit_p7"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_IMAGING_EQUIPMENT",
            },
            {
                "name": "installed_site",
                "values": [rng.choice(["radiology_wing_a", "mobile_unit_2", "outpatient_suite_b"])],
            },
            {
                "name": "commissioned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDEPENDENT_ECONOMIC_AGENT": lambda rng: {
        "table": f"trading_counterparty_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "independent_economic_agent",
                "values": [rng.choice(["harbor_freight_traders", "valley_millworks", "delta_energy_broker"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDEPENDENT_ECONOMIC_AGENT",
            },
            {
                "name": "exchange_mode",
                "values": [rng.choice(["open_tender", "bilateral_contract", "framework_agreement"])],
            },
            {
                "name": "agreement_started_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDIVIDUAL": lambda rng: {
        "table": f"study_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "individual",
                "values": [f"ind-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDIVIDUAL",
            },
            {
                "name": "consent_status",
                "values": [rng.choice(["signed", "renewed", "declined"])],
            },
            {
                "name": "enrolled_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDUSTRIAL_EMISSION_SOURCE": lambda rng: {
        "table": f"stack_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "industrial_emission_source",
                "values": [rng.choice(["boiler_flue_01", "kiln_exhaust_pipe", "smelter_stack"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INDUSTRIAL_EMISSION_SOURCE",
            },
            {
                "name": "pollutant_stream",
                "values": [rng.choice(["so2", "nox", "particulates"])],
            },
            {
                "name": "registered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INFORMATION_SYSTEM": lambda rng: {
        "table": f"records_platform_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "documentation_system",
                "values": [rng.choice(["chart_store", "imaging_vault", "lab_results_bank"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INFORMATION_SYSTEM",
            },
            {
                "name": "deployed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INSTITUTIONAL_INVESTOR": lambda rng: {
        "table": f"funding_commitment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "fund_house",
                "values": [rng.choice(["endowment_desk", "insurer_block", "pension_board_desk"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INSTITUTIONAL_INVESTOR",
            },
            {
                "name": "commitment_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INTELLIGENCE_OUTPUT": lambda rng: {
        "table": f"analytics_yield_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "artifact_form",
                "values": [rng.choice(["capacity_brief", "utilization_map", "outlier_flag"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INTELLIGENCE_OUTPUT",
            },
            {
                "name": "generated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INTERNAL_AUDITING_DIVISION": lambda rng: {
        "table": f"audit_unit_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "division_unit",
                "values": [rng.choice(["controls_group", "financials_group", "compliance_group"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_INTERNAL_AUDITING_DIVISION",
            },
            {
                "name": "staffed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_KEYFRAME_TIMELINE": lambda rng: {
        "table": f"animation_track_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "timeline_track",
                "values": [rng.choice(["walk_cycle_track", "camera_move_track", "arm_pose_track"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_KEYFRAME_TIMELINE",
            },
            {
                "name": "frame_rate",
                "values": [rng.choice(["24fps", "30fps", "60fps"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LATERAL_RESTRAINT_STRAP": lambda rng: {
        "table": f"prosthetic_fitting_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "strap_unit",
                "values": [rng.choice(["lrs-104", "lrs-227", "lrs-339"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LATERAL_RESTRAINT_STRAP",
            },
            {
                "name": "anchor_point",
                "values": [rng.choice(["medial_post", "patellar_bar", "ischial_ring"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LAW_ENFORCEMENT_AGENCY": lambda rng: {
        "table": f"agency_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "enforcement_agency",
                "values": [rng.choice(["lea-041", "lea-112", "lea-233"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LAW_ENFORCEMENT_AGENCY",
            },
            {
                "name": "jurisdiction_type",
                "values": [rng.choice(["municipal", "county", "transit"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LECTURE_NOTES": lambda rng: {
        "table": f"transcript_copy_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "notes_copy",
                "values": [rng.choice(["ln-0412", "ln-1187", "ln-2364"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LECTURE_NOTES",
            },
            {
                "name": "medium_kind",
                "values": [rng.choice(["paper_sheet", "microfilm", "audio_cassette"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGAL_ENTITY_OR_PERSON": lambda rng: {
        "table": f"party_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "legal_entity_or_person",
                "values": [rng.choice(["incorporated_company", "sole_trustee", "municipal_office"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGAL_ENTITY_OR_PERSON",
            },
            {
                "name": "jurisdiction_ref",
                "values": [rng.choice(["district_court", "state_registry", "federal_clerk"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGAL_TECH_SOLUTION_LAB": lambda rng: {
        "table": f"lab_facility_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "legal_tech_solution_lab",
                "values": [rng.choice(["proof_of_concept_lab", "sandbox_testing_lab", "field_trial_lab"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGAL_TECH_SOLUTION_LAB",
            },
            {
                "name": "trial_cohort",
                "values": [rng.choice(["in_house_counsel", "bar_association", "law_firm_panel"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGISLATIVE_DEPARTMENT": lambda rng: {
        "table": f"legislative_unit_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "legislative_department",
                "values": [rng.choice(["bill_drafting_unit", "committee_clerk_unit", "enrollment_unit"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_LEGISLATIVE_DEPARTMENT",
            },
            {
                "name": "session_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MAIL_DROP_LOCATION": lambda rng: {
        "table": f"mail_point_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mail_drop_location",
                "values": [rng.choice(["street_box", "cabinet_bay", "counter_slot"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MAIL_DROP_LOCATION",
            },
            {
                "name": "service_point_ref",
                "values": [f"sp-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MALNOURISHED_CHILD": lambda rng: {
        "table": f"wasting_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "child",
                "values": [f"ch-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MALNOURISHED_CHILD",
            },
            {
                "name": "deficiency_sign",
                "values": [rng.choice(["stunting", "wasting", "edema"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MINOR_PERSON": lambda rng: {
        "table": f"juvenile_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "minor",
                "values": [f"mn-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MINOR_PERSON",
            },
            {
                "name": "age_band",
                "values": [rng.choice(["0_11", "12_15", "16_17"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MUNICIPAL_POOL_FACILITY": lambda rng: {
        "table": f"public_aquatic_site_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "pool_facility",
                "values": [rng.choice(["rp-harborview", "rp-millbrook", "rp-fairmount"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_MUNICIPAL_POOL_FACILITY",
            },
            {
                "name": "site_type",
                "values": [rng.choice(["city_park", "recreation_center", "community_grounds"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_INFRASTRUCTURE": lambda rng: {
        "table": f"backbone_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_infrastructure",
                "values": [rng.choice(["metro_fiber_ring", "campus_ethernet_grid", "dark_fiber_bundle"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_INFRASTRUCTURE",
            },
            {
                "name": "bandwidth_class",
                "values": [rng.choice(["10gbps", "40gbps", "100gbps"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_PROVIDER": lambda rng: {
        "table": f"carrier_agreement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_provider",
                "values": [rng.choice(["metro_carrier", "regional_carrier", "backbone_carrier"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_PROVIDER",
            },
            {
                "name": "service_offering",
                "values": [rng.choice(["bandwidth", "routing", "transit_capacity"])],
            },
            {
                "name": "contract_started_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_RESOURCE": lambda rng: {
        "table": f"resource_allocation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_resource",
                "values": [rng.choice(["bandwidth_pool", "server_cluster", "storage_volume"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_RESOURCE",
            },
            {
                "name": "allocation_site",
                "values": [rng.choice(["enterprise_dc", "cloud_region", "edge_pop"])],
            },
            {
                "name": "allocated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_SCRUBBING_CENTER": lambda rng: {
        "table": f"traffic_cleaning_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_scrubbing_center",
                "values": [rng.choice(["scrub-041", "scrub-112", "scrub-233"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_SCRUBBING_CENTER",
            },
            {
                "name": "cleaning_function",
                "values": [rng.choice(["ddos_filtering", "malware_inspection", "traffic_shaping"])],
            },
            {
                "name": "commissioned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_USER": lambda rng: {
        "table": f"user_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "network_user",
                "values": [f"usr-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NETWORK_USER",
            },
            {
                "name": "usage_class",
                "values": [rng.choice(["residential", "business", "institutional"])],
            },
            {
                "name": "first_active_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NONPROFIT_ORGANIZATION": lambda rng: {
        "table": f"charity_registry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "nonprofit_organization",
                "values": [f"npo-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NONPROFIT_ORGANIZATION",
            },
            {
                "name": "mission_area",
                "values": [rng.choice(["food_security", "health_access", "education_outreach"])],
            },
            {
                "name": "incorporated_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NON_DIESEL_POWERED_VEHICLE": lambda rng: {
        "table": f"fleet_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "non_diesel_powered_vehicle",
                "values": [f"ndv-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_NON_DIESEL_POWERED_VEHICLE",
            },
            {
                "name": "power_source",
                "values": [rng.choice(["fuel_cell", "battery_electric", "plug_in_hybrid"])],
            },
            {
                "name": "first_registered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OBSERVATORY_MOUNTING_STRUCTURE": lambda rng: {
        "table": f"mount_assembly_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "observatory_mounting_structure",
                "values": [f"mnt-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OBSERVATORY_MOUNTING_STRUCTURE",
            },
            {
                "name": "supported_instrument",
                "values": [rng.choice(["wide_field_camera", "spectrograph", "adaptive_optics_module"])],
            },
            {
                "name": "commissioned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OPERATIONAL_EQUIPMENT": lambda rng: {
        "table": f"field_deployment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "operational_equipment",
                "values": [f"oe-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OPERATIONAL_EQUIPMENT",
            },
            {
                "name": "deployment_zone",
                "values": [rng.choice(["forward_staging", "survey_perimeter", "base_operations"])],
            },
            {
                "name": "deployed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OPERATIONAL_UNIT": lambda rng: {
        "table": f"patrol_post_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "precinct",
                "values": [rng.choice(["precinct_north", "precinct_south", "precinct_central"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OPERATIONAL_UNIT",
            },
            {
                "name": "opened_on",
                "values": [f"20{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ORGANIZATION": lambda rng: {
        "table": f"org_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "organization",
                "values": [f"orgn-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ORGANIZATION",
            },
            {
                "name": "established_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ORGANIZATIONAL_AGENT": lambda rng: {
        "table": f"planning_duty_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "organizational_agent",
                "values": [f"oa-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_ORGANIZATIONAL_AGENT",
            },
            {
                "name": "planning_cycle",
                "values": [rng.choice(["annual", "quarterly", "monthly"])],
            },
            {
                "name": "seated_on",
                "values": [f"20{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OZONE_OXYGEN_MIXTURE": lambda rng: {
        "table": f"prep_batch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mixture_batch",
                "values": [f"oz-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_OZONE_OXYGEN_MIXTURE",
            },
            {
                "name": "concentration_band",
                "values": [rng.choice(["low", "standard", "high"])],
            },
            {
                "name": "prepared_on",
                "values": [f"20{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PATIENT": lambda rng: {
        "table": f"care_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "patient",
                "values": [f"pt-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PATIENT",
            },
            {
                "name": "care_status",
                "values": [rng.choice(["inpatient", "outpatient", "follow_up"])],
            },
            {
                "name": "care_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_POPULATION": lambda rng: {
        "table": f"prevalence_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "population",
                "values": [rng.choice(["metro_adults", "rural_seniors", "urban_adolescents"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_POPULATION",
            },
            {
                "name": "prevalence_rate",
                "values": [f"{rng.randint(1, 60)}.{rng.randint(0, 99):02d}"],
            },
            {
                "name": "surveyed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRECLINICAL_IMAGING_CORE": lambda rng: {
        "table": f"imaging_core_bench_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "imaging_core",
                "values": [rng.choice(["anatomy_bench", "oncology_bench", "cardiac_bench"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRECLINICAL_IMAGING_CORE",
            },
            {
                "name": "mounted_equipment",
                "values": [rng.choice(["micro_ct_scanner", "in_vivo_mri_7t", "two_photon_imager"])],
            },
            {
                "name": "commissioned_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRIM_CHEMICAL": lambda rng: {
        "table": f"substance_lot_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "chemical",
                "values": [f"chm-{rng.randint(100000, 999999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRIM_CHEMICAL",
            },
            {
                "name": "composition_form",
                "values": [rng.choice(["single_substance", "aqueous_mixture", "saline_buffer_mixture"])],
            },
            {
                "name": "received_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRIM_SCIENTIFICORGANIZATION": lambda rng: {
        "table": f"research_org_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "scientific_organization",
                "values": [f"soc-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PRIM_SCIENTIFICORGANIZATION",
            },
            {
                "name": "research_mission",
                "values": [rng.choice(["applied_r_and_d", "basic_science", "joint_dev_program"])],
            },
            {
                "name": "chartered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PUBLIC_INFORMATION_OFFICER": lambda rng: {
        "table": f"media_desk_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "public_information_officer",
                "values": [f"pio-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_PUBLIC_INFORMATION_OFFICER",
            },
            {
                "name": "briefing_duty",
                "values": [rng.choice(["press_briefing", "release_issuance", "media_inquiries"])],
            },
            {
                "name": "appointed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_QUANTITATIVE_ANALYTIC_TOOL": lambda rng: {
        "table": f"analysis_instrument_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "quantitative_analytic_tool",
                "values": [f"qat-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_QUANTITATIVE_ANALYTIC_TOOL",
            },
            {
                "name": "instrument_kind",
                "values": [rng.choice(["regression_suite", "dose_response_model", "cohort_stats_engine"])],
            },
            {
                "name": "validated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_RECEIVING_FACILITY_STAFF": lambda rng: {
        "table": f"notification_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "receiving_facility_staff",
                "values": [f"rfs-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_RECEIVING_FACILITY_STAFF",
            },
            {
                "name": "intake_duty",
                "values": [rng.choice(["case_logging", "report_triage", "follow_up_coordination"])],
            },
            {
                "name": "unit_code",
                "values": [rng.choice(["ed_triage", "pharm_d", "lab_outreach"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REGULATORY_BODY": lambda rng: {
        "table": f"regulator_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "regulatory_body",
                "values": [rng.choice(["market_conduct_board", "safety_standards_board", "compliance_oversight_board"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REGULATORY_BODY",
            },
            {
                "name": "chartered_on",
                "values": [f"20{rng.randint(0, 9)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_RENDERING_ENGINE": lambda rng: {
        "table": f"render_node_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "rendering_engine",
                "values": [rng.choice(["frame_forge_node", "path_tracer_bay", "gpu_render_cluster"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_RENDERING_ENGINE",
            },
            {
                "name": "frames_completed",
                "values": [str(rng.randint(100, 9999))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REQUESTING_CLIENT": lambda rng: {
        "table": f"provider_change_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "requesting_client",
                "values": [rng.choice(["hospital_network_a", "payer_insurer_riv", "plan_admin_dept"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REQUESTING_CLIENT",
            },
            {
                "name": "request_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REQUESTING_ENTITY": lambda rng: {
        "table": f"copy_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "requesting_entity",
                "values": [rng.choice(["records_department", "county_clerk_office", "law_firm_counsel"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_REQUESTING_ENTITY",
            },
            {
                "name": "submitted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SCIENTIFIC_ORGANIZATION": lambda rng: {
        "table": f"journal_program_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "journal_publisher",
                "values": [rng.choice(["applied_physics_press", "organic_chem_forum", "genomics_society"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SCIENTIFIC_ORGANIZATION",
            },
            {
                "name": "meeting_cycle",
                "values": [rng.choice(["annual_symposium", "biannual_forum", "quarterly_workshop"])],
            },
            {
                "name": "first_issue_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SERVICE_PROVIDER": lambda rng: {
        "table": f"service_function_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "provider_entity",
                "values": [rng.choice(["transit_operations_co", "utilities_dispatch_unit", "records_maintenance_shop"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SERVICE_PROVIDER",
            },
            {
                "name": "service_function",
                "values": [rng.choice(["route_dispatch", "meter_reading", "document_archiving"])],
            },
            {
                "name": "contracted_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SOCIAL_SERVICES_DEPARTMENT": lambda rng: {
        "table": f"welfare_program_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "administering_department",
                "values": [rng.choice(["county_assistance_office", "state_welfare_bureau", "city_family_services_unit"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_SOCIAL_SERVICES_DEPARTMENT",
            },
            {
                "name": "program_area",
                "values": [rng.choice(["food_assistance", "housing_support", "community_care"])],
            },
            {
                "name": "program_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_TARGET_MATERIAL": lambda rng: {
        "table": f"directed_energy_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "treated_portion",
                "values": [rng.choice(["tumor_tissue_sector", "alloy_weld_joint", "implant_surface_layer"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_TARGET_MATERIAL",
            },
            {
                "name": "delivery_mode",
                "values": [rng.choice(["pulsed_laser", "ion_burst", "ultrasound_pulse"])],
            },
            {
                "name": "run_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_TELESCOPE_INSTRUMENT_ARTIFACT": lambda rng: {
        "table": f"optical_instrument_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "instrument_class",
                "values": [rng.choice(["reflector_bay", "detector_cradle", "optical_bench_unit"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_TELESCOPE_INSTRUMENT_ARTIFACT",
            },
            {
                "name": "calibrated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#MATERIAL_ENTITY_THREAT_INFORMATION": lambda rng: {
        "table": f"threat_intel_brief_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "threat_record",
                "values": [rng.choice(["actor_profile", "attack_vector_note", "exposure_advisory"])],
                "iri": "https://signals.zndx.org/sdg#MATERIAL_ENTITY_THREAT_INFORMATION",
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ACADEMIC_CURRICULUM_PROGRAM": lambda rng: {
        "table": f"curriculum_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "program_track",
                "values": [rng.choice(["cross_disciplinary_core", "joint_minor_track", "capstone_integration"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ACADEMIC_CURRICULUM_PROGRAM",
            },
            {
                "name": "cohort_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ADULT_ADOPTION_PROCESS": lambda rng: {
        "table": f"adoption_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "adoption_stage",
                "values": [rng.choice(["home_study", "finalization_hearing", "post_placement_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ADULT_ADOPTION_PROCESS",
            },
            {
                "name": "petitioned_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ANCESTOR_VENERATION_CEREMONY": lambda rng: {
        "table": f"ceremony_ledger_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "ceremony_phase",
                "values": [rng.choice(["offering_placement", "ancestral_invocation", "communal_feast"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ANCESTOR_VENERATION_CEREMONY",
            },
            {
                "name": "ritual_held_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ANIMATION_SEQUENCE_CREATION": lambda rng: {
        "table": f"sequence_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "sequence_phase",
                "values": [rng.choice(["keyframe_layout", "inbetween_pass", "render_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ANIMATION_SEQUENCE_CREATION",
            },
            {
                "name": "sequence_ref",
                "values": [f"seq-{rng.randint(100, 999)}"],
            },
            {
                "name": "session_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_APPROVAL_RECOMMENDATION": lambda rng: {
        "table": f"advisory_file_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "recommendation_action",
                "values": [rng.choice(["value_assessment", "condition_review", "compliance_check"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_APPROVAL_RECOMMENDATION",
            },
            {
                "name": "advisory_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ARCTIC_EXPEDITION": lambda rng: {
        "table": f"expedition_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "expedition_phase",
                "values": [rng.choice(["camp_setup", "ice_core_drilling", "survey_traverse"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ARCTIC_EXPEDITION",
            },
            {
                "name": "crew_ref",
                "values": [f"crew-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "departed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ASSESSMENT_METHOD": lambda rng: {
        "table": f"appraisal_round_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "assessment_method",
                "values": [rng.choice(["structured_rubric", "criteria_checklist", "standardized_scorecard"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ASSESSMENT_METHOD",
            },
            {
                "name": "evaluated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ATTENDANCE_TRACKING_PROCESS": lambda rng: {
        "table": f"attendance_ledger_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "attendance_status",
                "values": [rng.choice(["present", "tardy", "excused_absent"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ATTENDANCE_TRACKING_PROCESS",
            },
            {
                "name": "student_ref",
                "values": [f"st-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_AUTHORIZED_ENTITY_AUTHORIZATION_PROCESS": lambda rng: {
        "table": f"authorization_grant_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "authorization_outcome",
                "values": [rng.choice(["cleared", "pending_review", "revoked"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_AUTHORIZED_ENTITY_AUTHORIZATION_PROCESS",
            },
            {
                "name": "authorizing_entity_ref",
                "values": [f"ae-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_AUTOHEMOTHERAPY_PROCEDURE": lambda rng: {
        "table": f"autotransfusion_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "transfusion_phase",
                "values": [rng.choice(["draw", "wash", "reinfuse"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_AUTOHEMOTHERAPY_PROCEDURE",
            },
            {
                "name": "patient_ref",
                "values": [f"pt-{rng.randint(10000, 99999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BENEFIT_OPTION_ELECTION": lambda rng: {
        "table": f"pension_option_election_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "elected_benefit_option",
                "values": [rng.choice(["defined_benefit_plan", "cash_balance_plan", "account_accrual_plan"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BENEFIT_OPTION_ELECTION",
            },
            {
                "name": "member_ref",
                "values": [f"mb-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "elected_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BIOCULTURAL_RESEARCH_DESIGN": lambda rng: {
        "table": f"research_design_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "design_protocol",
                "values": [rng.choice(["foodways_interview_protocol", "harvest_dietary_diary_study", "foraging_knowledge_survey"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BIOCULTURAL_RESEARCH_DESIGN",
            },
            {
                "name": "method_applied",
                "values": [rng.choice(["participant_observation", "key_informant_interview", "food_frequency_questionnaire"])],
            },
            {
                "name": "fieldwork_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BOARD_MEETING_AGENDA_ITEM": lambda rng: {
        "table": f"agenda_item_execution_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "agenda_item_action",
                "values": [rng.choice(["ratify_budget_motions", "review_facility_bids", "approve_compensation_plan"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BOARD_MEETING_AGENDA_ITEM",
            },
            {
                "name": "executing_member_ref",
                "values": [f"bm-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "item_number",
                "values": [str(rng.randint(1, 12))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BOARD_MEMBER_ROLE_REALIZATION": lambda rng: {
        "table": f"role_realization_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "realized_member_role",
                "values": [rng.choice(["quorum_witness", "motion_seconding", "assessor_vote"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BOARD_MEMBER_ROLE_REALIZATION",
            },
            {
                "name": "meeting_ref",
                "values": [f"ba-{rng.randint(100, 999)}"],
            },
            {
                "name": "role_recorded_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BOARD_OF_ASSESSORS_MEETING": lambda rng: {
        "table": f"assessor_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "meeting_kind",
                "values": [rng.choice(["parcel_roll_review", "appeal_hearing", "tax_rate_session"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BOARD_OF_ASSESSORS_MEETING",
            },
            {
                "name": "convened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BOUNDED_COMMITTEE_MEMBERSHIP": lambda rng: {
        "table": f"committee_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "committee_assembly",
                "values": [rng.choice(["policy_committee", "budget_committee", "ethics_committee"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BOUNDED_COMMITTEE_MEMBERSHIP",
            },
            {
                "name": "member_count",
                "values": [str(rng.randint(5, 15))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BUSINESS_ACTIVITY_PROCESS": lambda rng: {
        "table": f"business_process_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "activity_process",
                "values": [rng.choice(["order_fulfillment", "inventory_restock", "payroll_run"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BUSINESS_ACTIVITY_PROCESS",
            },
            {
                "name": "entity_ref",
                "values": [f"be-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BUSINESS_DECISION": lambda rng: {
        "table": f"decision_forum_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "business_decision",
                "values": [rng.choice(["budget_approval", "vendor_award", "product_rollout"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BUSINESS_DECISION",
            },
            {
                "name": "decided_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_BUSINESS_INITIATIVE": lambda rng: {
        "table": f"venture_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "initiative_kind",
                "values": [rng.choice(["product_launch", "market_entry", "entity_restructuring"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_BUSINESS_INITIATIVE",
            },
            {
                "name": "entity_ref",
                "values": [f"be-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CALL_ADMISSION_CONTROL": lambda rng: {
        "table": f"admission_gate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "admission_outcome",
                "values": [rng.choice(["granted", "deferred", "rejected"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CALL_ADMISSION_CONTROL",
            },
            {
                "name": "request_ref",
                "values": [f"sr-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CANCER_SUPPORT_PROGRAM": lambda rng: {
        "table": f"support_track_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "support_service_mode",
                "values": [rng.choice(["counselling_round", "nutrition_review", "symptom_relief_round"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CANCER_SUPPORT_PROGRAM",
            },
            {
                "name": "patient_ref",
                "values": [f"pt-{rng.randint(10000, 99999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CAPACITY_BOUNDED_ENROLLMENT": lambda rng: {
        "table": f"cohort_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "enrollment_limit",
                "values": [rng.choice(["12_seats", "18_seats", "24_seats"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CAPACITY_BOUNDED_ENROLLMENT",
            },
            {
                "name": "cohort_ref",
                "values": [f"ch-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CHIEF_COMPLIANCE_OFFICER_SERVICE_DELIVERY_PROCESS": lambda rng: {
        "table": f"compliance_service_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_phase",
                "values": [rng.choice(["advisory_review", "policy_attestation", "regulatory_filing"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CHIEF_COMPLIANCE_OFFICER_SERVICE_DELIVERY_PROCESS",
            },
            {
                "name": "admin_unit",
                "values": [rng.choice(["provosts_office", "student_affairs", "registrar"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLEAN_TRAFFIC_REDIRECTION": lambda rng: {
        "table": f"traffic_redirection_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "diversion_mode",
                "values": [rng.choice(["return_route_swap", "filter_refresh", "upstream_diversion"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLEAN_TRAFFIC_REDIRECTION",
            },
            {
                "name": "scrubbing_center",
                "values": [f"scrub-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLEARING_BROKER_SERVICE_DELIVERY_PROCESS": lambda rng: {
        "table": f"broker_service_engagement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_track",
                "values": [rng.choice(["settlement_support", "margin_reporting", "position_confirmations"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLEARING_BROKER_SERVICE_DELIVERY_PROCESS",
            },
            {
                "name": "clearing_ref",
                "values": [f"cl-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLEARING_PROCESS": lambda rng: {
        "table": f"clearing_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "clearing_stage",
                "values": [rng.choice(["trade_validation", "position_reconciliation", "settlement_finalization"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLEARING_PROCESS",
            },
            {
                "name": "run_date",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLINICAL_CARE": lambda rng: {
        "table": f"care_encounter_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_episode",
                "values": [rng.choice(["outpatient_visit", "diagnostic_workup", "follow_up_consult"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLINICAL_CARE",
            },
            {
                "name": "clinician_id",
                "values": [f"clin-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "encounter_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLINICAL_DOCUMENTATION_ACTIVITY": lambda rng: {
        "table": f"charting_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "documentation_activity",
                "values": [rng.choice(["encounter_note_draft", "progress_note_review", "consult_note_draft"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLINICAL_DOCUMENTATION_ACTIVITY",
            },
            {
                "name": "encounter_id",
                "values": [f"enc-{rng.randint(100000, 999999)}"],
            },
            {
                "name": "system_code",
                "values": [rng.choice(["emr_orion", "lis_north", "pacs_core"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLINICAL_GENOMIC_ANALYSIS_PROGRAM": lambda rng: {
        "table": f"genomics_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "genomic_program",
                "values": [rng.choice(["pharmacogenomic_panel", "tumor_sequencing_run", "heritable_risk_screen"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLINICAL_GENOMIC_ANALYSIS_PROGRAM",
            },
            {
                "name": "health_profile_ref",
                "values": [f"php-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "model_code",
                "values": [rng.choice(["gradient_boost_11", "tree_forest_04", "bayes_net_v3"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CLINICAL_QUALITY_IMPROVEMENT_PROGRAM": lambda rng: {
        "table": f"quality_round_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "improvement_program",
                "values": [rng.choice(["antibiotic_stewardship", "medication_reconciliation", "fall_prevention_bundling"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CLINICAL_QUALITY_IMPROVEMENT_PROGRAM",
            },
            {
                "name": "unit_code",
                "values": [rng.choice(["icu-2", "med-b", "onc-1"])],
            },
            {
                "name": "cycle_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CME_SEMINAR": lambda rng: {
        "table": f"seminar_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "seminar_focus",
                "values": [rng.choice(["billing_procedures", "scheduling_standards", "licensure_renewal"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CME_SEMINAR",
            },
            {
                "name": "session_delivered_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_COLLATERAL_EXCHANGE_PROCESS": lambda rng: {
        "table": f"collateral_settlement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "exchange_action",
                "values": [rng.choice(["deposit", "withdrawal", "substitution"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_COLLATERAL_EXCHANGE_PROCESS",
            },
            {
                "name": "cleared_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_COLLEGE_ADMINISTRATION_PROCESS": lambda rng: {
        "table": f"administration_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "administration_area",
                "values": [rng.choice(["student_records", "enrollment_services", "facilities_ops"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_COLLEGE_ADMINISTRATION_PROCESS",
            },
            {
                "name": "cycle_opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_COMMUNICABLE_DISEASE_NOTIFICATION_REQUIREMENT": lambda rng: {
        "table": f"notification_mandate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "disclosure_scope",
                "values": [rng.choice(["summary_release", "record_release", "report_release"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_COMMUNICABLE_DISEASE_NOTIFICATION_REQUIREMENT",
            },
            {
                "name": "reported_within",
                "values": [rng.choice(["24h", "48h", "72h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_COMMUNITY_HEALTHCARE_SERVICE_DELIVERY_PROCESS": lambda rng: {
        "table": f"clinic_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_delivery_cycle",
                "values": [rng.choice(["home_visit_round", "screening_wave", "outreach_route"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_COMMUNITY_HEALTHCARE_SERVICE_DELIVERY_PROCESS",
            },
            {
                "name": "setting_ref",
                "values": [f"chc-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CONCEPT_CONSTRUCTION_PROCESS": lambda rng: {
        "table": f"concept_build_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "construction_run",
                "values": [rng.choice(["term_derivation", "relation_composition", "predicate_synthesis"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CONCEPT_CONSTRUCTION_PROCESS",
            },
            {
                "name": "subject_concept_ref",
                "values": [f"sc-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CONTINUOUS_EMISSIONS_MONITORING_EVALUATION": lambda rng: {
        "table": f"cms_audit_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "system_evaluation",
                "values": [rng.choice(["bias_check", "stack_test_compliance", "reference_comparison"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CONTINUOUS_EMISSIONS_MONITORING_EVALUATION",
            },
            {
                "name": "chemistry_method",
                "values": [rng.choice(["chamber_digestion", "titration_readout", "isokinetic_sampling"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CONTINUOUS_EMISSIONS_MONITORING_PROCESS": lambda rng: {
        "table": f"emission_watch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "monitoring_stream",
                "values": [rng.choice(["so2_channel", "nox_channel", "opacity_watch"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CONTINUOUS_EMISSIONS_MONITORING_PROCESS",
            },
            {
                "name": "emission_source_ref",
                "values": [f"ies-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CORPORATE_AUDIT_PROCESS": lambda rng: {
        "table": f"audit_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "audit_engagement",
                "values": [rng.choice(["financial_records_review", "controls_compliance_check", "procurement_records_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CORPORATE_AUDIT_PROCESS",
            },
            {
                "name": "subject_entity_ref",
                "values": [f"org-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "fieldwork_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CORPORATE_EMPLOYMENT_PROCESS": lambda rng: {
        "table": f"hiring_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "employment_act",
                "values": [rng.choice(["full_time_engagement", "contract_engagement", "apprentice_engagement"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CORPORATE_EMPLOYMENT_PROCESS",
            },
            {
                "name": "agent_ref",
                "values": [f"emp-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "role_issued_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CORPORATE_INFORMATION_DISCLOSURE_PROCESS": lambda rng: {
        "table": f"disclosure_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "disclosure_round",
                "values": [rng.choice(["regulator_filing", "public_report_release", "supervisory_briefing"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CORPORATE_INFORMATION_DISCLOSURE_PROCESS",
            },
            {
                "name": "financial_report_ref",
                "values": [f"fr-{rng.randint(100, 999)}"],
            },
            {
                "name": "filed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CREDIT_HOUR_CERTIFICATION_PROCESS": lambda rng: {
        "table": f"credit_certification_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "certification_pass",
                "values": [rng.choice(["term_credit_review", "hour_transcript_check", "completion_credit_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CREDIT_HOUR_CERTIFICATION_PROCESS",
            },
            {
                "name": "academic_term",
                "values": [rng.choice(["term_fall", "term_spring", "term_summer"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CUSTODY_SCOPE_TYPING": lambda rng: {
        "table": f"accession_census_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "custody_scope_class",
                "values": [rng.choice(["accessioned_records_only", "mixed_fonds", "pending_accession"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CUSTODY_SCOPE_TYPING",
            },
            {
                "name": "fund_ref",
                "values": [f"fnd-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_CYBERSECURITY_INCIDENT": lambda rng: {
        "table": f"incident_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "disruption_outcome",
                "values": [rng.choice(["service_outage", "system_lockout", "record_access_delay"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_CYBERSECURITY_INCIDENT",
            },
            {
                "name": "target_segment_ref",
                "values": [f"hni-{rng.randint(100, 999)}"],
            },
            {
                "name": "detected_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DATA_COLLECTION_PROCESS": lambda rng: {
        "table": f"monitoring_gather_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "collection_mode",
                "values": [rng.choice(["structured_gather", "monitoring_capture", "polling_extract"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DATA_COLLECTION_PROCESS",
            },
            {
                "name": "store_ref",
                "values": [f"db-{rng.randint(100, 999)}"],
            },
            {
                "name": "cycle_started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DATA_EXPORT_PROCESS": lambda rng: {
        "table": f"artifact_export_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "export_artifact_form",
                "values": [rng.choice(["csv_batch", "json_stream", "sql_dump"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DATA_EXPORT_PROCESS",
            },
            {
                "name": "interface_ref",
                "values": [f"swi-{rng.randint(100, 999)}"],
            },
            {
                "name": "batch_window",
                "values": [rng.choice(["5m", "1h", "24h"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DDOS_MITIGATION_PROCESS": lambda rng: {
        "table": f"ddos_response_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mitigation_process",
                "values": [f"mdp-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DDOS_MITIGATION_PROCESS",
            },
            {
                "name": "triggering_attack_ref",
                "values": [f"dae-{rng.randint(100, 999)}"],
            },
            {
                "name": "protected_infra_ref",
                "values": [f"nix-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DEBT_CASEWORK_PROCESS": lambda rng: {
        "table": f"debt_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "debt_casework",
                "values": [f"dcw-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DEBT_CASEWORK_PROCESS",
            },
            {
                "name": "housing_client_ref",
                "values": [f"hc-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "opened_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DECISION_MAKER_ROLE": lambda rng: {
        "table": f"decision_seat_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "decision_seat",
                "values": [rng.choice(["board_voting_seat", "committee_signoff", "executive_authority"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DECISION_MAKER_ROLE",
            },
            {
                "name": "realized_process_ref",
                "values": [f"dmp-{rng.randint(100, 999)}"],
            },
            {
                "name": "assumed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DIAGNOSTIC_IMAGING": lambda rng: {
        "table": f"imaging_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "diagnostic_study",
                "values": [f"di-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DIAGNOSTIC_IMAGING",
            },
            {
                "name": "imaging_equipment_ref",
                "values": [f"ieq-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "performed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DIGITAL_FORENSIC_ANALYSIS": lambda rng: {
        "table": f"forensic_analysis_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "analysis_kind",
                "values": [rng.choice(["memory_exam", "disk_image_triage", "network_capture_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DIGITAL_FORENSIC_ANALYSIS",
            },
            {
                "name": "incident_ref",
                "values": [f"cs-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DNS_ABUSE_STUDY": lambda rng: {
        "table": f"dns_abuse_study_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "study_kind",
                "values": [rng.choice(["parked_domain_survey", "nxdomain_spike_audit", "sinkhole_hit_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DNS_ABUSE_STUDY",
            },
            {
                "name": "abuse_event_ref",
                "values": [f"ae-{rng.randint(100, 999)}"],
            },
            {
                "name": "gtld_program_ref",
                "values": [rng.choice(["gtld-alpha", "gtld-beta", "gtld-gamma"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DONATION_EVENT": lambda rng: {
        "table": f"gift_ledger_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "event_kind",
                "values": [rng.choice(["major_gift", "planned_giving", "stock_donation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DONATION_EVENT",
            },
            {
                "name": "donor_agent_ref",
                "values": [f"da-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "amount",
                "values": [f"{rng.randint(25, 5000)}.{rng.randint(0, 99):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DONOR": lambda rng: {
        "table": f"donor_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "designation",
                "values": [rng.choice(["patient_care", "research", "operations"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DONOR",
            },
            {
                "name": "hospital_recipient_ref",
                "values": [f"hr-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "designated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_DUAL_CUSTODY_SIGNOFF": lambda rng: {
        "table": f"custody_signoff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "signoff_stage",
                "values": [rng.choice(["placement_ratification", "guardian_consent", "transfer_approval"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_DUAL_CUSTODY_SIGNOFF",
            },
            {
                "name": "signatories_present",
                "values": [str(rng.randint(2, 4))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_EDUCATIONAL_PLACEMENT_PROCESS": lambda rng: {
        "table": f"placement_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "placement_stage",
                "values": [rng.choice(["student_intake", "curriculum_match", "plan_ratification"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_EDUCATIONAL_PLACEMENT_PROCESS",
            },
            {
                "name": "curriculum_ref",
                "values": [f"cu-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_EMPLOYEE_FEEDBACK_SURVEY": lambda rng: {
        "table": f"employee_pulse_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "survey_cycle",
                "values": [rng.choice(["engagement_round", "wellbeing_round", "culture_audit_round"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_EMPLOYEE_FEEDBACK_SURVEY",
            },
            {
                "name": "surveyed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ENTREPRENEURIAL_ACTIVITY_PROCESS": lambda rng: {
        "table": f"startup_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "venture_activity",
                "values": [rng.choice(["product_offering", "service_rollout", "market_entry_pilot"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ENTREPRENEURIAL_ACTIVITY_PROCESS",
            },
            {
                "name": "demand_segment",
                "values": [rng.choice(["urban_commuter", "rural_farm", "campus_housing"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ENVIRONMENTAL_PERMITTING_SUPERVISION_ACTIVITY": lambda rng: {
        "table": f"permitting_supervision_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "supervision_scope",
                "values": [rng.choice(["permitting_decision", "chemical_installation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ENVIRONMENTAL_PERMITTING_SUPERVISION_ACTIVITY",
            },
            {
                "name": "inspected_facility_code",
                "values": [f"insp-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "supervised_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ESCALATION_CHANNEL_UNION": lambda rng: {
        "table": f"escalation_route_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "realization_channel",
                "values": [rng.choice(["supervisory_review", "external_audit_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ESCALATION_CHANNEL_UNION",
            },
            {
                "name": "incident_ref",
                "values": [f"inc-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "escalated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_EVALUATION_OUTCOME": lambda rng: {
        "table": f"assessment_outcome_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "outcome_kind",
                "values": [rng.choice(["quality_gain", "performance_metric", "identified_strength"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_EVALUATION_OUTCOME",
            },
            {
                "name": "assessment_event_ref",
                "values": [f"tae-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "recorded_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_ETHICAL_INVESTMENT": lambda rng: {
        "table": f"ethical_allocation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "activity_purpose",
                "values": [rng.choice(["ethics_screener_fund", "sustainable_bond_pool", "values_aligned_equity"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_ETHICAL_INVESTMENT",
            },
            {
                "name": "asset_ref",
                "values": [f"etia-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "allocated_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_EXECUTIVE_DIRECTION": lambda rng: {
        "table": f"directive_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "executive_direction",
                "values": [rng.choice(["budget_mandate", "reorganization_mandate", "market_entry_mandate"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_EXECUTIVE_DIRECTION",
            },
            {
                "name": "director_ref",
                "values": [f"ed-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "issued_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_EXECUTIVE_IMPLEMENTATION": lambda rng: {
        "table": f"execution_round_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "executive_implementation",
                "values": [rng.choice(["strategy_rollout", "service_migration", "operations_integration"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_EXECUTIVE_IMPLEMENTATION",
            },
            {
                "name": "directive_ref",
                "values": [f"dx-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "executed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FIELD_EQUIPMENT_COORDINATION": lambda rng: {
        "table": f"field_dispatch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "field_equipment_coordination",
                "values": [rng.choice(["rig_dispatch", "pump_assignment", "crane_allocation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FIELD_EQUIPMENT_COORDINATION",
            },
            {
                "name": "equipment_ref",
                "values": [f"oe-{rng.randint(100, 999)}"],
            },
            {
                "name": "dispatched_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINAL_CONSTRUCTION_INSPECTION_PROCESS": lambda rng: {
        "table": f"final_walkdown_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "final_construction_inspection",
                "values": [rng.choice(["pre_pour_check", "pressure_test", "as_built_verification"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINAL_CONSTRUCTION_INSPECTION_PROCESS",
            },
            {
                "name": "infrastructure_ref",
                "values": [f"ui-{rng.randint(100, 999)}"],
            },
            {
                "name": "verified_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINAL_INSPECTION_PROCESS": lambda rng: {
        "table": f"final_inspection_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "inspection_phase",
                "values": [rng.choice(["acceptance_check", "defect_review", "release_signoff"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINAL_INSPECTION_PROCESS",
            },
            {
                "name": "artifact_lot",
                "values": [f"lot-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_CRISIS_EVENT": lambda rng: {
        "table": f"crisis_watch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "crisis_type",
                "values": [rng.choice(["credit_freeze", "currency_devaluation", "market_dislocation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_CRISIS_EVENT",
            },
            {
                "name": "affected_market",
                "values": [rng.choice(["equity", "bond", "forex"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_PLANNING_PROCESS": lambda rng: {
        "table": f"planning_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "planning_scope",
                "values": [rng.choice(["capital_budgeting", "cash_forecasting", "dividend_policing"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_PLANNING_PROCESS",
            },
            {
                "name": "plan_ref",
                "values": [f"fp-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_PRODUCT_ADVICE_SERVICE": lambda rng: {
        "table": f"advice_ticket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advice_channel",
                "values": [rng.choice(["branch_meeting", "phone_session", "secure_portal"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_PRODUCT_ADVICE_SERVICE",
            },
            {
                "name": "product_class",
                "values": [rng.choice(["term_deposit", "bond_fund", "unit_trust"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_STRATEGY_AND_PLANNING_PROCESS": lambda rng: {
        "table": f"finance_plan_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "plan_process",
                "values": [rng.choice(["capital_program_plan", "recovery_budget_plan", "dividend_policy_plan"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FINANCIAL_STRATEGY_AND_PLANNING_PROCESS",
            },
            {
                "name": "executive_forum",
                "values": [rng.choice(["cfo_briefing", "board_finance_session", "treasury_review"])],
            },
            {
                "name": "report_draft_due",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_FLOOD_ASSET_CONDITION_ASSESSMENT": lambda rng: {
        "table": f"flood_asset_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "condition_survey",
                "values": [rng.choice(["levee_section_walk", "pump_station_round", "drain_gate_inspection"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_FLOOD_ASSET_CONDITION_ASSESSMENT",
            },
            {
                "name": "earth_observation_feed",
                "values": [rng.choice(["satellite_imagery", "drone_flight", "radar_composite"])],
            },
            {
                "name": "surveyed_on",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_GENERAL_MANAGER_FUNCTION_EXECUTION": lambda rng: {
        "table": f"manager_execution_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "execution_run",
                "values": [rng.choice(["daily_ops_brief", "weekly_capacity_review", "regional_standup"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_GENERAL_MANAGER_FUNCTION_EXECUTION",
            },
            {
                "name": "function_ref",
                "values": [f"fn-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_GENOMIC_SEQUENCING_TEST": lambda rng: {
        "table": f"sequencing_lane_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "sequencing_run",
                "values": [rng.choice(["tumor_panel_lane", "germline_screen_lane", "viral_surveillance_lane"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_GENOMIC_SEQUENCING_TEST",
            },
            {
                "name": "specimen_ref",
                "values": [f"sp-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "sequencer_class",
                "values": [rng.choice(["novaseq_x", "miseq_ac", "hiseq_4k"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_GRADUATION_ELIGIBILITY_ASSESSMENT": lambda rng: {
        "table": f"graduation_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "eligibility_screen",
                "values": [rng.choice(["credit_check", "gpa_verification", "debt_clearance"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_GRADUATION_ELIGIBILITY_ASSESSMENT",
            },
            {
                "name": "student_agent_ref",
                "values": [f"sa-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "program_of_study_ref",
                "values": [f"pos-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_HEALTHCARE_ANALYTICS_PROCESS": lambda rng: {
        "table": f"analytics_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "analytics_job",
                "values": [rng.choice(["readmission_model", "sepsis_screen", "utilization_forecast"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_HEALTHCARE_ANALYTICS_PROCESS",
            },
            {
                "name": "data_input_ref",
                "values": [f"di-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "intelligence_output_ref",
                "values": [f"io-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_HEALTH_RECORD_TRANSFER_PROCESS": lambda rng: {
        "table": f"record_transfer_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "transfer_activity",
                "values": [rng.choice(["discharge_summary_push", "medication_list_sync", "imaging_index_replication"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_HEALTH_RECORD_TRANSFER_PROCESS",
            },
            {
                "name": "origin_system",
                "values": [rng.choice(["emr_north", "pacs_west", "lab_east"])],
            },
            {
                "name": "transferred_at",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 9):02d}-{rng.randint(10, 28)}T{rng.randint(0, 23):02d}:00:00Z"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_HYPOTHESIS_TEST": lambda rng: {
        "table": f"inference_trial_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "test_trial",
                "values": [rng.choice(["two_sided_mean_test", "paired_difference_test", "independence_test"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_HYPOTHESIS_TEST",
            },
            {
                "name": "test_statistic_ref",
                "values": [f"ts-{rng.randint(100, 999)}"],
            },
            {
                "name": "significance_level",
                "values": [rng.choice(["0.05", "0.01", "0.10"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_H_S_S_S_E_Q__SITUATION": lambda rng: {
        "table": f"readiness_stasis_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "readiness_level",
                "values": [rng.choice(["staged", "nominal", "degraded"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_H_S_S_S_E_Q__SITUATION",
            },
            {
                "name": "unit_id",
                "values": [f"unit_{rng.randint(100, 999)}"],
            },
            {
                "name": "documented_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_IMPROPER_CONDUCT_EVENT": lambda rng: {
        "table": f"conduct_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "resource_misused",
                "values": [rng.choice(["library_database", "lab_bench", "rec_center_keycard"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_IMPROPER_CONDUCT_EVENT",
            },
            {
                "name": "reported_by",
                "values": [f"member_{rng.randint(100, 999)}"],
            },
            {
                "name": "opened_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INCUBATION_PROGRAM": lambda rng: {
        "table": f"incubation_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "workspace_allocation",
                "values": [rng.choice(["shared_bench", "private_suite", "maker_lab"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INCUBATION_PROGRAM",
            },
            {
                "name": "cohort_size",
                "values": [str(rng.randint(4, 16))],
            },
            {
                "name": "term_start",
                "values": [f"201{rng.randint(0, 9)}-{rng.choice(['01', '09'])}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INDEPENDENT_COMMITTEE_INVESTIGATION": lambda rng: {
        "table": f"committee_inquiry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "analytic_method",
                "values": [rng.choice(["root_cause", "case_comparative", "trend_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INDEPENDENT_COMMITTEE_INVESTIGATION",
            },
            {
                "name": "member_count",
                "values": [str(rng.randint(3, 7))],
            },
            {
                "name": "referral_date",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INFORMATION_SERVICE": lambda rng: {
        "table": f"service_delivery_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "delivery_operation",
                "values": [rng.choice(["timetable_publish", "catalog_publish", "alert_broadcast"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INFORMATION_SERVICE",
            },
            {
                "name": "response_format",
                "values": [rng.choice(["html", "json", "atom"])],
            },
            {
                "name": "went_live_on",
                "values": [f"201{rng.randint(0, 9)}-04-12"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INFORMED_CONSENT_PROCESS": lambda rng: {
        "table": f"consent_capture_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "consent_activity",
                "values": [rng.choice(["verbal_briefing", "document_walkthrough", "comprehension_check"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INFORMED_CONSENT_PROCESS",
            },
            {
                "name": "witness_presented",
                "values": [rng.choice(["yes", "no"])],
            },
            {
                "name": "logged_on",
                "values": [f"201{rng.randint(0, 9)}-08-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INSPECTION_EVENT": lambda rng: {
        "table": f"inspection_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "observation_mode",
                "values": [rng.choice(["floor_walkthrough", "document_review", "equipment_check"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INSPECTION_EVENT",
            },
            {
                "name": "outcome_code",
                "values": [rng.choice(["pass", "minor_fix", "major_fix"])],
            },
            {
                "name": "logged_at",
                "values": [f"201{rng.randint(0, 9)}-09-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INTELLIGENCE_ANALYSIS_PROCESS": lambda rng: {
        "table": f"analysis_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "analysis_activity",
                "values": [rng.choice(["threat_synthesis", "target_assessment", "pattern_correlation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INTELLIGENCE_ANALYSIS_PROCESS",
            },
            {
                "name": "case_reference",
                "values": [f"CASE-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "completed_on",
                "values": [f"201{rng.randint(0, 9)}-05-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INTERDISCIPLINARY_EDUCATION": lambda rng: {
        "table": f"joint_module_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "module_code",
                "values": [f"XLM{rng.randint(100, 999)}"],
            },
            {
                "name": "instructional_process",
                "values": [rng.choice(["joint_lecture", "team_project", "lab_rotation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INTERDISCIPLINARY_EDUCATION",
            },
            {
                "name": "host_departments",
                "values": [rng.choice(["bio_eng", "chem_phys", "math_lit"])],
            },
            {
                "name": "starts_term",
                "values": [f"20{rng.randint(24, 29)}-{rng.choice(['a', 's', 'q'])}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INTERNATIONAL_GROWTH_INITIATIVE": lambda rng: {
        "table": f"market_entry_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "campaign_id",
                "values": [f"MG-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "growth_initiative",
                "values": [rng.choice(["market_launch", "franchise_rollout", "partner_expansion"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INTERNATIONAL_GROWTH_INITIATIVE",
            },
            {
                "name": "target_region",
                "values": [rng.choice(["emea", "apac", "latam", "na_east"])],
            },
            {
                "name": "marketing_plan_ref",
                "values": [f"MP-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INVESTMENT_CHOICE_DECISION": lambda rng: {
        "table": f"allocation_decision_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "decision_id",
                "values": [f"ICD-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "decision_act",
                "values": [rng.choice(["option_selected", "instrument_chosen", "allocation_set"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INVESTMENT_CHOICE_DECISION",
            },
            {
                "name": "investor_type",
                "values": [rng.choice(["institutional", "advisor_proxy", "retail"])],
            },
            {
                "name": "asset_class",
                "values": [rng.choice(["equity", "fixed_income", "money_market"])],
            },
            {
                "name": "recorded_on",
                "values": [f"202{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_INVESTOR_FUNDING_SITUATION": lambda rng: {
        "table": f"venture_funding_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "round_id",
                "values": [f"VF-{rng.randint(100, 999)}"],
            },
            {
                "name": "funding_act",
                "values": [rng.choice(["equity_injected", "debt_extended", "capital_allocated"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_INVESTOR_FUNDING_SITUATION",
            },
            {
                "name": "funding_source",
                "values": [rng.choice(["institutional_vc", "growth_fund", "strategic_investor"])],
            },
            {
                "name": "instrument",
                "values": [rng.choice(["equity", "debt", "convertible_note"])],
            },
            {
                "name": "closed_on",
                "values": [f"202{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_IT_GOVERNANCE_PROCESS": lambda rng: {
        "table": f"it_capability_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "governance_process",
                "values": [rng.choice(["change_advisory", "capacity_planning", "budget_gating"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_IT_GOVERNANCE_PROCESS",
            },
            {
                "name": "review_ref",
                "values": [f"IR{rng.randint(1000, 9999)}"],
            },
            {
                "name": "opened_on",
                "values": [f"202{rng.randint(3, 6)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_JOINT_BOARD_CHAIR_SERVICE_DELIVERY_PROCESS": lambda rng: {
        "table": f"service_delivery_board_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "chair_delivery_track",
                "values": [rng.choice(["joint_session", "cross_committee", "chair_referral"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_JOINT_BOARD_CHAIR_SERVICE_DELIVERY_PROCESS",
            },
            {
                "name": "board_ref",
                "values": [f"SB{rng.randint(100, 999)}"],
            },
            {
                "name": "convened_on",
                "values": [f"202{rng.randint(3, 6)}-07-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LAW_AND_SOCIETY_TRACK_REQUIREMENT": lambda rng: {
        "table": f"course_selection_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "selection_process",
                "values": [rng.choice(["required_sequence", "elective_block", "core_capstone"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LAW_AND_SOCIETY_TRACK_REQUIREMENT",
            },
            {
                "name": "track_code",
                "values": [f"LAS{rng.randint(100, 499)}"],
            },
            {
                "name": "semester",
                "values": [rng.choice(["fall", "spring"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LECTURE_SESSION": lambda rng: {
        "table": f"teaching_calendar_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "session_kind",
                "values": [rng.choice(["expository", "workshop", "review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LECTURE_SESSION",
            },
            {
                "name": "notes_ref",
                "values": [f"LN{rng.randint(1000, 9999)}"],
            },
            {
                "name": "example_ref",
                "values": [f"ME{rng.randint(1000, 9999)}"],
            },
            {
                "name": "held_on",
                "values": [f"202{rng.randint(3, 6)}-10-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LEGAL_ADVERTISEMENT_ACTIVITY": lambda rng: {
        "table": f"legal_service_notice_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advertised_service",
                "values": [rng.choice(["pro_bono_mediation_clinic", "eviction_defense_workshop", "protection_order_info_line"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LEGAL_ADVERTISEMENT_ACTIVITY",
            },
            {
                "name": "notice_id",
                "values": [f"LN-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "published_on",
                "values": [f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LEGAL_ADVOCACY_PROCESS": lambda rng: {
        "table": f"advocacy_matter_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advocacy_action",
                "values": [rng.choice(["protective_order_motion", "custody_appeal_filing", "restraining_order_renewal"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LEGAL_ADVOCACY_PROCESS",
            },
            {
                "name": "court_document_type",
                "values": [rng.choice(["affidavit", "motion", "brief"])],
            },
            {
                "name": "filed_on",
                "values": [f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LEGAL_PROCEEDING": lambda rng: {
        "table": f"court_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "proceeding_stage",
                "values": [rng.choice(["jury_selection", "evidence_review", "jury_deliberation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LEGAL_PROCEEDING",
            },
            {
                "name": "docket_number",
                "values": [f"CR-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "hearing_on",
                "values": [f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_LEGAL_TECH_SERVICE_DELIVERY_PROCESS": lambda rng: {
        "table": f"legaltech_release_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "lab_delivered_service",
                "values": [rng.choice(["case_timeline_builder", "evidence_upload_portal", "intake_form_generator"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_LEGAL_TECH_SERVICE_DELIVERY_PROCESS",
            },
            {
                "name": "pilot_court_code",
                "values": [f"CT-{rng.randint(10, 99)}"],
            },
            {
                "name": "launched_on",
                "values": [f"2025-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MAIL_DISPATCH_PROCESS": lambda rng: {
        "table": f"post_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "dispatch_run",
                "values": [rng.choice(["am_sort", "pm_sort", "overnight", "express"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MAIL_DISPATCH_PROCESS",
            },
            {
                "name": "dest_drop",
                "values": [rng.choice(["hub_03", "hub_07", "depot_a", "depot_c"])],
            },
            {
                "name": "sort_rule",
                "values": [rng.choice(["dept_floor", "unit_code", "vendor_tag"])],
            },
            {
                "name": "run_date",
                "values": [f"202{rng.randint(0, 5)}-0{rng.randint(1, 9)}-0{rng.randint(1, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MALNUTRITION_TREATMENT_PROCESS": lambda rng: {
        "table": f"therapy_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "treatment_cycle",
                "values": [rng.choice(["phase_1", "phase_2", "phase_3", "phase_4"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MALNUTRITION_TREATMENT_PROCESS",
            },
            {
                "name": "child_ref",
                "values": [f"mnc-{rng.randint(100, 999)}"],
            },
            {
                "name": "recovery_state",
                "values": [rng.choice(["gained", "stable", "improving"])],
            },
            {
                "name": "cycle_week",
                "values": [f"202{rng.randint(0, 5)}-W{rng.randint(1, 52):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MANAGEMENT_MEETING_PROCESS": lambda rng: {
        "table": f"board_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "meeting_session",
                "values": [rng.choice(["q1_rev", "q2_rev", "q3_rev", "q4_rev"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MANAGEMENT_MEETING_PROCESS",
            },
            {
                "name": "agent_ref",
                "values": [f"corp_{rng.randint(1000, 9999)}"],
            },
            {
                "name": "decision_ref",
                "values": [f"dec_{rng.randint(100, 999)}"],
            },
            {
                "name": "session_date",
                "values": [f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-0{rng.randint(1, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MATHEMATICAL_LEARNING_PROCESS": lambda rng: {
        "table": f"lesson_track_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "learning_session",
                "values": [rng.choice(["algebra", "geometry", "precalc", "number_theory"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MATHEMATICAL_LEARNING_PROCESS",
            },
            {
                "name": "concept_ref",
                "values": [f"mc_{rng.randint(100, 999)}"],
            },
            {
                "name": "student_ref",
                "values": [f"st_{rng.randint(1000, 9999)}"],
            },
            {
                "name": "term_code",
                "values": [rng.choice(["fa24", "sp25", "fa25"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MEMBERSHIP": lambda rng: {
        "table": f"member_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "membership_status",
                "values": [rng.choice(["active", "probationary", "dormant"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MEMBERSHIP",
            },
            {
                "name": "joined_on",
                "values": [f"202{rng.randint(0, 9)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MEMBERSHIP_TRANSFER_PROCESS": lambda rng: {
        "table": f"membership_transfer_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "transfer_stage",
                "values": [rng.choice(["proposed", "approved", "executed"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MEMBERSHIP_TRANSFER_PROCESS",
            },
            {
                "name": "transferee_ref",
                "values": [f"IND-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MENTORSHIP_PROGRAM": lambda rng: {
        "table": f"mentor_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "cohort_id",
                "values": [f"CO-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MENTORSHIP_PROGRAM",
            },
            {
                "name": "pairing_role",
                "values": [rng.choice(["mentor", "mentee"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MOLECULAR_DYNAMICS_SIMULATION_PROCESS": lambda rng: {
        "table": f"md_trajectory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "run_phase",
                "values": [rng.choice(["minimization", "equilibration", "production"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MOLECULAR_DYNAMICS_SIMULATION_PROCESS",
            },
            {
                "name": "software_ref",
                "values": [rng.choice(["gromacs", "amber", "namd"])],
            },
            {
                "name": "target_ref",
                "values": [f"TGT-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_MUNICIPAL_POOL_OPERATIONS_MANAGEMENT_ACTIVITY": lambda rng: {
        "table": f"pool_ops_activity_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "pool_ops_activity",
                "values": [rng.choice(["water_quality_check", "lane_line_resurfacing", "lifeguard_rostering"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_MUNICIPAL_POOL_OPERATIONS_MANAGEMENT_ACTIVITY",
            },
            {
                "name": "pool_facility_code",
                "values": [f"MPF-{rng.randint(100, 999)}"],
            },
            {
                "name": "season_start",
                "values": [f"202{rng.randint(0, 4)}-06-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_NOTIFICATION_TARGET_UNION": lambda rng: {
        "table": f"notice_target_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "notification_target",
                "values": [rng.choice(["responsible_officer", "delegated_agent"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_NOTIFICATION_TARGET_UNION",
            },
            {
                "name": "notice_reference",
                "values": [f"NTU-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "notice_issued_on",
                "values": [f"202{rng.randint(0, 9)}-04-12"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_OPERATIONAL_DISRUPTION": lambda rng: {
        "table": f"disruption_incident_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "disruption_state",
                "values": [rng.choice(["equipment_failure_halt", "power_outage_pause", "staffing_shortage_delay"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_OPERATIONAL_DISRUPTION",
            },
            {
                "name": "affected_area",
                "values": [rng.choice(["north_wing", "dispatch_floor", "warehouse_bay"])],
            },
            {
                "name": "detected_on",
                "values": [f"202{rng.randint(0, 9)}-08-14"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_OPERATIONAL_EFFECTIVENESS": lambda rng: {
        "table": f"effectiveness_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "effectiveness_rating",
                "values": [rng.choice(["meets_standard", "below_target", "exceeds_target"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_OPERATIONAL_EFFECTIVENESS",
            },
            {
                "name": "operational_area_id",
                "values": [f"OA-{rng.randint(10, 99)}"],
            },
            {
                "name": "assessed_quarter",
                "values": [f"202{rng.randint(0, 9)}-Q{rng.randint(1, 4)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_OZONE_THERAPY_SESSION": lambda rng: {
        "table": f"ozone_treatment_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "therapy_session",
                "values": [rng.choice(["whole_blood", "topical", "inhalation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_OZONE_THERAPY_SESSION",
            },
            {
                "name": "administered_on",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PARENTING_PROCESS": lambda rng: {
        "table": f"family_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "care_process",
                "values": [rng.choice(["home_visit", "group", "telehealth"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PARENTING_PROCESS",
            },
            {
                "name": "participant_ref",
                "values": [f"ADU-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PDF_EXPORT_PROCESS": lambda rng: {
        "table": f"export_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "render_target",
                "values": [rng.choice(["pdf", "dvi", "html"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PDF_EXPORT_PROCESS",
            },
            {
                "name": "source_set_ref",
                "values": [f"IMG-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PERFORMANCE_BASED_CARE_DELIVERY_PROCESS": lambda rng: {
        "table": f"care_cohort_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "delivery_model",
                "values": [rng.choice(["shared_savings", "incentive", "episode_based"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PERFORMANCE_BASED_CARE_DELIVERY_PROCESS",
            },
            {
                "name": "population_ref",
                "values": [f"CPT-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PERSONAL_DATA_PROCESSING_ACTIVITY": lambda rng: {
        "table": f"pdp_activity_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "personal_data_processing_activity",
                "values": [rng.choice(["consent_sync", "retention_purge", "record_merge"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PERSONAL_DATA_PROCESSING_ACTIVITY",
            },
            {
                "name": "executing_system",
                "values": [f"pipeline-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_POLICE_PATROL_PROCESS": lambda rng: {
        "table": f"patrol_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "police_patrol_process",
                "values": [rng.choice(["downtown_loop", "night_sweep", "traffic_checkpoint"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_POLICE_PATROL_PROCESS",
            },
            {
                "name": "patrolling_unit",
                "values": [f"unit-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_POLICY_VIOLATION": lambda rng: {
        "table": f"violation_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "policy_violation",
                "values": [rng.choice(["unauthorized_access", "resource_misuse", "conduct_breach"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_POLICY_VIOLATION",
            },
            {
                "name": "violated_policy_ref",
                "values": [f"POL-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PRACTICE_SESSION": lambda rng: {
        "table": f"practice_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "practice_session",
                "values": [rng.choice(["scenario_run", "skills_drill", "drill_review"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PRACTICE_SESSION",
            },
            {
                "name": "improvement_target",
                "values": [rng.choice(["de_escalation", "report_writing", "first_aid"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PRIM_SKILLIMPROVEMENT": lambda rng: {
        "table": f"capability_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "improvement_cycle",
                "values": [rng.choice(["recertify", "cross_train", "drill_block"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PRIM_SKILLIMPROVEMENT",
            },
            {
                "name": "trainee_id",
                "values": [f"trn_{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PRIM_STATEACTION": lambda rng: {
        "table": f"governor_action_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "government_action",
                "values": [rng.choice(["proclamation", "ordinance", "executive_order"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PRIM_STATEACTION",
            },
            {
                "name": "signed_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-0{rng.randint(1, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PROFESSIONAL_EXPERIENCE": lambda rng: {
        "table": f"practice_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "practice_entry",
                "values": [rng.choice(["clinical_shift", "case_conference", "field_supervision"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PROFESSIONAL_EXPERIENCE",
            },
            {
                "name": "provider_id",
                "values": [f"prv_{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PROGRAM_OF_STUDY": lambda rng: {
        "table": f"degree_plan_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "study_plan",
                "values": [rng.choice(["bsc_plan", "mssw_plan", "phd_plan"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PROGRAM_OF_STUDY",
            },
            {
                "name": "credit_hours",
                "values": [f"{rng.randint(90, 150)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PROMOTIONAL_CALENDAR_ORDER": lambda rng: {
        "table": f"promo_calendar_order_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "order_status",
                "values": [rng.choice(["quoted", "printing", "shipped"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PROMOTIONAL_CALENDAR_ORDER",
            },
            {
                "name": "calendar_sku",
                "values": [f"CHR-{rng.randint(100, 999)}"],
            },
            {
                "name": "order_qty",
                "values": [f"{rng.randint(50, 500)}"],
            },
            {
                "name": "promo_start",
                "values": [f"20{rng.randint(25, 26)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PROTECTED_DISCLOSURE_PROCESS": lambda rng: {
        "table": f"protected_disclosure_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "case_status",
                "values": [rng.choice(["intake", "under_review", "closed"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PROTECTED_DISCLOSURE_PROCESS",
            },
            {
                "name": "reporting_channel",
                "values": [rng.choice(["hotline", "web_form", "in_person"])],
            },
            {
                "name": "official_role",
                "values": [rng.choice(["dean", "ombuds", "provost"])],
            },
            {
                "name": "opened_on",
                "values": [f"20{rng.randint(24, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_PROVIDER_CHANGE_REQUEST_PROCESS": lambda rng: {
        "table": f"provider_change_request_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "request_status",
                "values": [rng.choice(["submitted", "approved", "declined"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_PROVIDER_CHANGE_REQUEST_PROCESS",
            },
            {
                "name": "requested_provider_type",
                "values": [rng.choice(["pediatrics", "dermatology", "behavioral_health"])],
            },
            {
                "name": "client_ref",
                "values": [f"CL-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "submitted_on",
                "values": [f"20{rng.randint(24, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_QUALIFIED_SOURCE_TESTING_PROCESS": lambda rng: {
        "table": f"source_testing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "testing_status",
                "values": [rng.choice(["in_lab", "certified", "failed"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_QUALIFIED_SOURCE_TESTING_PROCESS",
            },
            {
                "name": "certification_level",
                "values": [rng.choice(["bronze", "silver", "gold"])],
            },
            {
                "name": "lot_id",
                "values": [f"LOT-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "tested_on",
                "values": [f"20{rng.randint(24, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_QUALITY_PRACTICE": lambda rng: {
        "table": f"quality_review_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "review_kind",
                "values": [rng.choice(["line_walk", "spec_check", "sample_audit"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_QUALITY_PRACTICE",
            },
            {
                "name": "conducted_on",
                "values": [f"202{rng.randint(0, 5)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RAFFLE": lambda rng: {
        "table": f"prize_draw_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "draw_kind",
                "values": [rng.choice(["ticket_draw", "ballot_draw", "digital_draw"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RAFFLE",
            },
            {
                "name": "prize_amount",
                "values": [f"${rng.randint(25, 500)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RESEARCH_MANAGEMENT_ACTIVITY": lambda rng: {
        "table": f"research_coordination_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "coordination_kind",
                "values": [rng.choice(["grant_coordination", "cohort_planning", "study_supervision"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RESEARCH_MANAGEMENT_ACTIVITY",
            },
            {
                "name": "project_id",
                "values": [f"PRJ-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RESIDENCY_REQUIREMENT": lambda rng: {
        "table": f"enrollment_mandate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mandate_kind",
                "values": [rng.choice(["week_minimum", "credit_minimum", "attendance_minimum"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RESIDENCY_REQUIREMENT",
            },
            {
                "name": "enforced_by_office",
                "values": [rng.choice(["registrar", "admissions", "housing_office"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RESPONSIVE_WEBSITE_DEVELOPMENT": lambda rng: {
        "table": f"web_build_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "responsive_site_build",
                "values": [rng.choice(["rebuild", "redesign", "refactor"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RESPONSIVE_WEBSITE_DEVELOPMENT",
            },
            {
                "name": "target_viewport",
                "values": [rng.choice(["mobile", "tablet", "desktop"])],
            },
            {
                "name": "min_width_px",
                "values": [str(rng.choice([320, 768, 1024]))],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RESTRICTED_PARTICIPATION_REVIEW": lambda rng: {
        "table": f"accreditation_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "credential_gated_review",
                "values": [rng.choice(["peer_accreditation", "license_renewal", "certification_audit"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RESTRICTED_PARTICIPATION_REVIEW",
            },
            {
                "name": "reviewer_clearance",
                "values": [rng.choice(["credentialed", "licensed", "board_approved"])],
            },
            {
                "name": "review_cycle",
                "values": [f"202{rng.randint(0, 9)}-q{rng.randint(1, 4)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_REVIEW_PANEL_QUORUM": lambda rng: {
        "table": f"panel_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "quorum_rule",
                "values": [rng.choice(["three_member_min", "majority_vote", "supermajority"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_REVIEW_PANEL_QUORUM",
            },
            {
                "name": "seat_count",
                "values": [str(rng.randint(3, 9))],
            },
            {
                "name": "convenes_on",
                "values": [f"202{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RIGHTS_EXERCISE_PROCESS": lambda rng: {
        "table": f"participation_event_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "rights_exercise_event",
                "values": [rng.choice(["consent_dialogue", "language_accommodation", "community_consultation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RIGHTS_EXERCISE_PROCESS",
            },
            {
                "name": "participant_age_band",
                "values": [rng.choice(["0-5", "6-12", "13-17"])],
            },
            {
                "name": "venue",
                "values": [rng.choice(["clinic", "school", "community_hall"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RISK_INFORMED_ACCEPTABILITY_EVALUATION": lambda rng: {
        "table": f"acceptability_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "review_ref",
                "values": [f"RAE-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RISK_INFORMED_ACCEPTABILITY_EVALUATION",
            },
            {
                "name": "risk_metric_reading",
                "values": [f"{rng.randint(1, 9)}.{rng.randint(0, 9)}"],
            },
            {
                "name": "condition_assessed",
                "values": [rng.choice(["degraded", "impaired", "stable"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_RITUAL_ACT_TO_RESTORE_ORDER": lambda rng: {
        "table": f"restoration_rite_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "rite_ref",
                "values": [f"RTO-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_RITUAL_ACT_TO_RESTORE_ORDER",
            },
            {
                "name": "artifact_consecrated",
                "values": [rng.choice(["blessed_oil", "processional_banner", "communal_vessel"])],
            },
            {
                "name": "participants",
                "values": [f"{rng.randint(2, 40)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SAFEGUARDING_REPORT_PROCESS": lambda rng: {
        "table": f"referral_filing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "filing_ref",
                "values": [f"SRP-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SAFEGUARDING_REPORT_PROCESS",
            },
            {
                "name": "executing_team",
                "values": [rng.choice(["cpt_district_north", "cpt_district_south", "cpt_mobile_unit"])],
            },
            {
                "name": "directed_to",
                "values": [rng.choice(["county_welfare_dept", "prosecutors_office", "guardian_ad_litem"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SAFEGUARD_MEASURE": lambda rng: {
        "table": f"gtld_directive_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "directive_ref",
                "values": [f"SM-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SAFEGUARD_MEASURE",
            },
            {
                "name": "abuse_pattern_mitigated",
                "values": [rng.choice(["domain_typoquatting", "bulk_phishing", "spam_registration"])],
            },
            {
                "name": "applies_to_program",
                "values": [rng.choice([".travel", ".hotel", ".museum", ".aero"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SERVICE_MANAGEMENT_PROCESS": lambda rng: {
        "table": f"service_ops_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "coordination_mode",
                "values": [rng.choice(["scheduled", "event_driven", "on_demand"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SERVICE_MANAGEMENT_PROCESS",
            },
            {
                "name": "sla_target",
                "values": [rng.choice(["99.0", "99.5", "99.9"])],
            },
            {
                "name": "review_cycle",
                "values": [rng.choice(["weekly", "monthly", "quarterly"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SERVICE_PROVISION_ACTIVITY": lambda rng: {
        "table": f"service_delivery_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "fulfillment_status",
                "values": [rng.choice(["scheduled", "in_progress", "delivered"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SERVICE_PROVISION_ACTIVITY",
            },
            {
                "name": "client_tier",
                "values": [rng.choice(["standard", "priority", "enterprise"])],
            },
            {
                "name": "contract_ref",
                "values": [f"CTR-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SERVICE_REQUEST": lambda rng: {
        "table": f"service_request_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "request_kind",
                "values": [rng.choice(["new_service", "change_order", "cancellation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SERVICE_REQUEST",
            },
            {
                "name": "urgency",
                "values": [rng.choice(["low", "medium", "high"])],
            },
            {
                "name": "submitted_on",
                "values": [f"2026-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SOCIAL_HARMONY_RESTORATION": lambda rng: {
        "table": f"reconciliation_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "resolution_style",
                "values": [rng.choice(["mediation", "restorative_circle", "direct_apology"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SOCIAL_HARMONY_RESTORATION",
            },
            {
                "name": "group_size",
                "values": [rng.choice(["3", "5", "7", "9"])],
            },
            {
                "name": "follow_up_days",
                "values": [rng.choice(["14", "21", "30"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_MANAGEMENT_PROCESS": lambda rng: {
        "table": f"sm_account_mgmt_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "account_mgmt_event",
                "values": [rng.choice(["grant_refresh", "account_deprovision", "permission_reset"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_MANAGEMENT_PROCESS",
            },
            {
                "name": "administrator_login",
                "values": [rng.choice(["a.reyes", "t.okafor", "l.chen"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_MANAGEMENT_PROCESS_DEFINITION": lambda rng: {
        "table": f"sm_authz_mgmt_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "authz_management_event",
                "values": [rng.choice(["grant_rotation", "record_reconciliation", "scope_revocation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_MANAGEMENT_PROCESS_DEFINITION",
            },
            {
                "name": "network_service",
                "values": [rng.choice(["xfeed", "blinkchat", "vimeo-lite"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_REGISTRATION": lambda rng: {
        "table": f"sm_account_reg_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "registration_event",
                "values": [rng.choice(["account_signup", "account_onboarding", "account_enrollment"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SOCIAL_MEDIA_ACCOUNT_REGISTRATION",
            },
            {
                "name": "credentials_ref",
                "values": [rng.choice(["lc_8821", "lc_4075", "lc_1392"])],
            },
            {
                "name": "directed_to_officer",
                "values": [rng.choice(["pio_mreid", "pio_ktan", "pio_bsa"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SOCIAL_TAGGING_PROCESS": lambda rng: {
        "table": f"sm_tagging_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "tagging_run",
                "values": [rng.choice(["tag_apply", "tag_attach", "tag_assign"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SOCIAL_TAGGING_PROCESS",
            },
            {
                "name": "target_item_ref",
                "values": [rng.choice(["ii_2210", "ii_5587", "ii_9031"])],
            },
            {
                "name": "emitted_tag",
                "values": [rng.choice(["#onboarding", "#outage", "#policy-update"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_SPECIFICATION_TESTING_EVENT": lambda rng: {
        "table": f"spec_test_event_{rng.randint(100, 999)}",
        "columns": [
            {
                "name": "testing_event_id",
                "values": [f"SPE-20{rng.randint(0, 9)}{rng.randint(10, 99)}-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_SPECIFICATION_TESTING_EVENT",
            },
            {
                "name": "protocol_ref",
                "values": [rng.choice(["TP_LOAD_01", "TP_THERMAL_04", "TP_VIBRATION_11"])],
            },
            {
                "name": "target_material",
                "values": [rng.choice(["rivet_lot_77", "panel_batch_12", "casing_run_3"])],
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(4, 6)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_STRAP_FIXING_PROCESS": lambda rng: {
        "table": f"strap_fix_{rng.randint(100, 999)}",
        "columns": [
            {
                "name": "fix_process_id",
                "values": [f"FIX-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_STRAP_FIXING_PROCESS",
            },
            {
                "name": "strap_id",
                "values": [f"STR-{rng.randint(100, 999)}"],
            },
            {
                "name": "floor_structure_ref",
                "values": [rng.choice(["floor_b2", "floor_c1", "mezz_03"])],
            },
            {
                "name": "secured_on",
                "values": [f"202{rng.randint(4, 6)}-0{rng.randint(1, 9)}-20"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_TAXI_RIDE_PROCESS": lambda rng: {
        "table": f"taxi_ride_{rng.randint(1000, 9999)}",
        "columns": [
            {
                "name": "ride_id",
                "values": [f"RIDE-{rng.randint(10000, 99999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_TAXI_RIDE_PROCESS",
            },
            {
                "name": "boarding_point",
                "values": [rng.choice(["central_station", "harbor_hotel", "airport_park"])],
            },
            {
                "name": "dropoff_point",
                "values": [rng.choice(["downtown_park", "expo_plaza", "riverside_mall"])],
            },
            {
                "name": "arrived_on",
                "values": [f"202{rng.randint(4, 6)}-0{rng.randint(1, 9)}-08"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_TEAM_ASSESSMENT_EVENT": lambda rng: {
        "table": f"team_assessment_{rng.randint(100, 999)}",
        "columns": [
            {
                "name": "assessment_event_id",
                "values": [f"TA-20{rng.randint(0, 9)}{rng.randint(10, 99)}-{rng.randint(10, 99)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_TEAM_ASSESSMENT_EVENT",
            },
            {
                "name": "participant_role",
                "values": [rng.choice(["lead_engineer", "designer", "qa_specialist"])],
            },
            {
                "name": "outcome_grade",
                "values": [rng.choice(["pass", "conditional_pass", "fail"])],
            },
            {
                "name": "scored_on",
                "values": [f"202{rng.randint(4, 6)}-0{rng.randint(1, 9)}-11"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_TELE_SALES_ACTIVITY": lambda rng: {
        "table": f"telesales_dialer_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "call_activity",
                "values": [rng.choice(["outbound_pitch", "order_processing_call", "closing_negotiation"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_TELE_SALES_ACTIVITY",
            },
            {
                "name": "dialer_queue",
                "values": [f"Q{rng.randint(1, 4)}-{rng.randint(10, 99)}"],
            },
            {
                "name": "talk_seconds",
                "values": [f"{rng.randint(60, 900)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_TOP_WORKPLACE_AWARD": lambda rng: {
        "table": f"employer_recognition_docket_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "conferred_award",
                "values": [rng.choice(["top_workplace_2024", "wellness_benchmark_2025", "health_standard_2026"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_TOP_WORKPLACE_AWARD",
            },
            {
                "name": "recognizing_body",
                "values": [f"RBO-{rng.randint(100, 999)}"],
            },
            {
                "name": "verified_on",
                "values": [f"20{rng.randint(24, 26)}-09-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_TRADITIONAL_RELIGIOUS_HEALING_PROCESS": lambda rng: {
        "table": f"healing_ceremony_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "healing_rite",
                "values": [rng.choice(["prayer_vigil", "sacred_smoke_ritual", "chanting_ceremony"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_TRADITIONAL_RELIGIOUS_HEALING_PROCESS",
            },
            {
                "name": "participant_roll",
                "values": [f"{rng.randint(1, 9)}"],
            },
            {
                "name": "held_at",
                "values": [rng.choice(["chapel", "home_shrine", "community_hall"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_URINE_DRUG_SCREENING_PROCEDURE": lambda rng: {
        "table": f"clinical_specimen_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "screening_procedure",
                "values": [rng.choice(["urine_immunoassay_panel", "urine_confirmatory_test", "urine_spot_screen"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_URINE_DRUG_SCREENING_PROCEDURE",
            },
            {
                "name": "analyte_target",
                "values": [f"AN-{rng.randint(100, 999)}"],
            },
            {
                "name": "specimen_drawn",
                "values": [f"20{rng.randint(24, 26)}-06-0{rng.randint(1, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_VARIATION_MARGIN_RULE_SET": lambda rng: {
        "table": f"margin_rules_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "margin_rule_set_id",
                "values": [f"VMR-{rng.randint(1000, 9999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_VARIATION_MARGIN_RULE_SET",
            },
            {
                "name": "mandated_exchange",
                "values": [rng.choice(["daily_collateral", "netting_sweep"])],
            },
            {
                "name": "verified_by",
                "values": [rng.choice(["compliance_audit", "monthly_audit"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_VECTOR_CONTROL_ACTIVITY": lambda rng: {
        "table": f"vector_control_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "vector_control_activity_id",
                "values": [f"VCA-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_VECTOR_CONTROL_ACTIVITY",
            },
            {
                "name": "program_code",
                "values": [rng.choice(["fog_a", "larvae_b", "fog_c"])],
            },
            {
                "name": "target_street",
                "values": [rng.choice(["oak_row", "cedar_lane", "elm_walk"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_VENUE_OPERATIONS_OVERSIGHT": lambda rng: {
        "table": f"venue_oversight_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "oversight_case_id",
                "values": [f"VOS-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_VENUE_OPERATIONS_OVERSIGHT",
            },
            {
                "name": "facility_infra",
                "values": [rng.choice(["roof_deck", "sprinkler_loop", "elevator_bank"])],
            },
            {
                "name": "enforced_protocol",
                "values": [rng.choice(["night_close", "fire_lane_clear"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_WATER_QUALITY_MONITORING_ACTIVITY": lambda rng: {
        "table": f"water_quality_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "monitoring_run_id",
                "values": [f"WQM-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#PROCESS_WATER_QUALITY_MONITORING_ACTIVITY",
            },
            {
                "name": "standard_ref",
                "values": [rng.choice(["dh_std_41", "dh_std_52", "dh_std_63"])],
            },
            {
                "name": "parameter",
                "values": [rng.choice(["chlorine_ppm", "ph_level", "turbidity_ntu"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_WEIGHT_REDUCTION_SERVICE_EXECUTION": lambda rng: {
        "table": f"weight_reduction_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_execution",
                "values": [rng.choice(["dietary_advice_visit", "meal_plan_review", "nutrition_follow_up"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_WEIGHT_REDUCTION_SERVICE_EXECUTION",
            },
            {
                "name": "patient_id",
                "values": [f"P{rng.randint(1000, 9999)}"],
            },
            {
                "name": "executed_on",
                "values": [f"202{rng.randint(3, 6)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#PROCESS_YOUNG_RESEARCHER_RESEARCH_PROCESS": lambda rng: {
        "table": f"youth_research_log_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "research_process",
                "values": [rng.choice(["field_data_collection", "lab_bench_work", "survey_interview_round"])],
                "iri": "https://signals.zndx.org/sdg#PROCESS_YOUNG_RESEARCHER_RESEARCH_PROCESS",
            },
            {
                "name": "student_id",
                "values": [f"YS{rng.randint(100, 999)}"],
            },
            {
                "name": "supervisor",
                "values": [rng.choice(["dr_okafor", "dr_lindqvist", "dr_park"])],
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(2, 6)}-0{rng.randint(1, 9)}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_CALL_VOLUME": lambda rng: {
        "table": f"call_center_daily_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "call_volume",
                "values": [f"{rng.randint(45, 620)}"],
                "iri": "https://signals.zndx.org/sdg#QUALITY_CALL_VOLUME",
            },
            {
                "name": "queue",
                "values": [rng.choice(["billing", "support", "sales"])],
            },
            {
                "name": "period_start",
                "values": [f"202{rng.randint(4, 6)}-0{rng.randint(1, 9)}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_COMMUNICATION_SECURITY_ASPECT": lambda rng: {
        "table": f"channel_security_review_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "security_aspect",
                "values": [rng.choice(["tls_cipher_suite", "message_signing", "key_rotation", "channel_authentication"])],
                "iri": "https://signals.zndx.org/sdg#QUALITY_COMMUNICATION_SECURITY_ASPECT",
            },
            {
                "name": "channel",
                "values": [rng.choice(["sip_trunk", "mobile_radio", "fiber_link"])],
            },
            {
                "name": "assessed_on",
                "values": [f"202{rng.randint(3, 6)}-0{rng.randint(1, 9)}-22"],
            },
            {
                "name": "residual_risk",
                "values": [rng.choice(["low", "medium", "high"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_CORPORATE_REPUTATION": lambda rng: {
        "table": f"stakeholder_survey_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "stakeholder_role",
                "values": [rng.choice(["investor", "regulator", "supplier", "customer"])],
            },
            {
                "name": "view_of_organization",
                "values": [rng.choice(["favorable", "mixed", "damaged", "neutral"])],
                "iri": "https://signals.zndx.org/sdg#QUALITY_CORPORATE_REPUTATION",
            },
            {
                "name": "surveyed_on",
                "values": [f"202{rng.randint(0, 6)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_FINANCIAL_RESOURCE": lambda rng: {
        "table": f"capital_allocation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "applied_value",
                "values": [f"{rng.randint(1, 900)}000.00"],
                "iri": "https://signals.zndx.org/sdg#QUALITY_FINANCIAL_RESOURCE",
            },
            {
                "name": "activity",
                "values": [rng.choice(["operational", "strategic", "reserve"])],
            },
            {
                "name": "released_on",
                "values": [f"202{rng.randint(0, 6)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_HOUSING_DEBT_ISSUE": lambda rng: {
        "table": f"rental_debt_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "lease_origin",
                "values": [rng.choice(["residential", "commercial", "mixed_use"])],
            },
            {
                "name": "lease_value",
                "values": [f"{rng.randint(5, 99)}000.00"],
                "iri": "https://signals.zndx.org/sdg#QUALITY_HOUSING_DEBT_ISSUE",
            },
            {
                "name": "issued_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_MARKET_DEMAND": lambda rng: {
        "table": f"demand_snapshot_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "instrument",
                "values": [rng.choice(["equity", "commodity"])],
            },
            {
                "name": "willingness_index",
                "values": [f"{rng.randint(5, 99)}.{rng.randint(0, 99):02d}"],
                "iri": "https://signals.zndx.org/sdg#QUALITY_MARKET_DEMAND",
            },
            {
                "name": "measured_on",
                "values": [f"202{rng.randint(0, 6)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_MARKET_TREND": lambda rng: {
        "table": f"equity_trend_readings_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "trend_direction",
                "values": [rng.choice(["uptrend", "downtrend", "sideways"])],
                "iri": "https://signals.zndx.org/sdg#QUALITY_MARKET_TREND",
            },
            {
                "name": "symbol",
                "values": [f"EQ{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_MONETARY_AMOUNT": lambda rng: {
        "table": f"ledger_balance_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "settled_amount",
                "values": [f"{rng.randint(100, 999)} {rng.choice(['USD', 'EUR', 'JPY'])}"],
                "iri": "https://signals.zndx.org/sdg#QUALITY_MONETARY_AMOUNT",
            },
            {
                "name": "posted_on",
                "values": [f"202{rng.randint(0, 5)}-06-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_PDF_FORMAT": lambda rng: {
        "table": f"document_quality_audit_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "format_profile",
                "values": [rng.choice(["pdf-1.4", "pdf-1.7", "pdf-2.0"])],
                "iri": "https://signals.zndx.org/sdg#QUALITY_PDF_FORMAT",
            },
            {
                "name": "checked_on",
                "values": [f"202{rng.randint(0, 5)}-03-12"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_PRIM_AGE": lambda rng: {
        "table": f"entity_census_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "age_in_years",
                "values": [str(rng.randint(1, 90))],
                "iri": "https://signals.zndx.org/sdg#QUALITY_PRIM_AGE",
            },
            {
                "name": "observed_on",
                "values": [f"202{rng.randint(0, 5)}-09-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#QUALITY_QO_S_CONSTRAINT": lambda rng: {
        "table": f"service_quality_agreement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "qos_constraint",
                "values": [rng.choice(["latency_ceiling_250ms", "jitter_max_40ms", "packet_loss_below_1pct"])],
                "iri": "https://signals.zndx.org/sdg#QUALITY_QO_S_CONSTRAINT",
            },
            {
                "name": "service_name",
                "values": [f"stream_{rng.randint(10, 99)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ANIMAL_CONTROL_OFFICER_ROLE": lambda rng: {
        "table": f"animal_enforcement_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "officer_role",
                "values": [rng.choice(["primary", "secondary", "supervisory"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ANIMAL_CONTROL_OFFICER_ROLE",
            },
            {
                "name": "enforcement_district",
                "values": [f"district_{rng.randint(1, 12)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ATHLETIC_ATTENDANT_ROLE": lambda rng: {
        "table": f"athletic_monitoring_session_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "attendant_role",
                "values": [rng.choice(["lead", "assistant", "relief"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ATHLETIC_ATTENDANT_ROLE",
            },
            {
                "name": "session_date",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_AUTHORIZED_REPRESENTATIVE_ROLE": lambda rng: {
        "table": f"entity_authorization_grant_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "representative_role",
                "values": [rng.choice(["signatory", "co_signatory", "alternate"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_AUTHORIZED_REPRESENTATIVE_ROLE",
            },
            {
                "name": "mandate_reference",
                "values": [f"mnd_{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_BENEFICIARY_DESIGNATION": lambda rng: {
        "table": f"beneficiary_designation_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "designation_category",
                "values": [rng.choice(["spouse", "child", "estate", "trust"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_BENEFICIARY_DESIGNATION",
            },
            {
                "name": "plan_code",
                "values": [f"RB{rng.randint(1000, 9999)}"],
            },
            {
                "name": "effective_on",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_BOARD_MEMBER": lambda rng: {
        "table": f"board_seat_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "seat_type",
                "values": [rng.choice(["elected", "appointed", "ex_officio"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_BOARD_MEMBER",
            },
            {
                "name": "org_code",
                "values": [f"ORG{rng.randint(100, 999)}"],
            },
            {
                "name": "term_start",
                "values": [f"20{rng.randint(20, 29)}-01-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_BOARD_MEMBER_ROLE": lambda rng: {
        "table": f"role_realization_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "realization_kind",
                "values": [rng.choice(["meeting_attendance", "committee_vote", "quorum_presencing"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_BOARD_MEMBER_ROLE",
            },
            {
                "name": "agent_ref",
                "values": [f"AGT{rng.randint(10000, 99999)}"],
            },
            {
                "name": "realized_on",
                "values": [f"202{rng.randint(0, 9)}-11-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_BOARD_OF_AUDITORS_ROLE": lambda rng: {
        "table": f"audit_engagement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "engagement_scope",
                "values": [rng.choice(["financial_statements", "internal_controls", "operational_processes"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_BOARD_OF_AUDITORS_ROLE",
            },
            {
                "name": "engagement_ref",
                "values": [f"AUD{rng.randint(100, 999)}-{rng.randint(10, 99)}"],
            },
            {
                "name": "fieldwork_start",
                "values": [f"20{rng.randint(20, 29)}-02-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_B_CORPORATION_MEMBER": lambda rng: {
        "table": f"b_corporation_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "membership_tier",
                "values": [rng.choice(["certified", "beneficial", "incubator"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_B_CORPORATION_MEMBER",
            },
            {
                "name": "joined_on",
                "values": [f"20{rng.randint(15, 25)}-0{rng.randint(1, 9)}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_CARDIOTHORACIC_RADIOLOGIST_ROLE": lambda rng: {
        "table": f"diagnostic_imaging_staff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "subspecialty",
                "values": [rng.choice(["cardiac_thoracic", "abdominal", "mammography"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_CARDIOTHORACIC_RADIOLOGIST_ROLE",
            },
            {
                "name": "oncall_rotation",
                "values": [rng.choice(["primary", "backup", "off"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_CHIEF_COMPLIANCE_OFFICER_ROLE": lambda rng: {
        "table": f"compliance_governance_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "officer_scope",
                "values": [rng.choice(["regulatory", "internal_controls", "privacy"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_CHIEF_COMPLIANCE_OFFICER_ROLE",
            },
            {
                "name": "appointed_on",
                "values": [f"20{rng.randint(10, 25)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_CHIEF_FINANCIAL_OFFICER_ROLE": lambda rng: {
        "table": f"financial_management_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "oversight_area",
                "values": [rng.choice(["treasury", "reporting", "capital_planning"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_CHIEF_FINANCIAL_OFFICER_ROLE",
            },
            {
                "name": "fiscal_year",
                "values": [f"20{rng.randint(21, 25)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_CLEARING_BROKER_ROLE": lambda rng: {
        "table": f"clearing_broker_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "broker_role",
                "values": [rng.choice(["direct", "prime", "dual"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_CLEARING_BROKER_ROLE",
            },
            {
                "name": "customer_account_id",
                "values": [f"ACCT-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "effective_on",
                "values": [f"201{rng.randint(0, 9)}-03-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_COLLEGE_OFFICIAL_ROLE": lambda rng: {
        "table": f"college_official_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "official_role",
                "values": [rng.choice(["dean", "provost", "registrar", "librarian"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_COLLEGE_OFFICIAL_ROLE",
            },
            {
                "name": "official_id",
                "values": [f"STF-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "appointed_on",
                "values": [f"201{rng.randint(0, 9)}-07-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_COMMUNITY_LEARNING_DISABILITY_NURSE": lambda rng: {
        "table": f"community_health_role_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "agent_role",
                "values": [rng.choice(["school", "clinic", "mobile", "home"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_COMMUNITY_LEARNING_DISABILITY_NURSE",
            },
            {
                "name": "healthcare_agent_id",
                "values": [f"NRS-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "started_on",
                "values": [f"201{rng.randint(0, 9)}-09-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_COMPLIANCE_COORDINATOR_ROLE": lambda rng: {
        "table": f"compliance_coordination_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "coordination_role",
                "values": [rng.choice(["academic", "financial", "enrollment"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_COMPLIANCE_COORDINATOR_ROLE",
            },
            {
                "name": "academic_record_id",
                "values": [f"REC-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "eligibility_check_on",
                "values": [f"201{rng.randint(0, 9)}-05-20"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_DATA_CONTROLLER_ROLE": lambda rng: {
        "table": f"processing_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "controller_assignment",
                "values": [rng.choice(["sole", "joint"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_DATA_CONTROLLER_ROLE",
            },
            {
                "name": "purpose_determined",
                "values": [rng.choice(["marketing", "service_delivery", "research"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_DEVELOPER_REPRESENTATIVE_ROLE": lambda rng: {
        "table": f"final_inspection_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "representative_tier",
                "values": [rng.choice(["lead", "alternate"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_DEVELOPER_REPRESENTATIVE_ROLE",
            },
            {
                "name": "report_ref",
                "values": [f"FI-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_DEVICE_EXPERT_ROLE": lambda rng: {
        "table": f"procurement_advisory_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advisory_specialty",
                "values": [rng.choice(["clinical", "technical", "ergonomic"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_DEVICE_EXPERT_ROLE",
            },
            {
                "name": "advisory_batch",
                "values": [f"AB-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_DISABLED_ADULT_PARENT_ROLE": lambda rng: {
        "table": f"adult_parenting_support_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "support_stream",
                "values": [rng.choice(["respite", "direct_care", "residential"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_DISABLED_ADULT_PARENT_ROLE",
            },
            {
                "name": "enrolled_on",
                "values": [f"20{rng.randint(24, 25)}-{rng.randint(1, 9):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_DISTRICT_EMPLOYEE": lambda rng: {
        "table": f"district_staff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "employee_role",
                "values": [rng.choice(["ward_liaison", "permit_clerk", "case_worker"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_DISTRICT_EMPLOYEE",
            },
            {
                "name": "district_code",
                "values": [rng.choice(["d04", "d09", "d12"])],
            },
            {
                "name": "appointed_on",
                "values": [f"201{rng.randint(0, 9)}-03-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ELIGIBLE_RAFFLE_CONDUCTOR_ROLE": lambda rng: {
        "table": f"raffle_stewards_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "conductor_role",
                "values": [rng.choice(["certified", "provisional", "assistant"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ELIGIBLE_RAFFLE_CONDUCTOR_ROLE",
            },
            {
                "name": "organization_id",
                "values": [f"np_{rng.randint(100, 999)}"],
            },
            {
                "name": "draw_date",
                "values": [f"201{rng.randint(0, 9)}-11-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ENTREPRENEUR_ROLE": lambda rng: {
        "table": f"venture_owners_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "owner_role",
                "values": [rng.choice(["sole_proprietor", "founder", "co_owner"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ENTREPRENEUR_ROLE",
            },
            {
                "name": "agent_id",
                "values": [f"agent_{rng.randint(100, 999)}"],
            },
            {
                "name": "registered_on",
                "values": [f"201{rng.randint(0, 9)}-06-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_DIRECTOR_FUNCTION": lambda rng: {
        "table": f"board_appointments_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "appointee_function",
                "values": [rng.choice(["chief_officer", "operations_lead", "program_director"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_DIRECTOR_FUNCTION",
            },
            {
                "name": "organization_id",
                "values": [f"org_{rng.randint(100, 999)}"],
            },
            {
                "name": "term_start",
                "values": [f"201{rng.randint(0, 9)}-01-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_DIRECTOR_ROLE": lambda rng: {
        "table": f"org_charter_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "director_seat",
                "values": [rng.choice(["presiding", "administrative", "acting"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_DIRECTOR_ROLE",
            },
            {
                "name": "charter_no",
                "values": [f"CH-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_OFFICER_ROLE": lambda rng: {
        "table": f"officer_register_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "officer_seat",
                "values": [rng.choice(["chief_executive", "chief_financial", "chief_operations"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_OFFICER_ROLE",
            },
            {
                "name": "appointment_no",
                "values": [f"APT-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_TEAM_ROSTER": lambda rng: {
        "table": f"leadership_cycle_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "roster_phase",
                "values": [rng.choice(["charter", "succession", "renewal"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXECUTIVE_TEAM_ROSTER",
            },
            {
                "name": "cycle_started",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXPERIENCED_MENTOR": lambda rng: {
        "table": f"mentorship_pairing_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "mentor_tier",
                "values": [rng.choice(["senior", "principal", "distinguished"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXPERIENCED_MENTOR",
            },
            {
                "name": "mentoring_started",
                "values": [f"201{rng.randint(0, 9)}-06-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_EXTERNAL_BUSINESS_PARTNER": lambda rng: {
        "table": f"market_exchange_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "partner_type",
                "values": [rng.choice(["consultant", "supplier", "reseller"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_EXTERNAL_BUSINESS_PARTNER",
            },
            {
                "name": "exchange_status",
                "values": [rng.choice(["ongoing", "terminated", "proposed"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_FACULTY_MEMBER_ROLE": lambda rng: {
        "table": f"academic_activity_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "academic_role",
                "values": [rng.choice(["lecturer", "tutor", "researcher"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_FACULTY_MEMBER_ROLE",
            },
            {
                "name": "term_start",
                "values": [f"202{rng.randint(0, 5)}-09-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_GENERAL_MANAGER_ROLE": lambda rng: {
        "table": f"store_management_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "management_role",
                "values": [rng.choice(["shift_lead", "branch_head", "operations_head"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_GENERAL_MANAGER_ROLE",
            },
            {
                "name": "unit_code",
                "values": [f"unit-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_INDIGENOUS_CHILD_ROLE": lambda rng: {
        "table": f"community_registration_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "child_role",
                "values": [rng.choice(["apprentice", "participant", "beneficiary"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_INDIGENOUS_CHILD_ROLE",
            },
            {
                "name": "enrollment_year",
                "values": [f"201{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_INVESTOR_RELATIONS_DIRECTOR_ROLE": lambda rng: {
        "table": f"officer_mandate_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "relations_lead",
                "values": [rng.choice(["earnings_call", "analyst_day", "shareholder_meeting"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_INVESTOR_RELATIONS_DIRECTOR_ROLE",
            },
            {
                "name": "mandate_ref",
                "values": [f"IR-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_JOINT_BOARD_CHAIR_ROLE": lambda rng: {
        "table": f"board_tenure_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "board_chair",
                "values": [rng.choice(["joint", "co_chair", "shared"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_JOINT_BOARD_CHAIR_ROLE",
            },
            {
                "name": "tenure_ref",
                "values": [f"JC-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_JUVENILE_OFFENDER_PATIENT_ROLE": lambda rng: {
        "table": f"youth_care_episode_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "patient_type",
                "values": [rng.choice(["detainee", "probation", "remand"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_JUVENILE_OFFENDER_PATIENT_ROLE",
            },
            {
                "name": "episode_opened",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LAW_ENFORCEMENT_PERSONNEL": lambda rng: {
        "table": f"field_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "duty_role",
                "values": [rng.choice(["patrol", "investigation", "traffic"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LAW_ENFORCEMENT_PERSONNEL",
            },
            {
                "name": "shift_code",
                "values": [f"S{rng.randint(1, 8)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LEGAL_SERVICE_ADVERTISER_ROLE": lambda rng: {
        "table": f"advertisement_slot_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "advertiser",
                "values": [rng.choice(["counsel", "firm", "clinic", "solo_practitioner"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LEGAL_SERVICE_ADVERTISER_ROLE",
            },
            {
                "name": "placement_medium",
                "values": [rng.choice(["directory", "print", "broadcast", "web"])],
            },
            {
                "name": "ran_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LEGAL_TECH_SERVICE_PROVISION": lambda rng: {
        "table": f"service_engagement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "provider",
                "values": [rng.choice(["vendor", "lab", "partner", "subsidiary"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LEGAL_TECH_SERVICE_PROVISION",
            },
            {
                "name": "delivery_channel",
                "values": [rng.choice(["cloud", "on_premise", "hybrid"])],
            },
            {
                "name": "effective_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LIBRARY_STUDENT_ASSISTANT_ROLE": lambda rng: {
        "table": f"shift_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "assistant",
                "values": [rng.choice(["student_a", "student_b", "student_c", "student_d"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LIBRARY_STUDENT_ASSISTANT_ROLE",
            },
            {
                "name": "station",
                "values": [rng.choice(["circulation", "stacks", "reference_desk"])],
            },
            {
                "name": "shift_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LIFEGUARD_STAFF": lambda rng: {
        "table": f"duty_roster_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "duty_station",
                "values": [rng.choice(["north_tower", "south_tower", "shallow_end", "deep_end"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LIFEGUARD_STAFF",
            },
            {
                "name": "cert_level",
                "values": [rng.choice(["level_1", "level_2", "lead"])],
            },
            {
                "name": "shift_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_LOCAL_AUTHORITY": lambda rng: {
        "table": f"district_governance_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "jurisdiction_holder",
                "values": [rng.choice(["municipal_council", "county_administration", "parish_board"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_LOCAL_AUTHORITY",
            },
            {
                "name": "service_territory",
                "values": [f"sector_{rng.randint(1, 9)}"],
            },
            {
                "name": "mandate_started",
                "values": [f"201{rng.randint(0, 9)}-07-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_MEMBER": lambda rng: {
        "table": f"group_membership_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "member_category",
                "values": [rng.choice(["full", "associate", "affiliate"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_MEMBER",
            },
            {
                "name": "benefit_tier",
                "values": [f"tier_{rng.randint(1, 4)}"],
            },
            {
                "name": "roster_updated",
                "values": [f"201{rng.randint(0, 9)}-11-02"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_MUNICIPAL_SWIMMING_POOL_ASSISTANT_MANAGER": lambda rng: {
        "table": f"pool_operations_assignment_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "assistant_manager_post",
                "values": [rng.choice(["deck_supervisor", "lane_scheduler", "facility_coordinator"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_MUNICIPAL_SWIMMING_POOL_ASSISTANT_MANAGER",
            },
            {
                "name": "pool_site",
                "values": [f"site_{rng.randint(1, 9)}"],
            },
            {
                "name": "shift_starts",
                "values": [f"0{rng.randint(6, 9)}:00"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ORGANIZATIONAL_FUNCTION": lambda rng: {
        "table": f"organization_capability_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "functional_scope",
                "values": [rng.choice(["operations", "finance", "workforce", "planning"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ORGANIZATIONAL_FUNCTION",
            },
            {
                "name": "directed_unit",
                "values": [f"unit_{rng.randint(1, 8)}"],
            },
            {
                "name": "charter_date",
                "values": [f"201{rng.randint(0, 9)}-03-20"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_ORGANIZATION_MEMBER": lambda rng: {
        "table": f"org_membership_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "membership_kind",
                "values": [rng.choice(["founding", "voting", "honorary"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_ORGANIZATION_MEMBER",
            },
            {
                "name": "joined_on",
                "values": [f"201{rng.randint(0, 9)}-06-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE": lambda rng: {
        "table": f"care_placement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "caregiver_role",
                "values": [rng.choice(["primary", "kinship", "foster"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE",
            },
            {
                "name": "started_on",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_PERFORMANCE_BASED_CARE_PROVIDER_ROLE": lambda rng: {
        "table": f"care_arrangement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "provider_kind",
                "values": [rng.choice(["capitation", "shared_savings", "episode_based"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PERFORMANCE_BASED_CARE_PROVIDER_ROLE",
            },
            {
                "name": "measure_window",
                "values": [f"202{rng.randint(4, 6)}Q{rng.randint(1, 4)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_PETITIONER_ROLE": lambda rng: {
        "table": f"adoption_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "filing_type",
                "values": [rng.choice(["joint", "sole", "stepparent"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PETITIONER_ROLE",
            },
            {
                "name": "case_opened_on",
                "values": [f"201{rng.randint(0, 9)}-08-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_POLICE_OFFICER_ROLE": lambda rng: {
        "table": f"patrol_post_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "officer_role",
                "values": [rng.choice(["foot", "vehicle", "mounted", "detail"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_POLICE_OFFICER_ROLE",
            },
            {
                "name": "shift_started_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"],
            },
            {
                "name": "beat_zone",
                "values": [f"BZ{rng.randint(1, 9)}{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_PRIM_CONTINUOUSMODE": lambda rng: {
        "table": f"sensor_watch_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "op_mode",
                "values": [rng.choice(["rolling", "steady", "24x7"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PRIM_CONTINUOUSMODE",
            },
            {
                "name": "interval_started_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"],
            },
            {
                "name": "watch_channel",
                "values": [f"CH-{rng.randint(1, 4)}{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_PROPERTY_OWNER": lambda rng: {
        "table": f"title_record_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "owner_role",
                "values": [rng.choice(["sole", "joint", "trust"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PROPERTY_OWNER",
            },
            {
                "name": "parcel_ref",
                "values": [f"PRC-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "title_noted_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_QUALIFIED_SOURCE_TESTING_INDIVIDUAL": lambda rng: {
        "table": f"source_test_run_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "tester_role",
                "values": [rng.choice(["certified", "accredited", "licensed"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_QUALIFIED_SOURCE_TESTING_INDIVIDUAL",
            },
            {
                "name": "sample_ref",
                "values": [f"SRC-{rng.randint(100, 999)}"],
            },
            {
                "name": "run_started_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_RECOGNITION_MANAGER_ROLE": lambda rng: {
        "table": f"authority_recognition_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "manager_rank",
                "values": [rng.choice(["senior", "interim", "deputy"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_RECOGNITION_MANAGER_ROLE",
            },
            {
                "name": "authority_unit",
                "values": [rng.choice(["licensing", "inspection", "appeals"])],
            },
            {
                "name": "appointed_on",
                "values": [f"201{rng.randint(0, 9)}-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_RECORD_AUTHORIZATION_ROLE": lambda rng: {
        "table": f"health_transfer_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "authorizer_party",
                "values": [rng.choice(["youth", "guardian"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_RECORD_AUTHORIZATION_ROLE",
            },
            {
                "name": "transfer_batch",
                "values": [f"tr-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "authorized_on",
                "values": [f"202{rng.randint(0, 5)}-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_REFERRAL_SOURCE_ROLE": lambda rng: {
        "table": f"referral_intake_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "source_division",
                "values": [rng.choice(["probation", "courts", "corrections"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_REFERRAL_SOURCE_ROLE",
            },
            {
                "name": "referral_case_id",
                "values": [f"ref-{rng.randint(10000, 99999)}"],
            },
            {
                "name": "referred_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(10, 12)}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_REGISTERED_MANAGER_ROLE": lambda rng: {
        "table": f"service_management_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "manager_class",
                "values": [rng.choice(["full", "acting", "deputised"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_REGISTERED_MANAGER_ROLE",
            },
            {
                "name": "managed_service",
                "values": [rng.choice(["day_services", "respite", "home_care"])],
            },
            {
                "name": "registered_on",
                "values": [f"200{rng.randint(0, 9)}-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_REGULATORY_AUTHORITY": lambda rng: {
        "table": f"regulation_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "regulator_authority",
                "values": [rng.choice(["federal", "state", "municipal"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_REGULATORY_AUTHORITY",
            },
            {
                "name": "rulebook_ref",
                "values": [f"RB-{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_RETAIL_LEADERSHIP_TEAM_MEMBER": lambda rng: {
        "table": f"retail_org_chart_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "leadership_member",
                "values": [rng.choice(["district_manager", "store_ops_director", "assistant_vp_retail"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_RETAIL_LEADERSHIP_TEAM_MEMBER",
            },
            {
                "name": "report_date",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_RETAINING_EMPLOYEE_ROLE": lambda rng: {
        "table": f"offboarding_transfer_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "retaining_role_holder",
                "values": [rng.choice(["manager", "supervisor", "department_lead"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_RETAINING_EMPLOYEE_ROLE",
            },
            {
                "name": "departing_employee_ref",
                "values": [f"EMP-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_SENIOR_ANIMAL_SERVICES_OFFICER_ROLE": lambda rng: {
        "table": f"animal_control_shift_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "animal_services_officer",
                "values": [rng.choice(["field_officer", "intake_officer", "patrol_officer"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_SENIOR_ANIMAL_SERVICES_OFFICER_ROLE",
            },
            {
                "name": "shift_started",
                "values": [f"201{rng.randint(0, 9)}-05-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_SERVICE_STAFF_ROLE": lambda rng: {
        "table": f"dining_staff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "service_role",
                "values": [rng.choice(["server", "host", "buss", "barback"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_SERVICE_STAFF_ROLE",
            },
            {
                "name": "started_on",
                "values": [f"202{rng.randint(1, 5)}-0{rng.randint(1, 9)}-15"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_SOCIAL_MEDIA_ACCOUNT_ADMINISTRATOR": lambda rng: {
        "table": f"district_media_staff_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "account_admin",
                "values": [rng.choice(["dwhit", "mcast", "jrey", "akim"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_SOCIAL_MEDIA_ACCOUNT_ADMINISTRATOR",
            },
            {
                "name": "channel",
                "values": [rng.choice(["usdae_official", "usdae_news", "usdae_events"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_STAFF_SOCIAL_MEDIA_ACCOUNT_ADMINISTRATOR_ROLE": lambda rng: {
        "table": f"staff_media_access_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "media_admin_role",
                "values": [rng.choice(["owner", "editor", "moderator"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_STAFF_SOCIAL_MEDIA_ACCOUNT_ADMINISTRATOR_ROLE",
            },
            {
                "name": "staff_number",
                "values": [f"SM{rng.randint(100, 999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_STAGE_5_CHECKLIST": lambda rng: {
        "table": f"permit_inspection_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "checklist_ref",
                "values": [f"S5-{rng.randint(100, 999)}"],
                "iri": "https://signals.zndx.org/sdg#ROLE_STAGE_5_CHECKLIST",
            },
            {
                "name": "items_passed",
                "values": [f"{rng.randint(8, 24)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_STUDENT_CONDUCT_ADMINISTRATOR_ROLE": lambda rng: {
        "table": f"conduct_case_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "admin_designation",
                "values": [rng.choice(["case_officer", "dean_reviewer", "board_adjudicator"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_STUDENT_CONDUCT_ADMINISTRATOR_ROLE",
            },
            {
                "name": "case_ref",
                "values": [f"SC-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "reviewed_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_STUDENT_ENTREPRENEUR_ROLE": lambda rng: {
        "table": f"student_venture_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "venture_role",
                "values": [rng.choice(["founder", "cofounder", "student_lead"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_STUDENT_ENTREPRENEUR_ROLE",
            },
            {
                "name": "venture_ref",
                "values": [f"SV-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "founded_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_TELEMARKETER_SALES_ADMINISTRATOR_ROLE": lambda rng: {
        "table": f"call_campaign_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "sales_admin_designation",
                "values": [rng.choice(["call_schedule_admin", "lead_list_admin", "dialer_console_admin"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_TELEMARKETER_SALES_ADMINISTRATOR_ROLE",
            },
            {
                "name": "campaign_ref",
                "values": [f"TC-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "assigned_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_TRADITIONAL_RELIGIOUS_HEALER_ROLE": lambda rng: {
        "table": f"healing_ceremony_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "healer_designation",
                "values": [rng.choice(["temple_healer", "medicine_keeper", "ceremony_master"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_TRADITIONAL_RELIGIOUS_HEALER_ROLE",
            },
            {
                "name": "ceremony_ref",
                "values": [f"TH-{rng.randint(1000, 9999)}"],
            },
            {
                "name": "attended_on",
                "values": [f"201{rng.randint(0, 9)}-{rng.randint(1, 12):02d}-01"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_TYPED_REALIZATION_CHANNEL": lambda rng: {
        "table": f"oncall_shift_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "coverage_role",
                "values": [rng.choice(["primary", "secondary", "escalation", "fallback"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_TYPED_REALIZATION_CHANNEL",
            },
            {
                "name": "shift_started_on",
                "values": [f"2025-0{rng.randint(1, 9)}-01"],
            },
            {
                "name": "paging_channel",
                "values": [rng.choice(["page_one", "page_two", "page_three"])],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_UNACCOMPANIED_MINOR_ROLE": lambda rng: {
        "table": f"taxi_ride_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "passenger_role",
                "values": [rng.choice(["unsupervised", "chaperoned", "escorted"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_UNACCOMPANIED_MINOR_ROLE",
            },
            {
                "name": "ride_started_on",
                "values": [f"2025-0{rng.randint(1, 9)}-14"],
            },
            {
                "name": "vehicle_id",
                "values": [f"TX-{rng.randint(1000, 9999)}"],
            },
        ],
    },
    "https://signals.zndx.org/sdg#ROLE_YOUNG_RESEARCHER_ROLE": lambda rng: {
        "table": f"research_participant_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "researcher_role",
                "values": [rng.choice(["student_investigator", "junior_fellow", "trainee"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_YOUNG_RESEARCHER_ROLE",
            },
            {
                "name": "lab_code",
                "values": [f"YR-{rng.randint(100, 999)}"],
            },
            {
                "name": "enrolled_on",
                "values": [f"2024-0{rng.randint(2, 9)}-01"],
            },
        ],
    },
}
