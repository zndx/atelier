"""Exact IRI value generators for ontology terms.

The coding harness writes GENERATORS. insitu_corpus applies an entry
only when the entity IRI equals the key.
"""

import random

GENERATORS = {
    "https://signals.zndx.org/sdg#PROCESS_CLINICAL_QUALITY_IMPROVEMENT_PROGRAM": lambda rng: rng.choice(["planned", "active", "completed"]),
    "https://signals.zndx.org/sdg#PROCESS_RESPONSIVE_WEBSITE_DEVELOPMENT": lambda rng: f"RWD-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#ICE_DONOR_PROFILE": lambda rng: f"DP-{rng.randint(10000, 99999)}",
    "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE": lambda rng: f"PR-{rng.randint(100, 999)}",
    "https://signals.zndx.org/sdg#ROLE_INDIGENOUS_CHILD_ROLE": lambda rng: f"ICR-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#ROLE_POLICE_OFFICER_ROLE": lambda rng: f"POR-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#PROCESS_PDF_EXPORT_PROCESS": lambda rng: f"PEXP-{rng.randint(10000, 99999)}",
    "https://signals.zndx.org/sdg#ROLE_SENIOR_ANIMAL_SERVICES_OFFICER_ROLE": lambda rng: f"ASO-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#PROCESS_SPECIFICATION_TESTING_EVENT": lambda rng: f"STE-{rng.randint(10000, 99999)}",
    "https://signals.zndx.org/sdg#PROCESS_MATHEMATICAL_LEARNING_PROCESS": lambda rng: f"MLP-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#PROCESS_PROTECTED_DISCLOSURE_PROCESS": lambda rng: rng.choice(["draft", "filed", "under review", "closed"]),
    "https://signals.zndx.org/sdg#PROCESS_IMPROPER_CONDUCT_EVENT": lambda rng: f"ICD-{rng.randint(1000, 9999)}",
    "https://signals.zndx.org/sdg#PROCESS_CREDIT_HOUR_CERTIFICATION_PROCESS": lambda rng: f"{rng.randint(3, 18)}.{rng.choice(['0', '5'])}",
}
