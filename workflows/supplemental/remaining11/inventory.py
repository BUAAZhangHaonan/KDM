"""Frozen supplemental model roster and independent-attempt task definition. No inventory/transfer job."""

from __future__ import annotations

from typing import Any, Iterator

MODELS = (
    "gemma3_12b", "glm46v", "internvl35_8b", "llava15_7b", "onevision",
    "minicpm45", "phi35", "qwen3vl", "qwen35_9b", "llava15_13b",
    "llava16_vicuna",
)

CENSUS_PREFIX = (
    "outputs/annotations/luna_census_remaining_v1/"
    "remaining108_plus2_20260924/census_merged_v1/"
)

def independent_task(sample: dict[str, Any], replicate: int) -> dict[str, Any]:
    return {"sample": sample, "method": "direct", "marker": "UNKNOWN",
            "reference_marker": "UNKNOWN", "guided": False,
            "reference_guided": False, "attempt": True, "replicate": replicate,
            "kind": "independent_attempt"}
