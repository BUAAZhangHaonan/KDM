#!/usr/bin/env python3
"""Export existing formal runtime, prompt/config variants, and cache availability."""

from __future__ import annotations

import csv
import gzip
import json
import math
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_20260929"
ARCHIVE = Path(
    "/home/g203-4028/projects/knowledge-deficit-mitigation-archive/20260929"
)
MANIFEST_PATH = "data/responses/formal/manifest.json"
CONDITIONS_PATH = "outputs/analysis/main_results/condition_metrics.csv"
ARCHIVE_MANIFEST = ARCHIVE / "historical_outputs_manifest.json"
ARCHIVE_PATH = ARCHIVE / "historical_outputs.tar"
SCALARS = "outputs/analysis/panel5_mechanism_path_scalars_20260928_v1"
INVENTORY_PATH = (
    "outputs/records/acceleration_v4/"
    "panel5_mechanism_trace_inventory_20260928_v2.json"
)
DRYRUN_PATH = (
    "outputs/records/acceleration_v4/"
    "mechanism_manifest_runner_v1_dryrun_20260928_v4/dry_run_receipt.json"
)
CONDITION_KEYS = (
    "model", "method", "kind", "marker", "reference_marker",
    "guided", "reference_guided", "replicate",
)
OPTIONAL_RUNTIME = (
    "batch_size", "prefill_s", "prefill_seconds", "decode_s", "decode_seconds",
    "peak_memory_allocated", "peak_memory_bytes", "peak_gpu_memory_mb",
    "max_memory_allocated", "cache_hits", "cache_misses", "cache_bytes",
)
PROBABILITY_FIELDS = (
    "selected_log_probabilities", "sequence_log_probability", "first_probability",
)
WALL_SCOPE = (
    "Existing per-answer wall_s: perf_counter immediately before image loading "
    "through image conversion, prompt/session construction, generation and session "
    "cleanup; measured before Ledger.add serialization. Model loading and ledger "
    "writing are outside this interval. Prefill/decode are not separated."
)
TOKEN_SCOPE = (
    "len(tokens) from the saved token-ID list, including any saved stop/EOS token; "
    "no retokenization. sum/mean use only rows with a saved list."
)
FILTER_SCOPE = (
    "Exact eight-field condition tuple AND config JSON equality AND identity equality, "
    "within the manifest's ordered source segment. first/last lines bound a potentially "
    "noncontiguous set; filter_json identifies the exact included rows."
)


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive_uri(path):
    return f"{ARCHIVE_PATH}::{path}"


def condition_tuple(row, csv_input=False):
    values = []
    for key in CONDITION_KEYS:
        value = row[key]
        if key in ("guided", "reference_guided"):
            if csv_input:
                if value not in ("True", "False"):
                    raise ValueError(f"Unexpected boolean {value!r}")
                value = value == "True"
            elif type(value) is not bool:
                raise ValueError(f"Unexpected raw boolean {value!r}")
        if key == "replicate":
            value = int(value)
        values.append(value)
    return tuple(values)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty export: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def flatten_named(value, wanted, prefix="$"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            loc = f"{prefix}.{key}"
            if key in wanted:
                found.append({"key": loc, "value": item})
            found.extend(flatten_named(item, wanted, loc))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(flatten_named(item, wanted, f"{prefix}[{index}]"))
    return found


class Variants:
    def __init__(self, label):
        self.label = label
        self.indices = {}
        self.records = []

    def add(self, value, source, line, raw_line, condition_id):
        encoded = compact(value)
        if encoded not in self.indices:
            variant_id = len(self.records) + 1
            self.indices[encoded] = variant_id
            self.records.append({
                f"{self.label}_id": variant_id,
                "value": value,
                "count": 0,
                "condition_ids": set(),
                "locations": {},
            })
        variant_id = self.indices[encoded]
        item = self.records[variant_id - 1]
        item["count"] += 1
        item["condition_ids"].add(condition_id)
        loc = item["locations"].setdefault(source["source_id"], {
            "source_id": source["source_id"],
            "source_path": source["source_path"],
            "consolidated_path": source["consolidated_path"],
            "first_line": raw_line,
            "last_line": raw_line,
            "first_consolidated_line": line,
            "last_consolidated_line": line,
            "count": 0,
        })
        loc["last_line"] = raw_line
        loc["last_consolidated_line"] = line
        loc["count"] += 1
        return variant_id

    def export(self):
        result = []
        for record in self.records:
            item = dict(record)
            item["condition_ids"] = sorted(item["condition_ids"])
            item["locations"] = list(item["locations"].values())
            result.append(item)
        return result


def main():
    manifest = load_json(ROOT / MANIFEST_PATH)
    archive_manifest = load_json(ARCHIVE_MANIFEST)
    archive_entries = {
        row["path"]: (index, row)
        for index, row in enumerate(archive_manifest["files"])
    }
    with (ROOT / CONDITIONS_PATH).open(encoding="utf-8", newline="") as handle:
        condition_rows = list(csv.DictReader(handle))
    expected = {condition_tuple(row, True): int(row["n"]) for row in condition_rows}
    if len(expected) != len(condition_rows):
        raise ValueError("Duplicate condition in authoritative metrics")
    conditions = {
        key: {"condition_id": index, **dict(zip(CONDITION_KEYS, key))}
        for index, key in enumerate(sorted(expected), 1)
    }

    read_members = []
    sources = []
    model_sources = defaultdict(list)
    bound_jsons = {}
    with tarfile.open(ARCHIVE_PATH, "r:") as archive:
        members = {member.name: member for member in archive.getmembers()}

        def read_member(path):
            member = members[path]
            if member.size > 10_000_000:
                raise ValueError(f"Unexpected metadata size: {path}")
            read_members.append(path)
            return archive.extractfile(member).read()

        def read_archived_json(path):
            data = json.loads(read_member(path))
            bound_jsons[path] = data
            return data

        for model_index, model in enumerate(manifest["models"]):
            first = 1
            for segment_index, entry in enumerate(model["sources"]):
                source_id = len(sources) + 1
                source_path = entry["path"]
                parent = Path(source_path).parent.as_posix()
                sidecar_path = f"{parent}/formal.identity.json"
                sidecar = read_archived_json(sidecar_path)
                definition = sidecar["definition"]
                execution = definition.get("execution", {})
                backend = definition.get("backend", {})
                version_key = (
                    "$.definition.execution.runtime_versions"
                    if execution.get("runtime_versions") is not None
                    else "$.definition.backend.versions"
                )
                versions = execution.get("runtime_versions", backend.get("versions"))
                environment_key = (
                    "$.definition.execution.runtime_environment_python"
                    if execution.get("runtime_environment_python") is not None
                    else "$.definition.backend.environment_python"
                )
                source = {
                    "source_id": source_id,
                    "model": model["model"],
                    "source_path": source_path,
                    "source_archive_path": archive_uri(source_path),
                    "source_recorded_bytes": entry["bytes"],
                    "source_recorded_sha256": entry["sha256"],
                    "consolidated_path": model["output_path"],
                    "consolidated_recorded_bytes": model["output_bytes"],
                    "manifest_locator": (
                        f"{MANIFEST_PATH}#$.models[{model_index}].sources[{segment_index}]"
                    ),
                    "segment_first_line": first,
                    "segment_last_line": first + entry["rows"] - 1,
                    "expected_rows": entry["rows"],
                    "identity": sidecar["identity"],
                    "identity_source": archive_uri(sidecar_path),
                    "host": execution.get("host"),
                    "hostname": execution.get("hostname"),
                    "physical_gpus": execution.get("physical_gpus"),
                    "gpu_uuids": execution.get("gpu_uuids"),
                    "dtype": backend.get("dtype"),
                    "runtime_versions": versions,
                    "runtime_versions_key": version_key,
                    "environment_python": execution.get(
                        "runtime_environment_python", backend.get("environment_python")
                    ),
                    "environment_python_key": environment_key,
                    "backend_spec_sha256": definition.get("backend_spec_sha256"),
                    "model_config_sha256": backend.get("model_config_sha256"),
                    "processor_files": backend.get("processor", {}).get("files"),
                    "adapter_source_sha256": backend.get("adapter_source_sha256"),
                    "cpu_dispatch_env": definition.get("cpu_dispatch_env"),
                    "receipt_fields": [],
                    "explicit_runtime_fields": [],
                }
                receipt_parent = (
                    parent.replace("outputs/raw/", "outputs/records/", 1)
                    if "/acceleration_v4/" in source_path else parent
                )
                for filename in ("admission.json", "progress.json", "complete.json"):
                    path = f"{receipt_parent}/{filename}"
                    if path not in members:
                        continue
                    receipt = read_archived_json(path)
                    for field in (
                        "started_utc", "started", "finished_utc", "updated_utc",
                        "updated", "completed", "expected", "rows", "status",
                        "generation_complete",
                    ):
                        if field in receipt:
                            source["receipt_fields"].append({
                                "source_path": archive_uri(path),
                                "source_key": f"$.{field}",
                                "value": receipt[field],
                                "scope": "saved source-run receipt field",
                            })
                    source["explicit_runtime_fields"].extend(
                        {"source_path": archive_uri(path), **item}
                        for item in flatten_named(receipt, set(OPTIONAL_RUNTIME))
                    )
                source["explicit_runtime_fields"].extend(
                    {"source_path": archive_uri(sidecar_path), **item}
                    for item in flatten_named(sidecar, set(OPTIONAL_RUNTIME))
                )
                sources.append(source)
                model_sources[model["model"]].append(source)
                first += entry["rows"]
            if first - 1 != model["rows"]:
                raise ValueError("Source segment rows disagree with model rows")

        scalar_receipt = read_archived_json(f"{SCALARS}/run_receipt_v2.json")
        scalar_coverage = list(csv.DictReader(
            read_member(f"{SCALARS}/source_coverage.csv").decode().splitlines()
        ))
        inventory = read_archived_json(INVENTORY_PATH)
        dryrun = read_archived_json(DRYRUN_PATH)

    prompt_variants = Variants("prompt")
    config_variants = Variants("config")
    groups = {}
    row_counts = Counter()
    source_counts = Counter()
    source_presence = defaultdict(Counter)
    source_nulls = defaultdict(Counter)
    source_top_keys = defaultdict(Counter)
    source_first_fields = defaultdict(dict)
    source_status = defaultdict(Counter)
    source_splits = defaultdict(Counter)
    condition_prompt_ids = defaultdict(set)
    condition_config_ids = defaultdict(set)
    row_key_counts = set()
    duplicate_keys = 0
    model_scan_rows = {}
    optional_observations = []

    for model in manifest["models"]:
        model_key = model["model"]
        segments = model_sources[model_key]
        segment_index = 0
        with gzip.open(ROOT / model["output_path"], "rt", encoding="utf-8") as handle:
            model_rows = 0
            for line, text in enumerate(handle, 1):
                row = json.loads(text)
                while line > segments[segment_index]["segment_last_line"]:
                    segment_index += 1
                source = segments[segment_index]
                sid = source["source_id"]
                raw_line = line - source["segment_first_line"] + 1
                key = condition_tuple(row)
                if key not in conditions:
                    raise ValueError(f"Unexpected formal condition at {model_key}:{line}")
                if row["identity"] != source["identity"]:
                    raise ValueError(f"Identity mismatch at {model_key}:{line}")
                if row["model"] != model_key:
                    raise ValueError(f"Model mismatch at {model_key}:{line}")
                condition_id = conditions[key]["condition_id"]
                prompt = {
                    field: row[field]
                    for field in ("prompt", "reference_prompt", "neutral_prompt")
                    if field in row
                }
                prompt_id = prompt_variants.add(
                    prompt, source, line, raw_line, condition_id
                )
                config_id = config_variants.add(
                    row["config"], source, line, raw_line, condition_id
                )
                condition_prompt_ids[condition_id].add(prompt_id)
                condition_config_ids[condition_id].add(config_id)
                group_key = (condition_id, sid, row["identity"], config_id)
                if group_key not in groups:
                    groups[group_key] = {
                        "condition": key,
                        "source": source,
                        "config_id": config_id,
                        "prompt_ids": set(),
                        "n_rows": 0,
                        "first_line": raw_line,
                        "last_line": raw_line,
                        "first_consolidated_line": line,
                        "last_consolidated_line": line,
                        "wall_s_n": 0,
                        "wall_s_missing": 0,
                        "wall_s_null": 0,
                        "wall_s_invalid": 0,
                        "wall_s_sum": 0.0,
                        "wall_s_min": None,
                        "wall_s_max": None,
                        "tokens_n": 0,
                        "tokens_missing": 0,
                        "tokens_invalid": 0,
                        "tokens_sum": 0,
                        "tokens_min": None,
                        "tokens_max": None,
                        "status_counts": Counter(),
                    }
                group = groups[group_key]
                group["prompt_ids"].add(prompt_id)
                group["n_rows"] += 1
                group["last_line"] = raw_line
                group["last_consolidated_line"] = line
                group["status_counts"][str(row.get("status"))] += 1
                wall = row.get("wall_s")
                if "wall_s" not in row:
                    group["wall_s_missing"] += 1
                elif wall is None:
                    group["wall_s_null"] += 1
                elif type(wall) not in (float, int) or not math.isfinite(wall) or wall < 0:
                    group["wall_s_invalid"] += 1
                else:
                    group["wall_s_n"] += 1
                    group["wall_s_sum"] += wall
                    group["wall_s_min"] = (
                        wall if group["wall_s_min"] is None
                        else min(wall, group["wall_s_min"])
                    )
                    group["wall_s_max"] = (
                        wall if group["wall_s_max"] is None
                        else max(wall, group["wall_s_max"])
                    )
                tokens = row.get("tokens")
                if "tokens" not in row:
                    group["tokens_missing"] += 1
                elif not isinstance(tokens, list):
                    group["tokens_invalid"] += 1
                else:
                    token_count = len(tokens)
                    group["tokens_n"] += 1
                    group["tokens_sum"] += token_count
                    group["tokens_min"] = (
                        token_count if group["tokens_min"] is None
                        else min(token_count, group["tokens_min"])
                    )
                    group["tokens_max"] = (
                        token_count if group["tokens_max"] is None
                        else max(token_count, group["tokens_max"])
                    )
                for field in row:
                    source_top_keys[sid][field] += 1
                for field in ("wall_s", "tokens", *PROBABILITY_FIELDS, *OPTIONAL_RUNTIME):
                    if field not in row:
                        continue
                    source_presence[sid][field] += 1
                    source_nulls[sid][field] += row[field] is None
                    source_first_fields[sid].setdefault(field, raw_line)
                    if field in OPTIONAL_RUNTIME:
                        optional_observations.append({
                            "source_id": sid, "source_path": source["source_path"],
                            "line": raw_line, "source_key": f"$.{field}",
                            "value": row[field],
                        })
                source_status[sid][str(row.get("status"))] += 1
                source_splits[sid][str(row["sample"].get("split"))] += 1
                identity_key = (model_key, row["key"])
                if identity_key in row_key_counts:
                    duplicate_keys += 1
                row_key_counts.add(identity_key)
                source_counts[sid] += 1
                row_counts[key] += 1
                model_rows += 1
        model_scan_rows[model_key] = model_rows
        print(f"scanned {model_key}: {model_rows}", flush=True)

    mismatches = [
        {"condition": list(key), "expected": count, "actual": row_counts[key]}
        for key, count in expected.items() if row_counts[key] != count
    ]
    source_mismatches = [
        source["source_id"] for source in sources
        if source_counts[source["source_id"]] != source["expected_rows"]
    ]
    if mismatches or source_mismatches or duplicate_keys:
        raise ValueError(
            compact({"condition_mismatches": mismatches,
                     "source_mismatches": source_mismatches,
                     "duplicate_keys": duplicate_keys})
        )
    if sum(model_scan_rows.values()) != manifest["total_rows"]:
        raise ValueError("Total row mismatch")

    runtime_rows = []
    for record_id, (group_key, group) in enumerate(sorted(groups.items()), 1):
        condition_id, sid, identity, config_id = group_key
        source = group["source"]
        condition = conditions[group["condition"]]
        wall_n = group["wall_s_n"]
        tokens_n = group["tokens_n"]
        filter_value = {
            **condition, "identity": identity, "config_id": config_id,
            "config": config_variants.records[config_id - 1]["value"],
            "source_segment": {
                "first_consolidated_line": source["segment_first_line"],
                "last_consolidated_line": source["segment_last_line"],
            },
        }
        method = condition["method"]
        stream_count = {"vcd": 2, "instruction_vcd": 3}.get(method)
        runtime_rows.append({
            "runtime_record_id": record_id,
            **condition,
            "source_id": sid,
            "source_path": source["source_path"],
            "consolidated_path": source["consolidated_path"],
            "source_segment_first_line": source["segment_first_line"],
            "source_segment_last_line": source["segment_last_line"],
            "first_line": group["first_line"],
            "last_line": group["last_line"],
            "first_consolidated_line": group["first_consolidated_line"],
            "last_consolidated_line": group["last_consolidated_line"],
            "identity": identity,
            "config_id": config_id,
            "prompt_ids": compact(sorted(group["prompt_ids"])),
            "n_rows": group["n_rows"],
            "wall_s_n": wall_n,
            "wall_s_missing": group["wall_s_missing"],
            "wall_s_null": group["wall_s_null"],
            "wall_s_invalid": group["wall_s_invalid"],
            "wall_s_sum": group["wall_s_sum"] if wall_n else None,
            "wall_s_mean": group["wall_s_sum"] / wall_n if wall_n else None,
            "wall_s_min": group["wall_s_min"],
            "wall_s_max": group["wall_s_max"],
            "tokens_n": tokens_n,
            "tokens_missing": group["tokens_missing"],
            "tokens_invalid": group["tokens_invalid"],
            "tokens_sum": group["tokens_sum"] if tokens_n else None,
            "tokens_mean": group["tokens_sum"] / tokens_n if tokens_n else None,
            "tokens_min": group["tokens_min"],
            "tokens_max": group["tokens_max"],
            "status_counts": compact(group["status_counts"]),
            "host": source["host"],
            "hostname": source["hostname"],
            "physical_gpus": compact(source["physical_gpus"]),
            "gpu_uuids": compact(source["gpu_uuids"]),
            "dtype": source["dtype"],
            "runtime_versions": compact(source["runtime_versions"]),
            "environment_python": source["environment_python"],
            "cpu_dispatch_env": source["cpu_dispatch_env"],
            "identity_source_path": source["identity_source"],
            "identity_source_key": "$.identity",
            "hardware_source_keys": (
                "$.definition.execution.{host,hostname,physical_gpus,gpu_uuids}"
            ),
            "dtype_source_key": "$.definition.backend.dtype",
            "runtime_versions_source_key": source["runtime_versions_key"],
            "environment_python_source_key": source["environment_python_key"],
            "algorithm_stream_count": stream_count,
            "algorithm_stream_count_source": (
                "src/kdm/pipeline.py:57-68" if stream_count is not None else ""
            ),
            "algorithm_stream_count_scope": (
                "static session construction count" if stream_count is not None else ""
            ),
            "wall_s_source_key": "$.wall_s",
            "tokens_source_key": "$.tokens",
            "formula": (
                "n_rows=count(rows); wall_s_sum=sum(finite nonnegative wall_s); "
                "wall_s_mean=wall_s_sum/wall_s_n; tokens_sum=sum(len(tokens)); "
                "tokens_mean=tokens_sum/tokens_n; min/max over same observed values"
            ),
            "measurement_scope": WALL_SCOPE,
            "token_scope": TOKEN_SCOPE,
            "filter_json": compact(filter_value),
            "filter_scope": FILTER_SCOPE,
        })

    availability = []

    def avail(model, feature, status, path, key, scope, *,
              source_id=None, available=None, total=None, size=None, details=None):
        size_source = None
        if size is not None:
            if path.startswith(str(ARCHIVE_PATH) + "::"):
                member_path = path.split("::", 1)[1]
                if feature == "existing_generated_path_scalar_summary":
                    member_path = f"{SCALARS}/path_scalar_summary.json"
                index = archive_entries[member_path][0]
                size_source = f"{ARCHIVE_MANIFEST}#$.files[{index}].bytes"
            else:
                index = next(
                    i for i, model_entry in enumerate(manifest["models"])
                    if model_entry["output_path"] == path
                )
                size_source = f"{MANIFEST_PATH}#$.models[{index}].output_bytes"
        availability.append({
            "model": model, "source_id": source_id, "feature": feature,
            "availability": status, "source_path": path, "source_key_or_line": key,
            "available_rows": available, "total_rows": total,
            "stored_file_bytes": size, "stored_file_bytes_source": size_source,
            "measurement_scope": scope,
            "details": compact(details) if isinstance(details, (dict, list)) else details,
        })

    for source in sources:
        sid = source["source_id"]
        for field in ("wall_s", "tokens", *PROBABILITY_FIELDS, *OPTIONAL_RUNTIME):
            n = source_presence[sid][field]
            explicit = [
                row for row in source["explicit_runtime_fields"]
                if row["key"].split(".")[-1] == field
            ]
            status = (
                "present_in_formal_rows" if n else
                "present_in_bound_receipt" if explicit else
                "absent_in_formal_rows_and_bound_receipts"
            )
            avail(
                source["model"], field, status, source["consolidated_path"],
                f"$.{field}; original source lines 1-{source['expected_rows']}",
                (WALL_SCOPE if field == "wall_s" else
                 TOKEN_SCOPE if field == "tokens" else
                 "Stored field coverage; absent measurements remain empty"),
                source_id=sid, available=n, total=source["expected_rows"],
                size=source["consolidated_recorded_bytes"],
                details={
                    "source_path": source["source_path"],
                    "segment_first_line": source["segment_first_line"],
                    "segment_last_line": source["segment_last_line"],
                    "null_rows": source_nulls[sid][field],
                    "first_observed_source_line": source_first_fields[sid].get(field),
                    "bound_receipt_values": explicit,
                    "size_scope": "whole consolidated gzip; repeated per field/source, do not sum",
                },
            )
        avail(
            source["model"], "hardware_identity", "present_in_bound_identity",
            source["identity_source"],
            "$.definition.execution.{host,hostname,physical_gpus,gpu_uuids}",
            "Execution identity matched to every raw row in this source segment",
            source_id=sid, available=source["expected_rows"], total=source["expected_rows"],
            details={k: source[k] for k in ("host", "hostname", "physical_gpus", "gpu_uuids")},
        )

    for model in manifest["models"]:
        avail(
            model["model"], "formal_probability_storage", "available",
            model["output_path"],
            "$.selected_log_probabilities; $.sequence_log_probability; $.first_probability",
            "Saved generated-token probabilities with the formal response rows",
            available=model["rows"], total=model["rows"], size=model["output_bytes"],
            details={
                "recorded_uncompressed_source_bytes": sum(x["bytes"] for x in model["sources"]),
                "size_scope": "complete formal response file, not probability-only bytes",
                "byte_source": f"{MANIFEST_PATH}#$.models",
            },
        )
    for row in scalar_coverage:
        avail(
            row["model"], "existing_generated_path_scalar_summary", "available",
            archive_uri(f"{SCALARS}/source_coverage.csv"),
            f"CSV model={row['model']}; source_rows,trace_steps",
            "Existing per-condition generated-token scalar summary",
            available=int(row["source_rows"]), total=int(row["receipt_union_rows"]),
            size=archive_entries[f"{SCALARS}/path_scalar_summary.json"][1]["bytes"],
            details={
                "trace_steps": int(row["trace_steps"]),
                "data_path": archive_uri(f"{SCALARS}/path_scalar_summary.json"),
                "csv_path": archive_uri(f"{SCALARS}/path_scalar_metrics.csv"),
                "csv_fields": [
                    "model", "method", "marker", "reference_marker", "guided",
                    "reference_guided", "kind", "status", "config_json", "response_count",
                    "trace_steps", "metric", "token_weighted_json", "response_equal_json",
                    "layer_token_counts_json", "layer_response_fraction_sums_json",
                ],
                "size_scope": "all-model summary JSON; repeated by model, do not sum",
            },
        )
    findings = inventory["five_panel_measure_path_findings"]
    avail(
        "panel5", "shared_prefix_full_vocabulary_cache", "unavailable_in_named_inventory",
        archive_uri(INVENTORY_PATH), "$.five_panel_measure_path_findings",
        "Availability from the saved bounded inventory of five-panel mechanism paths",
        details=findings,
    )
    for index, entry in enumerate(inventory["persistent_real_method_replay_gate_outputs"]["paths"]):
        path = entry["path"]
        avail(
            "llava16_mistral", "existing_method_gate_trace", "available_limited_scope",
            archive_uri(path), "$.trace; $.tokens",
            "Existing method-generation gate outputs: two eval samples per file",
            available=entry["rows"], size=archive_entries[path][1]["bytes"],
            details={
                "inventory_source": (
                    f"{archive_uri(INVENTORY_PATH)}"
                    f"#$.persistent_real_method_replay_gate_outputs.paths[{index}]"
                ),
                "unique_sample_ids": entry["unique_sample_ids"],
                "per_method": entry["per_method"],
                "aggregate_unique_logical_keys": 18,
                "aggregate_execution_copies": 3,
                "stored_probability_type": "selected-token trace, no full-vocabulary tensor",
            },
        )
    avail(
        "panel5", "matched_prefix_dryrun_manifest", "preparation_only",
        archive_uri(DRYRUN_PATH), "$.mode; $.fully_verified_source_rows; $.limitations",
        "Saved CPU input preparation and validation receipt",
        size=archive_entries[DRYRUN_PATH][1]["bytes"],
        details={
            "mode": dryrun["mode"],
            "fully_verified_source_rows": dryrun["fully_verified_source_rows"],
            "limitations": dryrun["limitations"],
        },
    )

    condition_variants = []
    for key, condition in sorted(conditions.items(), key=lambda item: item[1]["condition_id"]):
        cid = condition["condition_id"]
        condition_variants.append({
            **condition, "n_rows": row_counts[key],
            "prompt_ids": sorted(condition_prompt_ids[cid]),
            "config_ids": sorted(condition_config_ids[cid]),
        })
    prompt_config_export = {
        "schema": "paper_20260929_actual_prompts_and_configs_v1",
        "condition_key_fields": CONDITION_KEYS,
        "condition_id_rule": "one-based sorted authoritative typed condition tuples",
        "variant_id_rule": (
            "one-based first encounter in formal manifest model order and original line order; "
            "JSON equality after key-order canonicalization"
        ),
        "prompt_value_definition": (
            "Exact saved {prompt, reference_prompt, neutral_prompt} object; nulls preserved"
        ),
        "location_definition": (
            "source_path and first_line/last_line refer to the original source JSONL; "
            "consolidated line fields refer to the gzip stream; count is exact variant occurrences"
        ),
        "prompts": prompt_variants.export(),
        "configs": config_variants.export(),
        "conditions": condition_variants,
    }
    checks = {
        "schema": "paper_20260929_runtime_checks_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": "venv/bin/python -B scripts/paper_20260929/export_runtime.py",
        "input_manifest": MANIFEST_PATH,
        "conditions_source": CONDITIONS_PATH,
        "formal_rows": sum(model_scan_rows.values()),
        "expected_formal_rows": manifest["total_rows"],
        "conditions": len(conditions),
        "expected_conditions": len(condition_rows),
        "condition_row_counts_all_match": not mismatches,
        "source_row_counts_all_match": not source_mismatches,
        "row_identity_matches_bound_source": True,
        "duplicate_model_task_keys": duplicate_keys,
        "model_scan_rows": model_scan_rows,
        "runtime_records": len(runtime_rows),
        "prompt_variants": len(prompt_variants.records),
        "config_variants": len(config_variants.records),
        "sum_runtime_rows": sum(row["n_rows"] for row in runtime_rows),
        "wall_s_observed_rows": sum(row["wall_s_n"] for row in runtime_rows),
        "tokens_observed_rows": sum(row["tokens_n"] for row in runtime_rows),
        "availability_records": len(availability),
        "measurement_definition_source": {
            "wall_s": "src/kdm/pipeline.py:81-119",
            "algorithm_stream_count": "src/kdm/pipeline.py:57-68",
            "tokens": "src/kdm/decoding.py generate saved tokens",
        },
        "measurement_scope": WALL_SCOPE,
        "grouping": FILTER_SCOPE,
        "comparison_rule": (
            "Runtime summaries retain source execution, GPU UUID, runtime versions, dtype, "
            "identity and exact config. Cost comparisons require matching these execution "
            "attributes and the relevant prompt/sample design. No elapsed-time ratio is emitted."
        ),
        "optional_formal_runtime_observations": optional_observations,
        "sources": sources,
        "source_counts": dict(source_counts),
        "source_status_counts": dict(source_status),
        "source_split_counts": dict(source_splits),
        "source_top_level_field_counts": dict(source_top_keys),
        "existing_scalar_receipt": {
            "source_path": archive_uri(f"{SCALARS}/run_receipt_v2.json"),
            "source_rows": scalar_receipt["source_rows"],
            "trace_steps": scalar_receipt["trace_steps"],
            "condition_groups": scalar_receipt["condition_groups"],
        },
        "bounded_archive_metadata_reads": sorted(set(read_members)),
        "archive_metadata_read_count": len(set(read_members)),
        "archive_manifest_path": str(ARCHIVE_MANIFEST),
        "input_content_hashes_computed": False,
        "formal_full_scans": 1,
        "outputs": [
            "outputs/paper_20260929/runtime_records.csv",
            "outputs/paper_20260929/runtime_availability.csv",
            "outputs/paper_20260929/runtime_checks.json",
            "outputs/paper_20260929/prompts_and_configs.json",
        ],
    }
    if checks["sum_runtime_rows"] != manifest["total_rows"]:
        raise ValueError("Runtime aggregate denominator mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "runtime_records.csv", runtime_rows)
    write_csv(OUT / "runtime_availability.csv", availability)
    write_json(OUT / "prompts_and_configs.json", prompt_config_export)
    write_json(OUT / "runtime_checks.json", checks)
    print(compact({
        key: checks[key] for key in (
            "formal_rows", "conditions", "runtime_records", "prompt_variants",
            "config_variants", "wall_s_observed_rows", "tokens_observed_rows",
            "availability_records",
        )
    }), flush=True)



def compare_runtime():
    """Compare saved runtime aggregates only after exact sample-set matching."""
    import sqlite3

    runtime_path = OUT / "runtime_records.csv"
    variants_path = OUT / "prompts_and_configs.json"
    index_path = OUT / "work/index.sqlite"
    with runtime_path.open(encoding="utf-8", newline="") as handle:
        records = [
            row for row in csv.DictReader(handle)
            if row["method"] in {"vcd", "instruction_vcd"}
        ]
    variants = load_json(variants_path)
    configs = {row["config_id"]: row["value"] for row in variants["configs"]}
    prompts = {row["prompt_id"]: row["value"] for row in variants["prompts"]}
    condition_meta = {
        row["condition_id"]: row for row in variants["conditions"]
    }
    db = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    source_ids = defaultdict(set)
    for model, source_file_id in db.execute(
        "SELECT DISTINCT model,source_file_id FROM responses"
    ):
        source_ids[model].add(source_file_id)
    paths_by_model = defaultdict(set)
    for row in records:
        paths_by_model[row["model"]].add(row["consolidated_path"])
    for model in paths_by_model:
        if len(source_ids[model]) != 1 or len(paths_by_model[model]) != 1:
            raise ValueError(f"Ambiguous source-file mapping for {model}")

    condition_samples = {}
    for cid in sorted({int(row["condition_id"]) for row in records}):
        condition_samples[cid] = list(db.execute(
            "SELECT sample_id,source_file_id,source_line FROM responses "
            "WHERE condition_id=? ORDER BY sample_id", (cid,)
        ))
    db.close()

    prepared = []
    for row in records:
        cid = int(row["condition_id"])
        meta = condition_meta[cid]
        for key in CONDITION_KEYS:
            actual = row[key]
            if key in ("guided", "reference_guided"):
                actual = actual == "True"
            elif key == "replicate":
                actual = int(actual)
            if actual != meta[key]:
                raise ValueError(f"Condition mismatch in runtime record {row['runtime_record_id']}")
        config_id = int(row["config_id"])
        prompt_ids = json.loads(row["prompt_ids"])
        if meta["config_ids"] != [config_id] or set(meta["prompt_ids"]) != set(prompt_ids):
            raise ValueError("The available index cannot resolve varying config/prompt per sample")
        common_config = {
            key: value for key, value in configs[config_id].items()
            if key != "method"
        }
        if configs[config_id]["method"] != row["method"]:
            raise ValueError("Runtime method disagrees with saved config")
        prompt_objects = [prompts[pid] for pid in prompt_ids]
        main_prompts = {value["prompt"] for value in prompt_objects}
        if len(main_prompts) != 1:
            raise ValueError("A runtime record contains multiple actual main prompts")
        main_prompt = next(iter(main_prompts))
        source_file_id = next(iter(source_ids[row["model"]]))
        first = int(row["source_segment_first_line"])
        last = int(row["source_segment_last_line"])
        sample_rows = [
            item for item in condition_samples[cid]
            if item[1] == source_file_id and first <= item[2] <= last
        ]
        sample_ids = frozenset(item[0] for item in sample_rows)
        if len(sample_rows) != int(row["n_rows"]) or len(sample_ids) != len(sample_rows):
            raise ValueError(f"Sample count/uniqueness mismatch: {row['runtime_record_id']}")
        if len(condition_samples[cid]) != meta["n_rows"]:
            raise ValueError(f"Full condition count mismatch: {cid}")
        if any(int(row[field]) != int(row["n_rows"]) for field in ("wall_s_n", "tokens_n")):
            raise ValueError(f"Incomplete recorded cost: {row['runtime_record_id']}")
        gpu_uuids = sorted(json.loads(row["gpu_uuids"]).values())
        versions = json.loads(row["runtime_versions"])
        if not all((row["host"], gpu_uuids, row["dtype"], versions, row["environment_python"])):
            raise ValueError(f"Missing comparison identity: {row['runtime_record_id']}")
        cpu_tag = row["cpu_dispatch_env"]
        cpu_match_key = (
            ("recorded_cpu_dispatch_env", cpu_tag) if cpu_tag else
            ("same_execution_identity_cpu_tag_unrecorded", row["identity"])
        )
        signature = (
            row["model"], row["host"], tuple(gpu_uuids), row["dtype"],
            compact(versions), row["environment_python"], cpu_match_key,
            compact(common_config), main_prompt, int(row["replicate"]),
        )
        prepared.append({
            "row": row, "condition": {key: meta[key] for key in CONDITION_KEYS},
            "source_file_id": source_file_id, "sample_ids": sample_ids,
            "sample_count": len(sample_ids), "full_condition_n": meta["n_rows"],
            "config": common_config, "prompt": main_prompt,
            "prompt_objects": prompt_objects, "gpu_uuids": gpu_uuids,
            "versions": versions, "cpu_match_basis": cpu_match_key[0],
            "signature": signature,
        })

    grouped = defaultdict(lambda: {"vcd": [], "instruction_vcd": []})
    for entry in prepared:
        grouped[entry["signature"]][entry["row"]["method"]].append(entry)
    pairs = []
    matched_ids = set()
    for candidates in grouped.values():
        for vcd in candidates["vcd"]:
            for ip in candidates["instruction_vcd"]:
                if vcd["sample_ids"] == ip["sample_ids"]:
                    pairs.append((vcd, ip))
                    matched_ids.update((
                        int(vcd["row"]["runtime_record_id"]),
                        int(ip["row"]["runtime_record_id"]),
                    ))
    pairs.sort(key=lambda pair: (
        int(pair[0]["row"]["runtime_record_id"]),
        int(pair[1]["row"]["runtime_record_id"]),
    ))

    query = (
        "SELECT sample_id FROM responses WHERE condition_id=:condition_id "
        "AND source_file_id=:source_file_id AND source_line BETWEEN "
        ":source_segment_first_line AND :source_segment_last_line ORDER BY sample_id"
    )

    def make_row(vcd=None, ip=None, availability="available", reason=""):
        entry = vcd or ip
        row = entry["row"]
        same_set = vcd is not None and ip is not None
        data = {
            "comparison_id": None,
            "availability": availability,
            "availability_reason": reason,
            "model": row["model"],
            "host": row["host"],
            "gpu_uuids": compact(entry["gpu_uuids"]),
            "dtype": row["dtype"],
            "runtime_versions": compact(entry["versions"]),
            "environment_python": row["environment_python"],
            "cpu_dispatch_env": row["cpu_dispatch_env"],
            "cpu_environment_match_basis": entry["cpu_match_basis"],
            "same_execution_identity": (
                vcd["row"]["identity"] == ip["row"]["identity"] if same_set else None
            ),
            "common_decode_config_json": compact(entry["config"]),
            "common_decode_config_rule": "all original config fields equal except method",
            "main_prompt": entry["prompt"],
            "replicate": int(row["replicate"]),
            "sample_ids_equal": True if same_set else None,
            "shared_sample_count": entry["sample_count"] if same_set else None,
            "sample_match_rule": "exact full sample_id set equality, no intersection/subsampling",
            "sample_index_path": "outputs/paper_20260929/work/index.sqlite",
            "sample_query": query,
            "source_file_mapping_basis": (
                "unique model/source_file_id in SQLite responses joined to unique "
                "model/consolidated_path in runtime_records; both uniqueness checks passed"
            ),
        }
        for label, item in (("vcd", vcd), ("ip_vcd", ip)):
            original = item["row"] if item else {}
            for field in (
                "runtime_record_id", "condition_id", "source_id", "source_path",
                "consolidated_path", "source_segment_first_line",
                "source_segment_last_line", "identity", "config_id", "prompt_ids",
                "n_rows", "wall_s_n", "wall_s_sum", "tokens_n", "tokens_sum",
            ):
                data[f"{label}_{field}"] = original.get(field)
            data[f"{label}_condition_json"] = compact(item["condition"]) if item else None
            data[f"{label}_sqlite_source_file_id"] = item["source_file_id"] if item else None
            data[f"{label}_full_condition_n"] = item["full_condition_n"] if item else None
            data[f"{label}_sample_count"] = item["sample_count"] if item else None
            data[f"{label}_sample_coverage_fraction"] = (
                item["sample_count"] / item["full_condition_n"] if item else None
            )
            data[f"{label}_prompt_objects_json"] = compact(item["prompt_objects"]) if item else None
            data[f"{label}_recorded_value_source"] = (
                "outputs/paper_20260929/runtime_records.csv"
                f"#runtime_record_id={original['runtime_record_id']}; "
                "wall_s_sum,tokens_sum,wall_s_n,tokens_n" if item else None
            )
        if same_set:
            denominator = float(vcd["row"]["wall_s_sum"])
            if denominator <= 0:
                data["availability"] = "unavailable"
                data["availability_reason"] = "nonpositive_recorded_vcd_wall_sum"
                data["ip_over_vcd_wall_s_ratio"] = None
            else:
                data["ip_over_vcd_wall_s_ratio"] = float(ip["row"]["wall_s_sum"]) / denominator
        else:
            data["ip_over_vcd_wall_s_ratio"] = None
        data["ratio_formula"] = (
            "float(ip_vcd_wall_s_sum) / float(vcd_wall_s_sum); "
            "derived only from existing recorded wall_s totals over equal sample sets"
        )
        data["timing_scope"] = row["measurement_scope"]
        data["coverage_note"] = (
            "Each side retains its complete recorded runtime group. Device, environment, "
            "main prompt and config except method match. Reference/neutral prompts and "
            "method remain explicit conditions; saved output token counts may differ."
            if same_set else
            "This recorded runtime group has no counterpart satisfying all comparison "
            "requirements. Its original cost fields are retained and the ratio is empty."
        )
        return data

    output = [make_row(vcd, ip) for vcd, ip in pairs]
    for entry in sorted(prepared, key=lambda item: int(item["row"]["runtime_record_id"])):
        row = entry["row"]
        if int(row["runtime_record_id"]) in matched_ids:
            continue
        opposite = "instruction_vcd" if row["method"] == "vcd" else "vcd"
        candidates = grouped[entry["signature"]][opposite]
        reason = (
            "no_equal_sample_id_set_with_matching_device_environment_config_main_prompt"
            if candidates else
            "no_counterpart_with_matching_device_environment_config_main_prompt"
        )
        output.append(make_row(
            vcd=entry if row["method"] == "vcd" else None,
            ip=entry if row["method"] == "instruction_vcd" else None,
            availability="unavailable", reason=reason,
        ))
    for number, row in enumerate(output, 1):
        row["comparison_id"] = number
    write_csv(OUT / "runtime_comparisons.csv", output)
    print(compact({
        "comparison_rows": len(output),
        "available_pairs": sum(row["availability"] == "available" for row in output),
        "unmatched_runtime_records": len(prepared) - len(matched_ids),
        "input_vcd_records": sum(row["method"] == "vcd" for row in records),
        "input_ip_vcd_records": sum(row["method"] == "instruction_vcd" for row in records),
        "matched_vcd_records": sum(
            item["row"]["method"] == "vcd"
            and int(item["row"]["runtime_record_id"]) in matched_ids for item in prepared
        ),
        "matched_ip_vcd_records": sum(
            item["row"]["method"] == "instruction_vcd"
            and int(item["row"]["runtime_record_id"]) in matched_ids for item in prepared
        ),
        "sample_index_rows_examined": sum(map(len, condition_samples.values())),
        "input_paths": [str(runtime_path), str(variants_path), str(index_path)],
        "output_path": str(OUT / "runtime_comparisons.csv"),
    }), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.comparisons_only:
        compare_runtime()
    else:
        main()


