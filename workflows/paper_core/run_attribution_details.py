#!/usr/bin/env python3
"""Plan the bounded P1 attribution detail phases.

This entry point deliberately keeps planning CPU-only.  GPU execution is an
explicit later phase and must reuse the admission/backend code in
``run_attribution.py``.  The planner never re-encodes a response and never
infers a semantic label from a target class.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def longest_common_prefix(*sequences: Sequence[int]) -> int:
    """Return the token LCP length without decoding or re-tokenizing text."""
    if not sequences:
        return 0
    limit = min(map(len, sequences))
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    return limit


def _frozen_diagnostic_pairs(group: dict[str, dict]) -> list[tuple[int, list[int], list[int], str]]:
    """Return at most two real branch pairs, each measured at its own LCP."""
    candidates = [("vcd_unknown", "instruction_vcd_unknown", "guided_vcd_ip"),
                 ("direct_guided", "vcd_unknown", "direct_guided_vcd")]
    output = []
    seen = set()
    for left_name, right_name, label in candidates:
        if left_name not in group or right_name not in group:
            continue
        left = list(map(int, group[left_name]["tokens"]))
        right = list(map(int, group[right_name]["tokens"]))
        position = longest_common_prefix(left, right)
        if position >= min(len(left), len(right)) or position in seen:
            continue
        seen.add(position)
        output.append((position, left, right, label))
    # The preserved-answer stratum can have three equal guided paths while
    # the actual unguided path differs.  Use its real LCP only when neither
    # of the two priority comparisons has a divergence.
    if not output and "direct_unguided" in group and "direct_guided" in group:
        left = list(map(int, group["direct_unguided"]["tokens"]))
        right = list(map(int, group["direct_guided"]["tokens"]))
        position = longest_common_prefix(left, right)
        if position < min(len(left), len(right)):
            output.append((position, left, right, "direct_unguided_guided"))
    return output[:2]


def _budget(path: Path | None, model: str) -> dict:
    if path is None:
        return {"model": model, "used_gpu_seconds": 0.0, "source": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model") not in (None, model):
        raise ValueError(f"prior budget model mismatch: {path}")
    used = float(value.get("used_gpu_seconds", 0.0))
    if used < 0:
        raise ValueError(f"negative prior budget: {path}")
    return {"model": model, "used_gpu_seconds": used, "source": str(path),
            "sha256": _hash(path)}


def _diagnostic_positions(rows: Iterable[dict], source_records: dict[tuple[str, str], dict] | None = None) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for row in rows:
        sample_id = row.get("sample_id") or row.get("id")
        if not sample_id:
            raise ValueError("diagnostic row has no sample_id")
        branches = row.get("branches")
        if isinstance(branches, dict):
            token_lists = [value.get("tokens", []) for value in branches.values()]
        else:
            token_lists = [row.get(name, []) for name in
                           ("guided_vcd_tokens", "ip_tokens", "direct_tokens", "vcd_tokens")]
        if not any(token_lists) and source_records is not None:
            group = source_records.get((str(row.get("model")), str(sample_id)), {})
            # The sealed packet names these two frozen branches explicitly.  The
            # CSV is a selection manifest, so its tokens must come from that
            # packet rather than from text re-encoding.
            token_lists = [group.get("vcd_unknown", {}).get("tokens", []),
                           group.get("instruction_vcd_unknown", {}).get("tokens", [])]
        token_lists = [list(map(int, tokens)) for tokens in token_lists if tokens]
        if len(token_lists) < 2:
            raise ValueError(f"diagnostic row lacks two token branches: {sample_id}")
        lcp = longest_common_prefix(token_lists[0], token_lists[1])
        explicit = row.get("diagnostic_positions", [])
        if explicit:
            selected = sorted({int(position) for position in explicit
                               if lcp <= int(position) < max(map(len, token_lists))})
        else:
            selected = [lcp] if lcp < min(map(len, token_lists)) else []
        identity = f"{row.get('model', '')}::{sample_id}"
        positions[identity] = selected
    return positions


def build_plan(args: argparse.Namespace) -> dict:
    inputs = args.inputs.resolve()
    records = _jsonl(inputs)
    if len(records) != 2836:
        raise ValueError(f"attribution input count is {len(records)}, expected 2836")
    models = sorted({row.get("model") for row in records if row.get("model")})
    if len(models) != 5:
        raise ValueError(f"expected five models in attribution inputs, got {models}")
    keys = {(row.get("model"), row.get("sample_id"), row.get("condition"))
            for row in records}
    if len(keys) != len(records):
        raise ValueError("attribution input contains duplicate model/sample/condition keys")
    if None in {part for key in keys for part in key}:
        raise ValueError("attribution input is missing model/sample_id/condition")
    representative_path = args.representative_list.resolve()
    with representative_path.open(newline="", encoding="utf-8") as stream:
        representative_rows = list(csv.DictReader(stream))
    if len(representative_rows) != 101 or len({row["sample_id"] for row in representative_rows}) != 101:
        raise ValueError("representative manifest must contain 101 unique sample IDs")
    source_records: dict[tuple[str, str], dict] = {}
    for item in records:
        source_records.setdefault((str(item["model"]), str(item["sample_id"])), {})[item["condition"]] = item
    for model in models:
        for row in representative_rows:
            group = source_records.get((model, row["sample_id"]), {})
            if set(group) != {"direct_guided", "direct_unguided", "vcd_unknown", "instruction_vcd_unknown"}:
                raise ValueError(f"representative four-branch join incomplete: {model}/{row['sample_id']}")
    diagnostic_rows = []
    diagnostic_path = args.diagnostic_csv.resolve() if args.diagnostic_csv else None
    if diagnostic_path:
        with diagnostic_path.open(newline="", encoding="utf-8") as stream:
            diagnostic_rows = list(csv.DictReader(stream))
        if len(diagnostic_rows) != 198:
            raise ValueError(f"diagnostic CSV count is {len(diagnostic_rows)}, expected 198")
        diagnostic_positions = _diagnostic_positions(diagnostic_rows, source_records)
    else:
        diagnostic_positions = {}
    case_rows = []
    if args.cases:
        case_rows = _jsonl(args.cases.resolve())
        if len(case_rows) != 12:
            raise ValueError(f"case list count is {len(case_rows)}, expected 12")
    else:
        case_rows = _case_rows_from_records(records)
    budgets = {model: _budget(args.prior_budget if args.prior_budget and args.model == model else
                              (args.prior_budget_dir / f"{model}.budget.json"
                               if args.prior_budget_dir else None), model)
               for model in models}
    total_used = sum(item["used_gpu_seconds"] for item in budgets.values())
    if any(item["used_gpu_seconds"] > 2880 for item in budgets.values()):
        raise ValueError("a model already exceeds its 2880 GPU-second cap")
    return {
        "schema": "kdm_p1_attribution_detail_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "models": models,
        "inputs": {"path": str(inputs), "sha256": _hash(inputs), "records": len(records)},
        "required_counts": {"representative": 505, "diagnostic": 198, "cases": 12},
        "representative": {"path": str(representative_path), "sample_count": len(representative_rows),
                            "models": len(models), "joined_rows": len(models) * len(representative_rows),
                            "required_count": 505},
        "diagnostic": {"path": str(diagnostic_path) if diagnostic_path else None,
                        "count": len(diagnostic_rows), "positions": diagnostic_positions},
        "cases": {"path": str(args.cases.resolve()) if args.cases else "packet:purpose=case12",
                   "count": len(case_rows), "ids": case_rows},
        "budget": {"per_model_cap_gpu_seconds": 2880, "aggregate_cap_gpu_seconds": 14400,
                   "prior_total_gpu_seconds": total_used, "prior": budgets},
        "phases": ["representative", "diagnostic", "cases"],
        "execution_requires": ["--execute", "registered physical GPU locks",
                               "run_attribution.py admission/backend", "fixed source tokens"],
    }


def _source_groups(path: Path, model: str) -> dict[str, dict[str, dict]]:
    groups: dict[str, dict[str, dict]] = {}
    for row in _jsonl(path):
        if row.get("model") == model:
            groups.setdefault(row["sample_id"], {})[row["condition"]] = row
    return groups


def _case_rows_from_records(records: Sequence[dict]) -> list[dict]:
    """Recover the sealed twelve panel cases from packet provenance."""
    cases: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("purpose") != "case12":
            continue
        key = (row["model"], row["sample_id"])
        item = cases.setdefault(key, {"model": row["model"], "sample_id": row["sample_id"],
                                      "target_class": row.get("target_class"), "source_keys": []})
        item["source_keys"].append(row["source_key"])
    output = list(cases.values())
    if len(output) != 12:
        raise ValueError(f"packet case12 provenance contains {len(output)} unique cases, expected 12")
    for item in output:
        item["source_keys"] = sorted(set(item["source_keys"]))
    return sorted(output, key=lambda item: (item["model"], item["sample_id"]))


def _check_selected_sources(groups: dict[str, dict[str, dict]], selected: Sequence[str],
                            required: set[str]) -> None:
    if len(selected) != len(set(selected)):
        raise ValueError("selected attribution samples contain duplicates")
    missing = [sample_id for sample_id in selected
               if sample_id not in groups or not required <= set(groups[sample_id])]
    if missing:
        raise ValueError(f"selected source rows are incomplete: {missing[:3]}")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _reuse_phase_ledgers(old_root: Path, new_run: Path, model: str,
                         expected: dict, phase_definitions: dict) -> dict:
    """Import immutable completed rows with provenance; never rewrite old rows."""
    from kdm.io import Ledger, stable_hash

    old = old_root / model
    imported = {}
    for name in ("natural_reference.events.jsonl", "diagnostic.events.jsonl", "case12.events.jsonl"):
        source = old / name
        identity_path = source.with_suffix(".identity.json")
        if not source.exists():
            continue
        if not identity_path.exists():
            raise ValueError(f"reuse ledger has no identity: {source}")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        definition = identity.get("definition", {})
        if identity.get("identity") != stable_hash(definition):
            raise ValueError(f"reuse metadata identity is corrupt: {source}")
        for field, value in (("model", model), ("inputs_sha256", expected["inputs_sha256"]),
                             ("measurement_sha256", expected["measurement_sha256"]),
                             ("config_sha256", expected["config_sha256"])):
            if definition.get(field) != value:
                raise ValueError(f"reuse identity mismatch {field}: {source}")
        if name == "diagnostic.events.jsonl" and definition.get("csv_sha256") != expected["csv_sha256"]:
            raise ValueError(f"reuse diagnostic selection changed: {source}")
        destination = new_run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(destination, phase_definitions[name])
        source_sha256 = _hash(source)
        row_provenance = []
        source_keys = set()
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if raw.strip():
                    item = json.loads(raw)
                    key = item.get("key")
                    if not key or key in source_keys or item.get("identity") != identity["identity"]:
                        raise ValueError(f"reuse row identity/key is corrupt: {source}:{line_number}")
                    source_keys.add(key)
                    provenance = {"path": str(source), "identity_path": str(identity_path),
                                  "identity": identity["identity"], "line": line_number,
                                  "line_sha256": hashlib.sha256(raw).hexdigest(),
                                  "key": key, "source_file_sha256": source_sha256}
                    row_provenance.append(provenance)
                    if key not in ledger.keys:
                        ledger.add(key, {**item, "reused_source_ledger": provenance})
        imported[name] = {"path": str(source), "identity_path": str(identity_path),
                          "identity": identity, "rows": len(source_keys),
                          "row_keys": sorted(source_keys),
                          "row_provenance": row_provenance,
                          "sha256": source_sha256}
    return imported


class _BudgetSession:
    def __init__(self, session, started, cap, cards):
        self._session, self._started, self._cap, self._cards = session, started, cap, cards

    def next(self, prefix):
        elapsed = (time.perf_counter() - self._started) * len(self._cards)
        if elapsed >= self._cap:
            raise RuntimeError("attribution budget exhausted during session.next")
        return self._session.next(prefix)

    def __getattr__(self, name):
        return getattr(self._session, name)


class _BudgetBackend:
    def __init__(self, backend, started, cap, cards):
        self._backend, self._started, self._cap, self._cards = backend, started, cap, cards
        self.eos = backend.eos

    def session(self, *args, **kwargs):
        elapsed = (time.perf_counter() - self._started) * len(self._cards)
        if elapsed >= self._cap:
            raise RuntimeError("attribution budget exhausted before session creation")
        return _BudgetSession(self._backend.session(*args, **kwargs), self._started,
                              self._cap, self._cards)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _execute(args: argparse.Namespace, plan: dict) -> dict:
    """Run only after explicit admission; all phases use sealed source tokens."""
    if not args.model or not args.physical_gpus or not args.run_id:
        raise ValueError("--model, --physical-gpus and --run-id are required for execution")
    if not args.diagnostic_csv:
        raise ValueError("--execute requires the sealed diagnostic CSV")
    # These imports are intentionally inside the execute gate: plan checks stay CPU-only.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from PIL import Image
    from kdm.decoding import DecodeConfig, generate
    from kdm.execution import resolve_image_path
    from kdm.io import Ledger, atomic_json, stable_hash, stable_seed
    from kdm.pipeline import make_backend
    from kdm.prompts import task_prompt
    import execution
    from attribution import measure_attribution, measure_case
    from native import MODELS

    if args.model not in MODELS:
        raise ValueError(f"unregistered core model: {args.model}")
    cards = args.physical_gpus.split(",")
    if len(cards) != 1 or not cards[0].isdigit():
        raise ValueError("one registered physical GPU is required per thin worker")
    execution.REGISTRY = args.registry
    root = ROOT
    frozen_spec = json.loads((root / f"configs/runtime/{args.model}.json").read_text())
    admission = execution.admit(root, frozen_spec, args.model, cards)
    spec = execution.runtime_spec(root, frozen_spec, args.model)
    groups = _source_groups(args.inputs, args.model)
    samples = {row["id"]: row for row in _jsonl(root / "data/current/all.jsonl")}
    representative = list(csv.DictReader(args.representative_list.open(encoding="utf-8")))
    selected = [row["sample_id"] for row in representative]
    _check_selected_sources(groups, selected,
                            {"direct_guided", "direct_unguided", "vcd_unknown", "instruction_vcd_unknown"})
    run = root / "outputs/paper_core_20260930" / args.run_id / "details" / args.model
    run.mkdir(parents=True, exist_ok=True)
    common_identity = {"runner_sha256": _hash(Path(__file__)),
                       "measurement_sha256": _hash(Path(__file__).with_name("attribution.py")),
                       "config_sha256": _hash(root / f"configs/runtime/{args.model}.json"),
                       "inputs_sha256": _hash(args.inputs),
                       "admission_fixed_identity": admission["fixed_identity"],
                       "source_paths": sorted({source["source_path"] for group in groups.values()
                                               for source in group.values()})}
    phase_definitions = {
        "natural_reference.events.jsonl": {
            "schema": "kdm_p1_natural101_v1", "model": args.model,
            "phase": "natural101", **common_identity,
            "source": "sealed direct_unguided clean response; generated noise r only"},
        "diagnostic.events.jsonl": {
            "schema": "kdm_p1_diagnostic198_v1", "model": args.model,
            "csv_sha256": _hash(args.diagnostic_csv), **common_identity},
        "case12.events.jsonl": {
            "schema": "kdm_p1_case12_v1", "model": args.model, **common_identity},
    }
    if args.reuse_details_dir:
        reused = _reuse_phase_ledgers(args.reuse_details_dir.resolve(), run, args.model,
                                      {"inputs_sha256": common_identity["inputs_sha256"],
                                       "measurement_sha256": common_identity["measurement_sha256"],
                                       "config_sha256": common_identity["config_sha256"],
                                       "csv_sha256": _hash(args.diagnostic_csv)}, phase_definitions)
    else:
        reused = {}
    atomic_json(run / "reuse_receipt.json", reused)
    budget_path = run / "budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else (json.loads(args.prior_budget.read_text()) if args.prior_budget else {
        "model": args.model, "used_gpu_seconds": 0.0, "cap_gpu_seconds": 2880.0,
        "includes": "model loading, natural reference, diagnostics, cases and failures"})
    if budget.get("model") != args.model or float(budget.get("used_gpu_seconds", 0)) >= 2880:
        raise RuntimeError("model attribution budget is exhausted or belongs to another model")
    started = time.perf_counter()
    backend = None
    outputs: dict[str, int] = {}
    prior_used = float(budget.get("used_gpu_seconds", 0.0))
    remaining_budget = 2880.0 - prior_used
    if remaining_budget <= 0:
        raise RuntimeError("no remaining attribution budget")
    try:
        backend = _BudgetBackend(make_backend(spec, "cuda:0"), started, remaining_budget, cards)
        if (time.perf_counter() - started) * len(cards) >= remaining_budget:
            raise RuntimeError("model loading exhausted remaining attribution budget")
        if "representative" in args.phase:
            cfg = DecodeConfig(method="direct", max_tokens=32, temperature=0.0, top_p=1.0)
            ledger = Ledger(run / "natural_reference.events.jsonl",
                            phase_definitions["natural_reference.events.jsonl"])
            for sample_id in selected:
                key = stable_hash([args.model, sample_id, "natural101"])
                if key in ledger.keys:
                    continue
                if budget["used_gpu_seconds"] + (time.perf_counter() - started) * len(cards) >= 2880:
                    raise RuntimeError("model budget exhausted before natural101 sample")
                sample = samples[sample_id]
                seed = stable_seed(sample_id, args.model, 0)
                with Image.open(resolve_image_path(sample["image_path"], root)) as source:
                    image = source.convert("RGB")
                    prompt = task_prompt(sample["question"], guided=False)
                    noisy = backend.session(image, prompt, reference="noise", seed=seed)
                    noisy_result = generate(noisy, None, cfg, backend.eos, backend.decode, seed)
                ledger.add(key, {"model": args.model, "sample_id": sample_id, "seed": seed,
                                 "panel": "natural101", "prompt": prompt,
                                 "clean_source": groups[sample_id]["direct_unguided"],
                                 "noise_generated_r": noisy_result, "status": "ok"})
            outputs["representative"] = len(ledger.keys)
        if "diagnostic" in args.phase:
            diag = list(csv.DictReader(args.diagnostic_csv.open(encoding="utf-8")))
            diag = [row for row in diag if row["model"] == args.model]
            if not diag:
                raise ValueError(f"diagnostic CSV has no rows for {args.model}")
            diag_ledger = Ledger(run / "diagnostic.events.jsonl",
                                 phase_definitions["diagnostic.events.jsonl"])
            for row in diag:
                sample_id = row["sample_id"]
                key = stable_hash([args.model, sample_id, "diagnostic198"])
                if key in diag_ledger.keys:
                    continue
                if budget["used_gpu_seconds"] + (time.perf_counter() - started) * len(cards) >= 2880:
                    raise RuntimeError("model budget exhausted before diagnostic sample")
                group = groups[sample_id]
                pairs = _frozen_diagnostic_pairs(group)
                if not pairs:
                    source_tokens = {name: source["tokens"] for name, source in group.items()}
                    lcp = longest_common_prefix(*source_tokens.values())
                    diag_ledger.add(key, {"config_row": row, "measurement_status": "no_divergence",
                                          "source_tokens": source_tokens,
                                          "source_text": {name: source["text"] for name, source in group.items()},
                                          "lcp": lcp, "positions": [], "status": "ok"})
                    continue
                positions = [item[0] for item in pairs]
                sample = samples[sample_id]; seed = int(group["vcd_unknown"]["seed"])
                with Image.open(resolve_image_path(sample["image_path"], root)) as source:
                    measured_pairs = []
                    for position, left, right, label in pairs:
                        measured_pairs.append(measure_attribution(
                            backend, source.convert("RGB"), sample["question"], [left[:position]], seed,
                            diagnostic_positions=[position], observed_tokens=left,
                            candidate_pairs=[(left[position], right[position])],
                            source_meta={"model": args.model, "sample_id": sample_id,
                                         "panel": "diagnostic198", "pair_label": label,
                                         "position": position, "source_branches": group}))
                diag_ledger.add(key, {"config_row": row, "measured_pairs": measured_pairs, "status": "ok"})
            outputs["diagnostic"] = len(diag_ledger.keys)
        if "cases" in args.phase:
            case_rows = _jsonl(args.cases) if args.cases else _case_rows_from_records(_jsonl(args.inputs))
            case_rows = [case for case in case_rows if case.get("model") == args.model]
            case_ledger = Ledger(run / "case12.events.jsonl",
                                 phase_definitions["case12.events.jsonl"])
            for case in case_rows:
                sample_id = case["sample_id"]
                sample = samples[sample_id]
                key = stable_hash([args.model, sample_id, "case12"])
                if key in case_ledger.keys:
                    continue
                if budget["used_gpu_seconds"] + (time.perf_counter() - started) * len(cards) >= 2880:
                    raise RuntimeError("model budget exhausted before case12 sample")
                sources = [groups[sample_id][condition] for condition in
                           ("vcd_unknown", "instruction_vcd_unknown", "direct_guided")
                           if condition in groups[sample_id]][:2]
                if len(sources) < 2:
                    raise ValueError(f"case12 requires two sealed response paths: {args.model}/{sample_id}")
                measured_paths = []
                with Image.open(resolve_image_path(sample["image_path"], root)) as image:
                    for source in sources:
                        measured_paths.append(measure_case(
                            backend, image.convert("RGB"), sample["question"], source["tokens"],
                            int(source["seed"]), observed_text=source["text"],
                            source_meta={"model": args.model, "panel": "case12",
                                         "condition": source["condition"], "source": source}))
                case_ledger.add(key, {"case": case, "paths": measured_paths, "status": "ok"})
            outputs["cases"] = len(case_ledger.keys)
    finally:
        elapsed = (time.perf_counter() - started) * len(cards)
        budget["used_gpu_seconds"] = float(budget.get("used_gpu_seconds", 0.0)) + elapsed
        budget["last_call_gpu_seconds"] = elapsed
        budget["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(budget_path, budget)
    result = {"schema": "kdm_p1_attribution_detail_run_v1", "model": args.model,
              "run_id": args.run_id, "phase_counts": outputs, "admission": admission,
              "reused_ledgers": reused,
              "budget": budget, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(run / "run_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--diagnostic-csv", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--prior-budget-dir", type=Path)
    parser.add_argument("--prior-budget", type=Path)
    parser.add_argument("--reuse-details-dir", type=Path,
                        help="immutable prior details root containing <model> phase ledgers")
    parser.add_argument("--phase", choices=("representative", "diagnostic", "cases"),
                        action="append", default=[])
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="run admitted GPU phases; never implied by planning")
    parser.add_argument("--model", choices=("gemma3_4b", "llava16_mistral", "minicpm26",
                                             "qwen25vl", "qwen35_4b"))
    parser.add_argument("--run-id")
    parser.add_argument("--physical-gpus")
    parser.add_argument("--representative-list", type=Path,
                        default=Path("C:/Users/zhn19/Downloads/2/KDM_Core_Experiments_20260930/configs/attribution_representative101.csv"))
    parser.add_argument("--registry", default="workflows/paper_core/host_registry.json")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.phase:
        plan["selected_phases"] = args.phase
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        if not args.phase:
            raise ValueError("--execute requires at least one --phase")
        result = _execute(args, plan)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    print(json.dumps({"plan": str(args.plan_out), "records": 2836,
                      "required_counts": plan["required_counts"], "cpu_only": True},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
