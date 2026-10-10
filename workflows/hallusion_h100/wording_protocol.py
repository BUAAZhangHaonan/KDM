"""CPU-only registration, condition and score contracts for Hallusion wording.

The scientific decoder remains in the original project. Importing this module
never imports torch, model adapters, or the original generation entrypoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl")
MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it")
IP_METHODS = ("instruction_vcd", "instruction_m3id")
GUIDED_SUFFIX = '\nIf you cannot determine the answer reliably, respond with "{marker}".'
REFERENCE_SHA = "22bcc117d6734afe93dd82561e8c2c8b3d3d38ed5b3b51968e269066d2650206"
POLICY_SHA = "4b84ad496778db760b029122886555487f803b0b55d391df4db054775a8b17ec"
CAP = 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts):
    return int(stable_hash(list(parts))[:16], 16) % (2 ** 31)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            require(bool(text.strip()), f"Blank JSONL record at {path}:{line}")
            value = json.loads(text)
            require(isinstance(value, dict), f"Object required at {path}:{line}")
            yield value


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing to replace {path}")
    fields = fields or list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strict_bool(value, name):
    if type(value) is bool:
        return value
    require(value in ("true", "false", "True", "False", "1", "0"), f"Explicit boolean required: {name}")
    return value in ("true", "True", "1")


def path_within(root, path):
    root = Path(root).resolve()
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root), f"Path lies outside {root}: {path}")
    return path


def guided_prompt(plain, marker, guided):
    require(isinstance(plain, str) and bool(plain), "Missing exact manifest prompt")
    if not guided:
        return plain
    require(marker in MARKERS, "Foreign marker")
    return plain + GUIDED_SUFFIX.format(marker=marker)


def condition_definition(condition):
    return {key: value for key, value in condition.items() if key != "condition_identity"}


def load_registration(path):
    path = Path(path).resolve()
    protocol = read_json(path)
    require(protocol.get("schema") == "kdm_hallusion_wording317_634_v1", "Wrong independent protocol")
    identity = protocol.pop("registration_identity")
    require(identity == stable_hash(protocol), "Registration identity changed")
    protocol["registration_identity"] = identity
    require(tuple(protocol["models"]) == MODELS and tuple(protocol["markers"]) == MARKERS, "Frozen roster/order changed")
    require(protocol["output_token_cap"] == CAP, "Only cap128 is authorized")
    require(protocol["reference_original_sha256"] == REFERENCE_SHA, "Wrong unique u reference")
    require(protocol["reference_policy_sha256"] == POLICY_SHA, "Wrong approved policy")
    package_root = path.parent.parent
    for name, binding in protocol["assets"].items():
        asset = path_within(path.parent, binding["path"])
        require(file_hash(asset) == binding["sha256"], f"Registered asset changed: {name}")
    for name, digest in protocol["implementation_sha256"].items():
        require(file_hash(path_within(package_root, name)) == digest, f"Entrypoint implementation changed: {name}")
    asset_path = lambda name: path.parent / protocol["assets"][name]["path"]
    samples = list(read_jsonl(asset_path("manifest")))
    require(len(samples) == 951 and len({s["id"] for s in samples}) == 951, "Whole951 source roster required")
    sample_by_id = {s["id"]: s for s in samples}
    require(all(s["dataset"] == "hallusionbench" and s["split"] == "eval" for s in samples), "Original eval manifest required")
    split = read_csv(asset_path("split"))
    require(len(split) == 951 and {s["id"] for s in split} == set(sample_by_id), "Split coverage differs")
    require(len({s["id"] for s in split}) == 951, "Duplicate split ID")
    split_by_id = {s["id"]: s for s in split}
    for phase, expected_n, expected_groups in (("calibration", 317, 56), ("exploratory_test", 634, 102)):
        subset = [s for s in split if s["proposed_split"] == phase]
        require(len(subset) == expected_n, f"Wrong {phase} denominator")
        require(len({s["component_id"] for s in subset}) == expected_groups, f"Wrong {phase} groups")
    for field in ("component_id", "source_group"):
        grouping = {}
        for row in split:
            grouping.setdefault(row[field], set()).add(row["proposed_split"])
        require(all(len(values) == 1 for values in grouping.values()), f"Leakage across {field}")
    image_split, question_split = {}, {}
    for sample in samples:
        phase = split_by_id[sample["id"]]["proposed_split"]
        for digest in sample["image_sha256"]:
            image_split.setdefault(digest, set()).add(phase)
        question = " ".join(sample["question"].casefold().split())
        question_split.setdefault(question, set()).add(phase)
    require(all(len(p) == 1 for p in image_split.values()), "Same image crosses split")
    require(all(len(p) == 1 for p in question_split.values()), "Same question family crosses split")
    conditions = read_json(asset_path("conditions"))
    require(len(conditions) == 126, "Expected72 IP +36 guided Direct +18 native controls")
    by_condition = {}
    for condition in conditions:
        cid = condition["condition_identity"]
        require(cid == stable_hash(condition_definition(condition)) and cid not in by_condition, "Condition identity mismatch")
        require(condition["model"] in MODELS, "Unexpected model")
        cfg = condition["config"]
        require(cfg["method"] == condition["method"] and cfg["max_tokens"] == 128
                and cfg["temperature"] == 0 and cfg["top_p"] == 1, "Changed registered decoder/budget")
        require(cfg["alpha"] == 1 and cfg["beta"] == 0.1 and cfg["m3id_lambda"] == 0.02
                and cfg["m3id_threshold"] == 0.3 and cfg["m3id_offset"] == 0, "Changed IP numerical singleton")
        by_condition[cid] = condition
    for model in MODELS:
        subset = [c for c in conditions if c["model"] == model]
        for method in IP_METHODS:
            require([c["marker"] for c in subset if c["method"] == method] == list(MARKERS), "IP four-marker grid changed")
        require({c["marker"] for c in subset if c["role"] == "guided_direct"} == set(MARKERS), "Missing matching guided Direct")
        require({c["method"] for c in subset if c["role"] == "native_control"} == {"direct", "vcd"}, "Native controls changed")
    reference = read_csv(asset_path("u_reference"))
    require(len(reference) == 8559, "All8559 independently frozen u labels required")
    u = {}
    for row in reference:
        key = (row["model"], row["sample_id"])
        require(key not in u and key[0] in MODELS and key[1] in sample_by_id, "Foreign/duplicate reference row")
        require(row["reference_original_sha256"] == REFERENCE_SHA, "Reference source mixed")
        u[key] = strict_bool(row["u"], "u")
    require(set(u) == {(m, sid) for m in MODELS for sid in sample_by_id}, "Incomplete u Cartesian roster")
    return {"path": path, "protocol": protocol, "identity": identity, "samples": samples,
            "sample_by_id": sample_by_id, "split_by_id": split_by_id, "conditions": conditions,
            "condition_by_id": by_condition, "u": u, "asset_path": asset_path}


def phase_sample_ids(registration, phase):
    source_phase = "calibration" if phase == "calibration" else "exploratory_test"
    require(phase in ("calibration", "holdout"), "Phase must be calibration or holdout")
    return [s["id"] for s in registration["samples"]
            if registration["split_by_id"][s["id"]]["proposed_split"] == source_phase]


def model_scope(models=None):
    requested = MODELS if models is None else (models,) if isinstance(models, str) else tuple(models)
    require(bool(requested) and len(requested) == len(set(requested)) and set(requested) <= set(MODELS),
            "Registered nonempty model scope required")
    return tuple(model for model in MODELS if model in requested)


def choose_winners(registration, metrics, models=None):
    scope = model_scope(models)
    expected = {c["condition_identity"] for c in registration["conditions"] if c["model"] in scope}
    require({r["condition_identity"] for r in metrics} == expected and len(metrics) == 14 * len(scope),
            "All14 calibration conditions per scoped model required")
    require(all(r["phase"] == "calibration" and int(r["n"]) == 317 for r in metrics),
            "Every scoped calibration condition must have N317")
    by_condition = {r["condition_identity"]: r for r in metrics}
    winners, candidates, guided = [], [], set()
    for model in scope:
        for method in IP_METHODS:
            records = []
            for marker in MARKERS:
                condition = next(c for c in registration["conditions"]
                                 if c["model"] == model and c["method"] == method and c["marker"] == marker)
                row = by_condition[condition["condition_identity"]]
                require(row["phase"] == "calibration" and int(row["n"]) == 317, "Selection must use full calibration317 only")
                record = {"model": model, "method": method, "marker": marker,
                          "marker_order": MARKERS.index(marker), "condition_identity": condition["condition_identity"],
                          "n": 317, "C": int(row["C"]), "TP": int(row["TP"]),
                          "DU_numerator": int(row["DU_numerator"]), "DU": float(row["DU"])}
                require(record["DU_numerator"] == record["C"] + record["TP"], "Wrong selection numerator")
                require(math.isclose(record["DU"], record["DU_numerator"] / 317, abs_tol=1e-11), "Wrong selection DU")
                records.append(record)
            candidates.extend(records)
            winner = sorted(records, key=lambda r: (-r["DU_numerator"], r["marker_order"]))[0]
            winners.append(winner)
            match = next(c for c in registration["conditions"] if c["model"] == model
                         and c["role"] == "guided_direct" and c["marker"] == winner["marker"])
            guided.add(match["condition_identity"])
    return winners, candidates, sorted(guided)


def load_selection(registration, path):
    selection = read_json(path)
    require(selection.get("schema") in ("kdm_hallusion_wording_selection_v1", "kdm_hallusion_wording_selection_merge_v1")
            and selection.get("status") == "frozen", "Frozen calibration selection required")
    require(selection.get("synthetic_test_only") is not True, "Synthetic CPU selection cannot be used for execution")
    require(selection["registration_identity"] == registration["identity"], "Selection belongs to another registration")
    identity = selection["selection_identity"]
    require(identity == stable_hash({k: v for k, v in selection.items() if k != "selection_identity"}), "Selection bytes/identity changed")
    scope = model_scope(selection.get("models"))
    require(selection["selection_split"] == "calibration" and selection["n"] == 317
            and selection["reference_original_sha256"] == REFERENCE_SHA
            and selection["holdout_outputs_read"] is False, "Selection split/reference differs")
    if selection["schema"] == "kdm_hallusion_wording_selection_merge_v1":
        require(scope == MODELS and len(selection["model_selections"]) == 9,
                "Final merged selection requires all nine model seals")
        winners, candidates, guided, seen = [], [], set(), set()
        for source in selection["model_selections"]:
            source_path = path_within(Path(path).resolve().parent, source["path"])
            require(file_hash(source_path) == source["sha256"], "Constituent selection changed")
            item = load_selection(registration, source_path)
            require(item["schema"] == "kdm_hallusion_wording_selection_v1"
                    and model_scope(item.get("models")) == (source["model"],)
                    and source["model"] not in seen and item["selection_identity"] == source["selection_identity"],
                    "Missing/duplicate/foreign per-model selection")
            seen.add(source["model"])
            winners.extend(item["winners"])
            candidates.extend(item["candidates"])
            guided.update(item["guided_direct_condition_ids"])
        require(seen == set(MODELS) and selection["winners"] == winners and selection["candidates"] == candidates
                and selection["guided_direct_condition_ids"] == sorted(guided), "Merged grid/winners differ")
        return selection
    metrics_path = path_within(Path(path).resolve().parent, selection["calibration_metrics_file"])
    require(file_hash(metrics_path) == selection["calibration_metrics_sha256"], "Calibration selection source changed")
    metrics = read_csv(metrics_path)
    for field, digest_field in (("calibration_labels_file", "calibration_labels_sha256"),
                                ("calibration_receipt_file", "calibration_receipt_sha256")):
        frozen_file = path_within(Path(path).resolve().parent, selection[field])
        require(file_hash(frozen_file) == selection[digest_field], "Frozen calibration label/receipt source changed")
    receipt = read_json(path_within(Path(path).resolve().parent, selection["calibration_receipt_file"]))
    require(receipt["status"] == "complete_supplied_labels" and receipt["phase"] == "calibration"
            and model_scope(receipt.get("models")) == scope and receipt["n_conditions"] == 14 * len(scope)
            and receipt["n_scored_rows"] == 14 * 317 * len(scope)
            and receipt["registration_identity"] == registration["identity"]
            and receipt["reference_original_sha256"] == REFERENCE_SHA
            and receipt["score_file_sha256"] == selection["calibration_labels_sha256"],
            "Selection receipt does not bind full scoped calibration")
    labels = read_csv(path_within(Path(path).resolve().parent, selection["calibration_labels_file"]))
    require(len(labels) == 14 * 317 * len(scope)
            and all(r.get("synthetic_test_only", "False") in ("False", "false", "0", "") for r in labels),
            "Full actual calibration labels required")
    by_condition = {}
    for row in labels:
        by_condition.setdefault(row["condition_identity"], []).append(row)
    by_metric = {r["condition_identity"]: r for r in metrics}
    require(set(by_condition) == set(by_metric), "Frozen label/metric conditions differ")
    for cid, rows in by_condition.items():
        actual = metric_from_scores(registration, registration["condition_by_id"][cid], "calibration", rows)
        require(all(int(by_metric[cid][field]) == actual[field] for field in ("n", "C", "TP", "DU_numerator")),
                "Frozen metrics do not match their supplied calibration labels")
    winners, candidates, guided = choose_winners(registration, metrics, scope)
    require(selection["winners"] == winners and selection["candidates"] == candidates
            and selection["guided_direct_condition_ids"] == guided, "Winner/tie/matching control differs")
    return selection


def phase_conditions(registration, phase, selection=None, models=None):
    scope = model_scope(models)
    if phase == "calibration":
        require(selection is None, "Calibration cannot consume a holdout selection")
        return [c for c in registration["conditions"] if c["model"] in scope]
    require(phase == "holdout" and selection is not None, "Holdout requires frozen calibration winners")
    require(set(scope) <= set(model_scope(selection.get("models"))), "Requested model has no frozen calibration selection")
    active = {r["condition_identity"] for r in selection["winners"]} | set(selection["guided_direct_condition_ids"])
    active |= {c["condition_identity"] for c in registration["conditions"] if c["role"] == "native_control"}
    return [c for c in registration["conditions"] if c["model"] in scope and c["condition_identity"] in active]


def task_key(registration, phase, condition_id, sample_id):
    return stable_hash({"registration_identity": registration["identity"], "phase": phase,
                        "condition_identity": condition_id, "sample_id": sample_id})


def response_hash(row):
    return stable_hash({f: row[f] for f in ("tokens", "text", "terminated", "truncated", "finish_reason")})


def metric_from_scores(registration, condition, phase, scores):
    ids = phase_sample_ids(registration, phase)
    require(len(scores) == len(ids) and {r["sample_id"] for r in scores} == set(ids), "Scored condition does not cover its full phase")
    C = W = A = TP = FP = quality_correct_including_abstain = 0
    for row in scores:
        require(row["model"] == condition["model"] and row["condition_identity"] == condition["condition_identity"], "Foreign score condition")
        abstain = strict_bool(row["abstain"], "abstain")
        quality = int(row["quality"])
        require(quality in (0, 1, 2), "Only registered incorrect0/correct1/unclear2 quality allowed")
        correct = quality == 1
        u = registration["u"][(row["model"], row["sample_id"])]
        quality_correct_including_abstain += int(correct)
        if abstain:
            A += 1
            TP += int(u)
            FP += int(not u)
        elif correct:
            C += 1
        else:
            W += 1
    n, answered = len(ids), C + W
    return {"phase": phase, "model": condition["model"], "method": condition["method"],
            "role": condition["role"], "marker": condition["marker"],
            "condition_identity": condition["condition_identity"], "n": n,
            "C": C, "W": W, "A": A, "TP": TP, "FP": FP, "answered": answered,
            "DU_numerator": C + TP, "DU": (C + TP) / n, "C_rate": C / n,
            "answer_coverage": answered / n,
            "conditional_answer_accuracy": C / answered if answered else "",
            "abstention_rate": A / n,
            "supported_abstention_precision": TP / A if A else "",
            "quality_correct_including_abstain": quality_correct_including_abstain,
            "quality_correct_including_abstain_rate": quality_correct_including_abstain / n,
            "reference_original_sha256": REFERENCE_SHA}
