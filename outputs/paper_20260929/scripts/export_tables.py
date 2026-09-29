"""Stream frozen KDM scores into compact paper tables; perform existing-key checks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COND = ("model", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
SCORE = "outputs/annotations/main_results/score_rows.jsonl.gz"
AUTO = "outputs/annotations/main_results/automatic_behavior.jsonl.gz"
SELECTIONS = "outputs/analysis/main_results/controls/selections.jsonl.gz"
ACCEPTED = "outputs/annotations/reference_gt/reference_gt.jsonl"
UNIFORM = "outputs/annotations/reference_gt/uniform_reference_gt.jsonl"
REVIEWED = "outputs/annotations/main_results/reviewed_answers.jsonl"
METRICS = "outputs/analysis/main_results/condition_metrics.csv"


def lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    if value is None:
        return None
    raise ValueError(f"Invalid stored boolean: {value!r}")


def condition(row):
    return tuple(boolean(row[key]) if key in {"guided", "reference_guided"}
                 else int(row[key]) if key == "replicate" else row[key] for key in COND)


class StreamParquet:
    def __init__(self, path, schema):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
        self.schema = schema
        self.pending = []
        self.count = 0

    def add(self, row):
        self.pending.append(row)
        self.count += 1
        if len(self.pending) >= 10000:
            self.flush()

    def flush(self):
        if self.pending:
            self.writer.write_table(pa.Table.from_pylist(self.pending, schema=self.schema))
            self.pending.clear()

    def close(self):
        self.flush()
        self.writer.close()


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    if (out / "scores.parquet").exists() or (work / "index.sqlite").exists():
        raise FileExistsError("Choose a new version subdirectory for an existing export.")
    checks = {"check_type": "CPU serialization, key joins, and count checks on existing frozen files",
              "discrepancies": [], "field_availability": []}

    def check(name, actual, expected):
        passed = actual == expected
        checks[name] = {"actual": actual, "expected": expected, "passed": passed}
        if not passed:
            checks["discrepancies"].append({"check": name, "actual": actual, "expected": expected})
        return passed

    manifest = json.loads((root / "data/manifest.json").read_text())
    source_rows, source_id = [], {}

    def source(rel, role, n=None):
        if rel not in source_id:
            sid = len(source_id) + 1
            p = root / rel
            source_id[rel] = sid
            source_rows.append({"source_file_id": sid, "role": role, "project_relative_path": rel,
                                "real_path": str(p.resolve()), "bytes": p.stat().st_size if p.exists() else None,
                                "known_rows": n, "exists": p.exists()})
        return source_id[rel]

    for entry in sorted(manifest["sources"], key=lambda row: row["path"]):
        source(entry["path"], "formal_response", entry["rows"])
    for rel, role in [(SCORE, "frozen_final_score"), (AUTO, "stored_behavior"),
                      (SELECTIONS, "frozen_copy_selection"), (ACCEPTED, "accepted_reference"),
                      (UNIFORM, "uniform_reference"), (REVIEWED, "reviewed_QA"),
                      (METRICS, "condition_metrics"), ("data/current/food101.jsonl", "sample_manifest")]:
        source(rel, role)
    for model in sorted(entry["model"] for entry in manifest["sources"]):
        source(f"data/responses/independent/{model}.jsonl.gz", "independent_attempts", 48480)
        source(f"data/responses/candidate/{model}.jsonl.gz", "candidate_rank_records", 4848)
        source(f"configs/runtime/{model}.json", "registered_runtime")

    with (root / METRICS).open(encoding="utf-8", newline="") as f:
        original_metrics = list(csv.DictReader(f))
    metric_by_key = {condition(row): row for row in original_metrics}
    condition_ids = {key: number for number, key in enumerate(sorted(metric_by_key), 1)}
    condition_dict = {number: dict(zip(COND, key)) for key, number in condition_ids.items()}
    check("source_condition_keys", len(condition_ids), 352)
    samples = {row["id"]: row for _, row in lines(root / "data/current/food101.jsonl")}
    accepted = {(row["model"], row["sample_id"]): (number, row) for number, row in lines(root / ACCEPTED)}
    uniform = {(row["model"], row["sample_id"]): (number, row) for number, row in lines(root / UNIFORM)}
    check("accepted_reference_rows", len(accepted), 24240)
    check("uniform_reference_rows", len(uniform), 24240)
    check("reference_key_sets_match", set(accepted) == set(uniform), True)
    references = []
    for key in sorted(uniform):
        un, u = uniform[key]
        an, a = accepted.get(key, (None, {}))
        references.append({"model": key[0], "sample_id": key[1], "split": u.get("split"),
                           "target_class": u.get("target"), "uniform_reference": u.get("gt"),
                           "accepted_reference": a.get("gt"), "gold_rank": u.get("gold_rank"),
                           "independent_correct_attempts": u.get("independent_correct_attempts_exact"),
                           "independent_attempts": u.get("attempts"),
                           "uniform_source_file_id": source_id[UNIFORM], "uniform_source_line": un,
                           "accepted_source_file_id": source_id[ACCEPTED], "accepted_source_line": an,
                           "uniform_original_json": dump(u), "accepted_original_json": dump(a)})
    pq.write_table(pa.Table.from_pylist(references), out / "references.parquet", compression="zstd")
    refs = {(x["model"], x["sample_id"]): x for x in references}
    reviewed = {row["qa_key"]: (number, row) for number, row in lines(root / REVIEWED)}
    db = sqlite3.connect(work / "index.sqlite")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE automatic(model TEXT,response_key TEXT,payload TEXT,PRIMARY KEY(model,response_key)) WITHOUT ROWID")
    batch = []
    auto_count = 0
    for _, row in lines(root / AUTO):
        batch.append((row["model"], row["key"], dump(row)))
        auto_count += 1
        if len(batch) >= 10000:
            db.executemany("INSERT INTO automatic VALUES(?,?,?)", batch)
            batch.clear()
    db.executemany("INSERT INTO automatic VALUES(?,?,?)", batch)
    db.commit()
    check("automatic_behavior_rows", auto_count, 853248)
    db.execute("CREATE TABLE responses(model TEXT,response_key TEXT,condition_id INTEGER,sample_id TEXT,target_class TEXT,correct_canonical INTEGER,correct_literal INTEGER,abstain INTEGER,uniform_reference INTEGER,accepted_reference INTEGER,source_file_id INTEGER,source_line INTEGER,qa_id INTEGER,seed INTEGER,detail_id INTEGER,binding_id INTEGER,score_line INTEGER,behavior TEXT,score_reason TEXT,PRIMARY KEY(model,response_key),UNIQUE(condition_id,sample_id))")
    names = ["condition_id", "sample_id", "target_class", "correct_canonical", "correct_literal", "abstain",
             "uniform_reference", "accepted_reference", "source_file_id", "source_line", "qa_id", "seed",
             "detail_id", "binding_id", "score_source_line", "behavior", "score_reason"]
    types = [pa.int16(), pa.string(), pa.string(), pa.bool_(), pa.bool_(), pa.bool_(), pa.bool_(), pa.bool_(),
             pa.int16(), pa.int32(), pa.int32(), pa.int64(), pa.int32(), pa.int16(), pa.int32(), pa.string(), pa.string()]
    scores = StreamParquet(out / "scores.parquet", pa.schema(list(zip(names, types))))
    qa_ids, details, detail_values, bindings, binding_values = {}, {}, [], {}, []
    counts = defaultdict(Counter)
    classes = defaultdict(Counter)
    condition_bindings = defaultdict(set)
    split_counts, auto_labels, auto_sources, unknowns = Counter(), Counter(), Counter(), Counter()
    qa_extracted_null = auto_joined = uniform_joined = accepted_joined = 0
    batch = []
    detail_keys = ("literal_extracted_names", "literal_extracted_name", "canonical_class_candidates",
                   "canonical_name_in_primary", "review_sources", "review_statuses", "multiple_primary_classes")
    for score_line, row in lines(root / SCORE):
        key = condition(row)
        cid = condition_ids[key]
        sid = row["sample_id"]
        meta = samples.get(sid, {})
        split_counts[str(meta.get("split"))] += 1
        if meta.get("class") != row["target_class"]:
            checks["discrepancies"].append({"check": "sample_target_join", "score_line": score_line, "sample_id": sid})
        ref = refs.get((row["model"], sid), {})
        uniform_joined += (row["model"], sid) in uniform
        accepted_joined += (row["model"], sid) in accepted
        automatic = db.execute("SELECT payload FROM automatic WHERE model=? AND response_key=?", (row["model"], row["key"])).fetchone()
        auto = json.loads(automatic[0]) if automatic else {}
        auto_joined += automatic is not None
        if auto and auto.get("sample_id") != sid:
            checks["discrepancies"].append({"check": "automatic_sample_join", "score_line": score_line})
        auto_labels[str(auto.get("behavior_label"))] += 1
        auto_sources[str(auto.get("source"))] += 1
        qid = qa_ids.setdefault(row["qa_key"], len(qa_ids) + 1)
        detail = {k: row.get(k) for k in detail_keys}
        detail.update({"stored_behavior_label": auto.get("behavior_label"), "stored_behavior_abstain": auto.get("abstain"),
                       "stored_behavior_source": auto.get("source")})
        signature = dump(detail)
        if signature not in details:
            details[signature] = len(details) + 1
            detail_values.append({"detail_id": details[signature], **{k: dump(v) for k, v in detail.items()}})
        did = details[signature]
        binding = {k: row.get(k) for k in ("main_prompt_sha256", "reference_prompt_sha256", "neutral_prompt_sha256", "config_sha256")}
        bsig = dump(binding)
        if bsig not in bindings:
            bindings[bsig] = len(bindings) + 1
            binding_values.append({"binding_id": bindings[bsig], **binding, "first_source_file_id": source_id[row["source_path"]], "first_source_line": row["source_line"]})
        bid = bindings[bsig]
        condition_bindings[cid].add(bid)
        cc, cl, ab = (boolean(row.get(k)) for k in ("canonical_name_in_primary_score", "literal_extracted_name_score", "abstain"))
        for field, value in [("canonical", cc), ("literal", cl), ("abstain", ab)]:
            unknowns[field] += value is None
        uq, aq = boolean(ref.get("uniform_reference")), boolean(ref.get("accepted_reference"))
        score = dict(zip(names, [cid, sid, row["target_class"], cc, cl, ab, uq, aq,
                               source_id[row["source_path"]], row["source_line"], qid, row.get("seed"), did, bid,
                               score_line, row.get("behavior"), row.get("score_reason")]))
        scores.add(score)
        counts[cid].update({"n": 1, "canonical_correct": cc is True, "literal_correct": cl is True,
                            "abstain_true": ab is True, "reference_gt_true": aq is True, "uniform_gt_true": uq is True})
        classes[cid][row["target_class"]] += 1
        qa_extracted_null += row.get("canonical_name_in_primary") is None
        batch.append((row["model"], row["key"], cid, sid, row["target_class"], cc, cl, ab, uq, aq,
                      score["source_file_id"], score["source_line"], qid, score["seed"], did, bid, score_line,
                      row.get("behavior"), row.get("score_reason")))
        if len(batch) >= 10000:
            db.executemany("INSERT INTO responses VALUES(" + ",".join("?" * 19) + ")", batch)
            batch.clear()
    db.executemany("INSERT INTO responses VALUES(" + ",".join("?" * 19) + ")", batch)
    db.commit()
    scores.close()
    check("score_rows", scores.count, 853248)
    check("uniform_joined_rows", uniform_joined, 853248)
    check("accepted_joined_rows", accepted_joined, 853248)
    check("stored_behavior_joined_rows", auto_joined, 853248)
    check("score_splits", dict(split_counts), {"eval": 853248})
    check("final_field_unknown_rows", dict(unknowns), {"canonical": 0, "literal": 0, "abstain": 0})
    row_checks = []
    condition_export = []
    for key, cid in condition_ids.items():
        expected = metric_by_key[key]
        actual = counts[cid]
        diffs = {k: {"actual": actual[k], "expected": int(expected[k])} for k in actual if actual[k] != int(expected[k])}
        if len(classes[cid]) != 101 or set(classes[cid].values()) != {24}:
            diffs["class_balance"] = {"classes": len(classes[cid]), "counts": sorted(set(classes[cid].values()))}
        unique = db.execute("SELECT COUNT(DISTINCT sample_id) FROM responses WHERE condition_id=?", (cid,)).fetchone()[0]
        if unique != 2424:
            diffs["unique_samples"] = unique
        row_checks.append({"condition_id": cid, "n": actual["n"], "unique_samples": unique, "classes": len(classes[cid]), "differences": diffs})
        if diffs:
            checks["discrepancies"].append({"check": "condition_metrics", "condition_id": cid, "differences": diffs})
        condition_export.append({"condition_id": cid, **condition_dict[cid], "split": "eval", "main_marker": key[3],
                                 "complete_condition_key": dump(list(key)), "n": actual["n"],
                                 "binding_ids": dump(sorted(condition_bindings[cid]))})
    save_csv(out / "conditions.csv", condition_export)
    save_json(out / "bindings.json", binding_values)
    pq.write_table(pa.Table.from_pylist(detail_values), out / "score_details.parquet", compression="zstd")
    qa_rows = []
    for key, qid in qa_ids.items():
        number, row = reviewed.get(key, (None, {}))
        qa_rows.append({"qa_id": qid, "qa_key": key, "question": row.get("question"), "answer": row.get("answer"),
                        "review_variants_json": dump(row["variants"]) if "variants" in row else None,
                        "review_source_file_id": source_id[REVIEWED] if row else None, "review_source_line": number})
    pq.write_table(pa.Table.from_pylist(qa_rows), out / "qa.parquet", compression="zstd")
    checks["field_availability"] = [
        {"field": "score.canonical_name_in_primary", "null_rows": qa_extracted_null, "source": SCORE,
         "meaning": "Original nullable extracted-name field retained; final correctness is taken directly from the separate decided score."},
        {"field": "qa.question/answer", "available_QA": sum(x["answer"] is not None for x in qa_rows), "total_QA": len(qa_rows),
         "source": REVIEWED, "meaning": "Exact reviewed QA text retained where present; all formal rows have current raw source file and line locators."},
    ]
    checks["stored_behavior_labels"] = dict(auto_labels)
    checks["stored_behavior_sources"] = dict(auto_sources)
    checks["condition_checks"] = row_checks

    with (root / "outputs/analysis/main_results/controls/condition_metrics.jsonl").open() as f:
        original_controls = [json.loads(line) for line in f if line.strip()]
    controls_by_key = {condition(row): row for row in original_controls}
    control_ids = {key: number for number, key in enumerate(sorted(controls_by_key), 1)}
    control_fields = ["control_condition_id", "sample_id", "selected_condition_id", "base_condition_id", "direct_condition_id",
                      "selected_source_file_id", "selected_source_line", "qa_id", "selection_reason", "direct_abstain", "selected_abstain", "source_line"]
    control_schema = pa.schema(list(zip(control_fields, [pa.int16(), pa.string(), pa.int16(), pa.int16(), pa.int16(), pa.int16(),
                                                         pa.int32(), pa.int32(), pa.string(), pa.bool_(), pa.bool_(), pa.int32()])))
    selections = StreamParquet(out / "control_selections.parquet", control_schema)
    control_counts = defaultdict(Counter)
    control_seen = defaultdict(set)
    db.execute("CREATE TABLE controls(control_condition_id INTEGER,sample_id TEXT,selected_condition_id INTEGER,source_line INTEGER,PRIMARY KEY(control_condition_id,sample_id))")
    batch = []
    for line_num, row in lines(root / SELECTIONS):
        cid = control_ids[condition(row)]
        records = []
        for field in ("selected_key", "treatment_key", "direct_key"):
            match = db.execute("SELECT * FROM responses WHERE model=? AND response_key=?", (row["model"], row[field])).fetchone()
            if match is None:
                checks["discrepancies"].append({"check": "control_response_join", "source_line": line_num, "field": field})
            records.append(match)
        selected, treatment, direct = records
        if any(x is None for x in records):
            continue
        for match in records:
            if match["sample_id"] != row["sample_id"]:
                checks["discrepancies"].append({"check": "control_sample_identity", "source_line": line_num})
        selections.add(dict(zip(control_fields, [cid, row["sample_id"], selected["condition_id"], treatment["condition_id"],
                                                  direct["condition_id"], selected["source_file_id"], selected["source_line"], selected["qa_id"],
                                                  row["selection_reason"], boolean(row["direct_abstain"]), boolean(row["selected_abstain"]), line_num])))
        control_counts[cid].update({"n": 1, "accuracy_correct_numerator": selected["correct_canonical"] == 1,
                                    "abstention_n": selected["abstain"] == 1})
        control_seen[cid].add(row["sample_id"])
        batch.append((cid, row["sample_id"], selected["condition_id"], line_num))
        if len(batch) >= 10000:
            db.executemany("INSERT INTO controls VALUES(?,?,?,?)", batch)
            batch.clear()
    db.executemany("INSERT INTO controls VALUES(?,?,?,?)", batch)
    db.commit()
    selections.close()
    check("control_selections", selections.count, 484800)
    check("control_conditions", len(control_ids), 200)
    control_export = []
    for key, cid in control_ids.items():
        values, expected = control_counts[cid], controls_by_key[key]
        diffs = {k: {"actual": values[k], "expected": expected[k]} for k in values if values[k] != expected[k]}
        if len(control_seen[cid]) != 2424:
            diffs["unique_samples"] = len(control_seen[cid])
        if diffs:
            checks["discrepancies"].append({"check": "control_metrics", "control_condition_id": cid, "differences": diffs})
        control_export.append({"control_condition_id": cid, "control_type": "copy_original_abstention", **dict(zip(COND, key)),
                               "complete_base_condition_key": dump(list(key)), "base_condition_id": condition_ids[key], "split": "eval", "n": values["n"]})
    save_csv(out / "control_conditions.csv", control_export)
    save_csv(out / "sources.csv", source_rows)

    def find(model, method, marker, kind):
        matches = [cid for cid, row in condition_dict.items() if row["model"] == model and row["method"] == method
                   and row["marker"] == marker and row["reference_marker"] == marker and row["kind"] == kind and row["replicate"] == 0]
        if len(matches) != 1:
            raise ValueError((model, method, marker, kind, matches))
        return matches[0]

    def members(cid):
        return {row["sample_id"]: dict(row) for row in db.execute("SELECT * FROM responses WHERE condition_id=?", (cid,))}

    cached = {}

    def get_members(cid):
        if cid not in cached:
            cached[cid] = members(cid)
        return cached[cid]

    core = []
    for model, marker, expected in [("llava16_mistral", "UNKNOWN", [689, 852, 852, 287, 33, 61, 287, 35, 62]),
                                     ("minicpm26", "UNCLEAR", [733, 873, 903, 374, 239, 249, 428, 270, 281])]:
        ids = [find(model, "direct", marker, "main"), find(model, "vcd", marker, "main"), find(model, "instruction_vcd", marker, "instruction_preserving")]
        d, v, ip = [get_members(cid) for cid in ids]
        U = {sid for sid, row in d.items() if row["abstain"] and row["uniform_reference"]}
        A = {sid for sid, row in d.items() if row["abstain"] and row["accepted_reference"]}
        actual = [sum(row["correct_canonical"] for row in collection.values()) for collection in (d, v, ip)]
        actual += [len(U), sum(v[sid]["abstain"] for sid in U), sum(ip[sid]["abstain"] for sid in U),
                   len(A), sum(v[sid]["abstain"] for sid in A), sum(ip[sid]["abstain"] for sid in A)]
        check("core_" + model, actual, expected)
        core.append({"model": model, "marker": marker, "condition_ids": ids, "counts": actual})
        if model == "minicpm26":
            corrected = {sid for sid in d if not d[sid]["correct_canonical"] and v[sid]["correct_canonical"]}
            check("minicpm_vcd_corrections", [len(corrected), sum(ip[sid]["correct_canonical"] for sid in corrected)], [226, 212])
            control_id = control_ids[tuple(condition_dict[ids[1]][k] for k in COND)]
            check("minicpm_copy_control_correct", control_counts[control_id]["accuracy_correct_numerator"], 722)
    checks["core_counts"] = core
    pair_checks = []
    for number, pair in lines(root / "outputs/analysis/main_results/paired_comparisons.jsonl"):
        if pair["comparison"] != "instruction_vs_base_method":
            continue
        ctx = pair["context"]
        ia = find(ctx["model"], pair["method_a"], ctx["marker"], "instruction_preserving")
        ib = find(ctx["model"], pair["method_b"], ctx["marker"], "main")
        idirect = find(ctx["model"], "direct", ctx["marker"], "main")
        a, b, d = [get_members(x) for x in (ia, ib, idirect)]
        accuracy = sum(a[s]["correct_canonical"] - b[s]["correct_canonical"] for s in a) / len(a)
        values = {"accuracy_difference": accuracy}
        wanted = {"accuracy_difference": pair["resolved_accuracy_difference_a_minus_b"]}
        for ref_name, stored in [("uniform_reference", "direct_reasonable_abstention_set_preservation_difference_uniform_gt"),
                                 ("accepted_reference", "direct_reasonable_abstention_set_preservation_difference_reference_gt")]:
            group = [s for s in d if d[s]["abstain"] and d[s][ref_name]]
            num = sum(a[s]["abstain"] - b[s]["abstain"] for s in group)
            values[ref_name + "_num"] = num
            values[ref_name + "_den"] = len(group)
            wanted[ref_name + "_num"] = pair[stored]["numerator"]
            wanted[ref_name + "_den"] = pair[stored]["denominator"]
        diffs = {k: [values[k], wanted[k]] for k in values if abs(values[k] - wanted[k]) > 1e-12}
        if diffs:
            checks["discrepancies"].append({"check": "instruction_pair_point_estimate", "source_line": number, "differences": diffs})
        pair_checks.append({"source_line": number, "condition_a": ia, "condition_b": ib, "direct_condition_id": idirect,
                            **values, "passed": not diffs})
    check("instruction_pair_checks", len(pair_checks), 40)
    checks["instruction_pair_point_checks"] = pair_checks

    candidates, group_availability, selected_samples = [], [], set()

    def selected_response(cid, sid):
        row = dict(db.execute("SELECT * FROM responses WHERE condition_id=? AND sample_id=?", (cid, sid)).fetchone())
        detail = detail_values[row["detail_id"] - 1]
        row["source_file"] = source_rows[row["source_file_id"] - 1]["project_relative_path"]
        row["qa_key"] = next_key_by_id[row["qa_id"]]
        row["score_details"] = {k: json.loads(v) for k, v in detail.items() if k != "detail_id"}
        return row

    next_key_by_id = {value: key for key, value in qa_ids.items()}
    groups = [("original_uniform_abstention_preserved", 2), ("vcd_correction_retained", 2), ("ip_vcd_new_correct", 1)]
    for group, per_model in groups:
        for model, marker in [("llava16_mistral", "UNKNOWN"), ("minicpm26", "UNCLEAR")]:
            ids = [find(model, "direct", marker, "main"), find(model, "vcd", marker, "main"), find(model, "instruction_vcd", marker, "instruction_preserving")]
            d, v, ip = [get_members(cid) for cid in ids]
            if group == "original_uniform_abstention_preserved":
                available = [s for s in d if d[s]["abstain"] and d[s]["uniform_reference"] and not v[s]["abstain"] and not v[s]["correct_canonical"] and ip[s]["abstain"]]
            elif group == "vcd_correction_retained":
                available = [s for s in d if not d[s]["correct_canonical"] and v[s]["correct_canonical"] and ip[s]["correct_canonical"]]
            else:
                available = [s for s in d if not v[s]["correct_canonical"] and ip[s]["correct_canonical"]]
            group_availability.append({"group": group, "model": model, "marker": marker, "available_inputs": len(available)})
            used_classes = set()
            chosen = []
            for sid in sorted(available):
                if sid in selected_samples or samples[sid]["class"] in used_classes:
                    continue
                chosen.append(sid);used_classes.add(samples[sid]["class"])
                if len(chosen) == per_model:
                    break
            for sid in chosen:
                selected_samples.add(sid)
                candidates.append({"group": group, "model": model, "marker": marker, "sample_id": sid, "ids": ids})
    for model in ["qwen25vl", "qwen35_4b"]:
        marker = "UNKNOWN"
        ids = [find(model, "direct", marker, "main"), find(model, "vcd", marker, "main"), find(model, "instruction_vcd", marker, "instruction_preserving")]
        d, v, ip = [get_members(cid) for cid in ids]
        available = [s for s in d if d[s]["correct_canonical"] and not v[s]["correct_canonical"] and ip[s]["correct_canonical"]]
        group_availability.append({"group": "direct_correct_restored_by_ip", "model": model, "marker": marker, "available_inputs": len(available)})
        choice = next((s for s in sorted(available) if s not in selected_samples), None)
        if choice:
            selected_samples.add(choice)
            candidates.append({"group": "direct_correct_restored_by_ip", "model": model, "marker": marker, "sample_id": choice, "ids": ids})
    requests = []
    for number, case in enumerate(candidates, 1):
        model, sid, ids = case["model"], case["sample_id"], case.pop("ids")
        ref, sample = refs[(model, sid)], samples[sid]
        responses = {name: selected_response(cid, sid) for name, cid in zip(("direct", "vcd", "ip_vcd"), ids)}
        coid = control_ids[tuple(condition_dict[ids[1]][k] for k in COND)]
        selected = db.execute("SELECT * FROM controls WHERE control_condition_id=? AND sample_id=?", (coid, sid)).fetchone()
        responses["copy_control"] = {**selected_response(selected["selected_condition_id"], sid),
                                     "control_condition_id": coid, "selected_condition_id": selected["selected_condition_id"],
                                     "selection_source_file": SELECTIONS, "selection_source_line": selected["source_line"]}
        requests.append({"case_id": f"case_{number:02d}", **case, "target_class": sample["class"],
                         **{k: ref[k] for k in ("uniform_reference", "accepted_reference", "gold_rank", "independent_correct_attempts", "independent_attempts")},
                         "image_source": sample["image_path"], "exact_question": sample["question"], "responses": responses})
    save_json(work / "case_requests.json", {"selection_rule": "Frozen scores only; sorted sample IDs with class diversity; globally distinct inputs",
                                           "cases": requests, "group_availability": group_availability})
    save_csv(out / "case_group_availability.csv", group_availability)
    check("distinct_case_inputs", len({x["sample_id"] for x in requests}), len(requests))
    checks["case_requests"] = len(requests)
    (out / "baseline_tables").mkdir(exist_ok=True)
    for rel in [METRICS, "outputs/analysis/main_results/paired_comparisons.jsonl",
                "outputs/analysis/main_results/controls/condition_metrics.jsonl", "outputs/analysis/main_results/controls/instruction_pairs.jsonl"]:
        name = ("copy_" if "/controls/" in rel else "") + Path(rel).name
        shutil.copyfile(root / rel, out / "baseline_tables" / name)
    checks["qa_rows"] = len(qa_rows)
    checks["detail_dictionary_rows"] = len(detail_values)
    checks["binding_dictionary_rows"] = len(binding_values)
    checks["passed"] = not checks["discrepancies"]
    save_json(out / "table_checks.json", checks)
    db.close()
    print(json.dumps({"scores": scores.count, "conditions": len(condition_ids), "references": len(references),
                      "control_selections": selections.count, "cases": len(requests), "discrepancies": len(checks["discrepancies"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.root, args.out or args.root / "outputs/paper_20260929")
