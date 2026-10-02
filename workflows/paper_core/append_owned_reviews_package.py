"""Copy a review package and append only an accepted finite owned-key delta."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within


def require(test, reason):
    if not test:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pointer(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            yield line, json.loads(raw), raw


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "score-dir", "binding-dir", "prepare", "root-source-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This packaging task must explicitly hide GPUs")
    paths = {k: within(ROOT, getattr(args, k.replace("-", "_"))) for k in
             ("package", "score-dir", "binding-dir", "prepare", "root-source-dir", "output")}
    source, output = paths["package"], paths["output"]
    require(source != output and not output.exists(), "A package destination must be exclusive")
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    started = time.monotonic()
    score_receipt = load(paths["score-dir"] / "receipt.json")
    audit = load(paths["score-dir"] / "application_audit.json")
    require(score_receipt["passed"] and audit["passed"] and audit["accepted_actual_Luna_QA"] == 500
            and audit["actual_changed_generation_keys"] == 1023 and score_receipt["rows"] == 292541
            and audit["Food_identical_objects"] == 177333
            and audit["unmatched_Viz_identical_objects"] == 114185,
            "The accepted score delta has not passed its finite-source audit")
    score_path = paths["score-dir"] / "score_rows.jsonl.gz"
    delta_path = paths["score-dir"] / "actual_review_delta.jsonl.gz"
    for path in (score_path, delta_path):
        require(file_hash(path) == score_receipt["outputs"][path.name], "An immutable accepted score source changed")
    delta = {row["input_score_line"]: row for _, row, _ in read_jsonl(delta_path)}
    require(len(delta) == 1023 and len({r["key"] for r in delta.values()}) == 1023
            and len({r["qa_key"] for r in delta.values()}) == 500, "Accepted finite delta ownership repeats or changed")
    builder = import_file("review_builder", source / "scripts/build_existing_review_staging.py")
    projector = import_file("review_projector", source / "scripts/finalize_review_projection.py")
    receipts = load(source / "compact/ledger_receipts.json")
    previous_score = Path(receipts["extended_received292541"]["source_path"])
    require(str(previous_score) == str(ROOT / score_receipt["input_cohorts"][0]["path"])
            and receipts["extended_received292541"]["rows"] == 292541,
            "Package v5 does not point to the accepted delta's original score cohort")
    tables = {name: pq.read_table(source / "compact" / (name + "_dictionary.parquet"))
              for name in ("condition", "sample", "qa", "identity", "source", "label")}
    dictionaries = {name: {row[name + "_id"]: json.loads(row["original_json"])
                          for row in table.to_pylist()} for name, table in tables.items()}
    old_ledger = pq.read_table(source / "compact/ledgers/extended_received292541.parquet")
    require(old_ledger.num_rows == 292541 and old_ledger["ledger_line"].to_pylist() == list(range(1, 292542)),
            "The compact ledger no longer has the original score line mapping")
    records = old_ledger.to_pylist()
    label_rows = tables["label"].to_pylist()
    require([r["label_id"] for r in label_rows] == list(range(len(label_rows))), "Old label IDs are not contiguous")
    labels = [r["original_json"] for r in label_rows]
    label_pool = {text: i for i, text in enumerate(labels)}
    require(len(label_pool) == len(labels), "The projected label dictionary repeats values")
    omitted = set(receipts["extended_received292541"]["source_only_digest_or_identifier_fields"])

    def project_label(row):
        excluded = set(builder.COND_FIELDS + builder.IDENTITY_FIELDS + builder.SOURCE_FIELDS + builder.SCALAR_FIELDS)
        excluded.update(builder.RECORD_ONLY_FIELDS)
        excluded.update(("sample_id", "target_class", "question", "answer", "source_line"))
        payload = {k: v for k, v in row.items() if k not in excluded and not k.endswith("sha256")}
        payload = builder.remove_digests(payload, "label", omitted)
        source_only = sorted(set(payload) & projector.SOURCE_ONLY_LABEL_FIELDS)
        science = {k: v for k, v in payload.items() if k not in projector.SOURCE_ONLY_LABEL_FIELDS}
        science["source_only_original_fields"] = source_only
        return projector.encode(science)

    observed, model_keys, changed_index = set(), Counter(), []
    with gzip.open(score_path, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            if line not in delta:
                continue
            row, change, compact = json.loads(raw), delta[line], records[line - 1]
            condition = {k: row.get(k) for k in builder.COND_FIELDS}
            condition["original_present_fields"] = [k for k in builder.COND_FIELDS if k in row]
            require(row["key"] == change["key"] and row["qa_key"] == change["qa_key"]
                    and row["dataset"] == "vizwiz"
                    and condition == dictionaries["condition"][compact["condition_id"]]
                    and {"question": row["question"], "answer": row["answer"]} == dictionaries["qa"][compact["qa_id"]]
                    and {k: row[k] for k in builder.IDENTITY_FIELDS if k in row} == dictionaries["identity"][compact["identity_id"]],
                    "The accepted delta changed a complete QA, condition, or actual runtime identity")
            source_object = {k: row[k] for k in builder.SOURCE_FIELDS if k in row}
            if isinstance(source_object.get("source"), dict):
                source_object["source"] = {k: v for k, v in source_object["source"].items()
                                           if k not in ("line", "key", "line_sha256")}
            require(source_object == dictionaries["source"][compact["source_id"]], "Accepted raw source identity changed")
            old = {**row, **change["previous"]}
            require(project_label(old) == labels[compact["label_id"]]
                    and compact["abstain"] == old["abstain"]
                    and compact["answer_quality_credit"] == old["answer_quality_credit"]
                    and compact["official_consensus_raw_credit"] == row.get("official_consensus_raw_credit")
                    and compact["official_answerable"] == builder.bool_or_null(row.get("annotated_answerable"), "official answerable"),
                    "Old package scalar values or projected labels do not map to the accepted original score")
            text = project_label(row)
            if text not in label_pool:
                label_pool[text] = len(labels)
                labels.append(text)
            records[line - 1] = {**compact, "label_id": label_pool[text],
                "abstain": row["abstain"], "answer_quality_credit": row["answer_quality_credit"]}
            changed_index.append({"ledger_line": line, "key": row["key"], "qa_key": row["qa_key"],
                "model": row["model"], "sample_id": row["sample_id"], "condition_id": compact["condition_id"],
                "qa_id": compact["qa_id"], "old_label_id": compact["label_id"], "new_label_id": label_pool[text],
                "old_abstain": compact["abstain"], "new_abstain": row["abstain"],
                "old_quality": compact["answer_quality_credit"], "new_quality": row["answer_quality_credit"],
                "original_score_line_sha256": change["input_score_line_sha256"],
                "new_score_line_sha256": hashlib.sha256(raw).hexdigest(),
                "accepted_delta_file": str(delta_path), "new_score_file": str(score_path)})
            observed.add(row["key"])
            model_keys[row["model"]] += 1
    require(observed == {r["key"] for r in delta.values()}, "Accepted score input omitted an owned member")
    print(json.dumps({"event": "original_dictionary_mapping_and_owned_source_validated", "keys": 1023}), flush=True)
    new_ledger = pa.Table.from_pylist(records, schema=old_ledger.schema)
    new_labels = pa.table({"label_id": pa.array(range(len(labels)), type=pa.int32()), "original_json": labels})
    require(new_labels.slice(0, len(label_rows)).equals(tables["label"]), "An existing label ID mapping changed")
    changed_positions = {r["ledger_line"] - 1 for r in changed_index}
    for field in old_ledger.column_names:
        if field not in ("label_id", "abstain", "answer_quality_credit"):
            require(old_ledger[field].equals(new_ledger[field]), "A frozen compact scientific/identity field changed: " + field)
        else:
            before, after = old_ledger[field].to_pylist(), new_ledger[field].to_pylist()
            require(all(a == b for i, (a, b) in enumerate(zip(before, after)) if i not in changed_positions),
                    "A compact field outside the actual owned delta changed")
    shutil.copytree(source, output)
    history = output / "history/package_v5"
    history.mkdir(parents=True, exist_ok=False)
    for name in ("STAGING_MANIFEST.json", "REVIEW_PROJECTION_RECEIPT.json", "scientific_decision_coverage.csv",
                 "extended_label_coverage.csv", "nine_model_coverage.csv", "nine_model_inventory.csv", "README.zh.md"):
        shutil.copyfile(source / name, history / name)
    ledger_path = output / "compact/ledgers/extended_received292541.parquet"
    projector.checked(new_ledger, ledger_path)
    projector.checked(new_labels, output / "compact/label_dictionary.parquet")
    pd.DataFrame(changed_index).to_csv(output / "owned1023_score_update_index.csv", index=False)
    receipts["extended_received292541"].update(source_path=str(score_path), previous_source_path=str(previous_score),
        source_sha256=file_hash(score_path), previous_source_sha256=score_receipt["input_cohorts"][0]["sha256"],
        source_lines_one_based=True, bytes=ledger_path.stat().st_size, owned_member_update_count=1023,
        all_other_scientific_objects_unchanged=True, source_only_digest_or_identifier_fields=sorted(omitted),
        actual_update_receipt=pointer(paths["score-dir"] / "receipt.json"))
    receipts["dictionary_label"].update(rows=len(labels), bytes=(output / "compact/label_dictionary.parquet").stat().st_size,
        previous_projected_rows=len(label_rows), existing_ID_values_preserved=True)
    write_json(output / "compact/ledger_receipts.json", receipts)
    projection = load(output / "REVIEW_PROJECTION_RECEIPT.json")
    projection.update(source_package=str(source), current_owned_member_update_count=1023,
        frozen5_and_all_other_ledger_scientific_values_unchanged=True)
    projection["labels"].update(rows=len(labels), bytes=(output / "compact/label_dictionary.parquet").stat().st_size,
        prior_projected_rows=len(label_rows), added_projected_rows=len(labels) - len(label_rows),
        existing_label_IDs_unchanged=True)
    write_json(output / "REVIEW_PROJECTION_RECEIPT.json", projection)
    metrics, coverage = [], []
    conditions = dictionaries["condition"]
    for path in sorted((output / "compact/ledgers").glob("*.parquet")):
        frame = pq.read_table(path).to_pandas()
        frame["model"] = frame["condition_id"].map(lambda x: conditions[x]["model"])
        frame["dataset"] = frame["condition_id"].map(lambda x: conditions[x]["dataset"])
        for (model, dataset), group in frame.groupby(["model", "dataset"], sort=True):
            food = dataset == "food101"
            coverage.append({"ledger": path.stem, "model": model, "dataset": dataset, "rows": len(group),
                "canonical_pending": int(group["canonical"].isna().sum()) if food else None,
                "literal_pending": int(group["literal"].isna().sum()) if food else None,
                "abstain_pending": int(group["abstain"].isna().sum()),
                "food_reference_pending": int(group["reference_current"].isna().sum()) if food else None,
                "viz_official_answerable_pending": int(group["official_answerable"].isna().sum()) if not food else None,
                "viz_answer_quality_pending": int(group["answer_quality_credit"].isna().sum()) if not food else None})
        for identifier, group in frame.groupby("condition_id", sort=True):
            condition = conditions[identifier]
            metric = builder.metric_counts(group, condition)
            metrics.append({**metric, "ledger": path.stem, "condition_id": int(identifier), **condition})
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "compact/condition_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "scientific_decision_coverage.csv", index=False)
    extended = pd.read_csv(source / "extended_label_coverage.csv")
    for index, row in extended.iterrows():
        selected = metric_frame.loc[(metric_frame["ledger"] == row["ledger"]) & (metric_frame["model"] == row["model"])
                                    & (metric_frame["dataset"] == row["dataset"])]
        extended.loc[index, "pending_abstain"] = int(selected["pending_abstain"].sum())
        extended.loc[index, "pending_correct_or_quality"] = int(selected["pending_correct_or_quality"].sum())
        extended.loc[index, "pending_reference"] = int(selected["pending_reference"].sum())
        extended.loc[index, "full_conditions_with_complete_labels"] = int(selected["main_table_eligible"].sum())
    extended.to_csv(output / "extended_label_coverage.csv", index=False)
    evidence_index = load(output / "sources_evidence.json")
    evidence_root = output / "evidence/actual_luna500_root48"
    evidence_root.mkdir(exist_ok=False)
    binding = paths["binding-dir"]
    batch_base = binding.parent
    copy_sources = [(binding / "source_binding_validation_receipt.json", "source_binding_validation_receipt.json"),
        (paths["score-dir"] / "receipt.json", "applied_score_receipt.json"),
        (paths["score-dir"] / "application_audit.json", "application_audit.json"),
        (paths["score-dir"] / "unapplied_historical_shared_QA_conflicts.jsonl", "unapplied_historical_shared_QA_conflicts.jsonl"),
        (paths["root-source-dir"] / "root_decisions48.jsonl", "root_decisions48.jsonl"),
        (paths["root-source-dir"] / "root_receipt.json", "root_receipt.json"),
        (paths["prepare"] / "binding_index.csv", "binding_index.csv"),
        (paths["prepare"] / "source_memberships.jsonl", "source_memberships.jsonl"),
        (binding / "accepted_actual500_authority_view/receipt.json", "accepted500_authority_receipt.json")]
    for b in (0, 1):
        provider = batch_base / "actual_luna500_provider" / ("batch_" + str(b))
        for filename in ("model_final.jsonl", "provider_receipt.json", "actually_displayed_QA_payload.jsonl",
                         "read_tool_provider_event.json", "read_tool_full_output.txt", "model_final_provider_event.json"):
            copy_sources.append((provider / filename, "batch_" + str(b) + "/" + filename))
    for path, name in copy_sources:
        destination = evidence_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(file_hash(path) == file_hash(destination), "An actual source evidence copy changed")
        evidence_index.append({"original_path": str(path), "package_path": str(destination.relative_to(output)),
            "source_bytes": path.stat().st_size, "package_bytes": destination.stat().st_size,
            "source_sha256": file_hash(path), "source_scope": "actual finite500 QA and root48 accepted source evidence"})
    write_json(output / "sources_evidence.json", evidence_index)
    recompute_out = output.parent / (output.name + "_offline_recomputed")
    require(not recompute_out.exists(), "Offline recomputation output must also be exclusive")
    commands = [[sys.executable, str(output / "scripts/recompute_review_staging.py"),
                 "--package", str(output), "--out", str(recompute_out)]]
    # The existing coverage script requires an exclusive output filename.
    (output / "nine_model_coverage.csv").rename(history / "nine_model_coverage_copy.csv")
    commands.append([sys.executable, str(output / "scripts/nine_model_coverage.py"), "--package", str(output)])
    for index, command in enumerate(commands):
        log = output / ("offline_verification_" + str(index) + ".log")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=False)
        require(process.returncode == 0, "Existing offline package verification failed: " + str(log))
    verification_path = recompute_out / "verification.json"
    verification = load(verification_path)
    require(verification["saved_metric_values_recomputed_equal"] and verification["frozen_score_rows"] == 853248
            and verification["controls_reconstructed"] == 484800, "Actual offline frozen/condition recomputation failed")
    shutil.copyfile(verification_path, evidence_root / "offline_recomputed_verification.json")
    evidence_index.append({"original_path": str(verification_path),
        "package_path": str((evidence_root / "offline_recomputed_verification.json").relative_to(output)),
        "source_bytes": verification_path.stat().st_size, "package_bytes": verification_path.stat().st_size,
        "source_sha256": file_hash(verification_path), "source_scope": "actual existing offline recomputation proof"})
    write_json(output / "sources_evidence.json", evidence_index)
    frozen_files = list((source / "frozen5").rglob("*"))
    for path in frozen_files:
        if path.is_file():
            require(file_hash(path) == file_hash(output / path.relative_to(source)), "A frozen5 package file changed")
    for name in ("condition", "sample", "qa", "identity", "source"):
        require(tables[name].equals(pq.read_table(output / "compact" / (name + "_dictionary.parquet"))),
                "An existing dictionary mapping changed: " + name)
    for path in (source / "compact/ledgers").glob("*.parquet"):
        if path.name != ledger_path.name:
            require(file_hash(path) == file_hash(output / path.relative_to(source)), "Another source ledger changed")
    old_food = [i for i, row in enumerate(old_ledger["condition_id"].to_pylist()) if conditions[row]["dataset"] == "food101"]
    require(old_ledger.take(pa.array(old_food)).equals(new_ledger.take(pa.array(old_food))) and len(old_food) == 177333,
            "A compact Food value changed")
    pending_rows = extended.loc[(extended["ledger"] == "extended_received292541") & (extended["dataset"] == "vizwiz")]
    require(int(pending_rows["pending_abstain"].sum()) == int(pending_rows["pending_correct_or_quality"].sum()) == 49219,
            "Packaged remaining Viz decisions differ from the accepted delta")
    completion = [{"stage": "actual_Luna_semantic_judgment", "actual_QA": 500, "actual_generation_keys": 1023,
                   "status": "accepted", "source": "evidence/actual_luna500_root48/application_audit.json"},
                  {"stage": "root_full_QA_review", "actual_QA": 48, "status": "complete"},
                  {"stage": "root_ordinary_QA_sample", "actual_QA": 19, "status": "complete", "issues": 0},
                  {"stage": "received_Viz_remaining_semantic_QA", "actual_QA": 33812, "status": "pending"},
                  {"stage": "core_dev_selection_and_Viz512", "status": "root_to_append_actual_results"}]
    write_json(output / "ACTUAL_COMPLETION_V6.json", completion)
    appended = load(output / "APPENDED_SOURCES.json")
    appended.append({"source_path": str(score_path), "package_path": "compact/ledgers/extended_received292541.parquet",
        "source_line_scope": "same292541 rows; only1023 owned accepted members updated", "source_sha256": file_hash(score_path)})
    write_json(output / "APPENDED_SOURCES.json", appended)
    readme = output / "README.zh.md"
    text = readme.read_text(encoding="utf-8").replace("Viz115,208中50,242质量/弃权待决", "Viz115,208中49,219质量/弃权待决")
    text += "\n## v6 实际增量\n\n复制 v5 并接入已验收 Luna500/root48 的 1023 owned 生成键；旧 v5 保留。未决 QA 34312→33812，质量/弃权待决键各 50242→49219。原字典ID、全部 Food、其他已决 Viz 和五模型冻结资产保持。实际根复核为48疑难+19抽样，不称root读500。\n\n500原始模型判断、完整实际展示QA、provider读写事件、48root原始裁定与SHA在 evidence/actual_luna500_root48。owned1023_score_update_index.csv保存实际变更键和旧/新标签ID。共享QA的36个范围外旧行保持原样，其中1条“no”历史判断差异完整保存且未应用。没有新增生成、身份、语义裁定或GT；CURRENT_STATE和论文未修改。核心开发选择/Viz512后续由root追加。\n"
    readme.write_text(text, encoding="utf-8")
    result = {"schema": "kdm_review_package_owned_delta_v6", "passed": True, "source_package": str(source),
        "source_manifest_sha256": file_hash(source / "STAGING_MANIFEST.json"), "package": str(output),
        "received_rows_unchanged": 292541, "actual_updated_keys": 1023, "updated_per_model": dict(model_keys),
        "accepted_actual_QA": 500, "root_full_QA_review": 48, "ordinary_sample_review": 19,
        "old_label_dictionary_rows": len(label_rows), "new_label_dictionary_rows": len(labels),
        "added_label_dictionary_rows": len(labels) - len(label_rows), "all_old_dictionary_IDs_preserved": True,
        "Food_science_rows_unchanged": 177333, "other_Viz_science_rows_unchanged": 114185,
        "frozen5_science_and_files_unchanged": True, "pending_QA_before": 34312, "pending_QA_after": 33812,
        "pending_behavior_keys_before": 50242, "pending_behavior_keys_after": 49219,
        "pending_quality_keys_before": 50242, "pending_quality_keys_after": 49219,
        "unapplied_shared_QA_rows": 36, "unapplied_historical_semantic_conflicts": 1,
        "actual_offline_recomputation": verification, "actual_commands": commands,
        "source_score_receipt": pointer(paths["score-dir"] / "receipt.json"), "new_generations": 0,
        "new_semantic_judgments": 0, "new_GT_judgments": 0, "GPU_initialized": False, "CURRENT_STATE_modified": False,
        "elapsed_seconds": time.monotonic() - started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_hash(Path(__file__))}
    write_json(output / "OWNED1023_V6_ACCEPTANCE.json", result)
    manifest = load(output / "STAGING_MANIFEST.json")
    manifest.update(source_package=str(source), accepted_actual_QA_increment=500, accepted_owned_member_increment=1023,
                    current_source_score=str(score_path), current_source_receipt=pointer(paths["score-dir"] / "receipt.json"))
    files = sorted(p for p in output.rglob("*") if p.is_file() and p != output / "STAGING_MANIFEST.json")
    manifest["files"] = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size} for p in files]
    manifest["total_actual_bytes"] = sum(p.stat().st_size for p in files)
    write_json(output / "STAGING_MANIFEST.json", manifest)
    actual_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(json.dumps({**result, "actual_package_bytes_before_compression": actual_bytes,
                      "actual_package_file_count": sum(p.is_file() for p in output.rglob("*"))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
