#!/usr/bin/env python3
"""Save a bounded final-review preparation checkpoint from accepted packages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
CURRENT_ARCHIVE = BASE / "final_review_package_complete_accepted_sources_v2_20261003/KDM_Nine_Complete_Review.zip"
CURRENT_ARCHIVE_SHA = "f12f2c90e00e0f4134a9e8e4d7842c813d3001dcbbc8d8d1e3edaee1a39c1d48"
MEMBER_MAP = ROOT / "configs/kdm/review_archive_members_20261004.json"
ARCHIVES = (
    (
        "Food_v5",
        CURRENT_ARCHIVE,
        CURRENT_ARCHIVE_SHA,
        ("README.zh.md", "ACTUAL_TASK_STATUS.json", "PACKAGE_RECEIPT.json", "PACKAGE_MANIFEST.json"),
    ),
    (
        "latest_dev_J",
        CURRENT_ARCHIVE,
        CURRENT_ARCHIVE_SHA,
        ("README.md", "manifest.json", "dev_selection/receipt.json", "Food_eval_J/receipt.json"),
    ),
)
DETAILS = BASE / "native_baselines/REMAINING4_BOUNDED_DETAILS_COMPLETED_ONLY_ACTUAL_20261003_231718.json"
DETAILS_SHA = "ded4b82196fd6a40fcf6b39be58859287c9ed3ac8f4339f1d75b8c7d9c84a5bf"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def logical_archive_members(role: str, path: Path, expected: str):
    """Read byte-identical historical input members from the retained final archive."""
    if digest(path) != expected:
        raise ValueError(f"Source archive SHA differs: {path}")
    mapping = json.loads(MEMBER_MAP.read_text(encoding="utf-8"))
    if mapping["replacement_sha256"] != expected:
        raise ValueError("Archive migration targets a different accepted package")
    rows = mapping["roles"][role]["members"]
    if len({row["original_member"] for row in rows}) != len(rows):
        raise ValueError("Repeated logical source members")
    with zipfile.ZipFile(path) as archive:
        for row in rows:
            name = row["original_member"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"Unsafe logical archive path: {name}")
            data = archive.read(row["replacement_member"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Migrated archive member differs: {name}")
            yield row, data


def archive_checkpoint(role: str, path: Path, expected: str, metadata: tuple[str, ...], output: Path) -> dict:
    members = list(logical_archive_members(role, path, expected))
    copied = []
    for row, data in members:
        name = row["original_member"]
        if name in metadata:
            destination = output / "source_metadata" / role / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            copied.append({"member": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    if {row["member"] for row in copied} != set(metadata):
        raise ValueError(f"Missing requested source metadata: {role}")
    return {
        "role": role, "path": str(path), "sha256": expected,
        "zip_bytes": path.stat().st_size, "entries": len(members), "CRC_passed": True,
        "uncompressed_bytes": sum(len(data) for _, data in members),
        "metadata_copied_byte_identically": copied,
        "member_index": [{"path": row["original_member"], "bytes": len(data),
                          "replacement_member": row["replacement_member"]} for row, data in members],
        "migration_index": str(MEMBER_MAP),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.relative_to(BASE.resolve())
    output.mkdir(parents=True, exist_ok=False)
    archives = [archive_checkpoint(*definition, output) for definition in ARCHIVES]
    if digest(DETAILS) != DETAILS_SHA:
        raise ValueError("The completed-only bounded-details source changed")
    detail_data = DETAILS.read_bytes()
    details = json.loads(detail_data)
    destination = output / "source_metadata/bounded_details_remaining4.json"
    destination.write_bytes(detail_data)
    receipt = {
        "schema": "kdm_nine_final_review_preparation_v1",
        "preparation_passed": True,
        "final_package_complete": False,
        "final_ZIP_created": False,
        "input_archives": archives,
        "bounded_details_source": {"path": str(DETAILS), "sha256": DETAILS_SHA, "bytes": len(detail_data)},
        "source_archive_bytes_before_any_deduplication": sum(row["zip_bytes"] for row in archives),
        "target_final_zip_bytes": 25000000,
        "source_score_or_dictionary_fields_changed": 0,
        "pending_inputs": [{
            "role": "Viz_final_actual_closed",
            "status": "awaiting exact accepted path and SHA from root",
            "expected_scope_supplied_by_root": "31 prior + 25 new conditions, 28672 actual decisions",
            "required_evidence": ["accepted receipt", "complete condition identities", "zero unresolved main labels",
                                  "actual annotated answer_text", "official raw quality", "source provenance"],
        }],
        "packaging_rules": {
            "Food_v5_README_and_actual_status_destination": "prior_snapshot/Food_v5",
            "scientific_result_fields_and_dictionary_IDs": "preserve all existing values",
            "repeated_images_and_data": "deduplicate only with equal actual SHA and reversible source mapping",
            "final_main_table": "native baselines and actual dev-selected IP conditions",
            "old_four_expression_best_observed": "separate observation table",
            "matrix_interpretation": "mechanism only",
            "extra_remaining4_details": "116 selected, 103 divergent diagnostics, 13 no-divergence, 11 cases and 22 paths",
            "core5_original_detail_scope": "preserve existing 814 positions, 1049 pairs, 24 paths",
            "size_limit": "report actual excess while preserving scientific payload",
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": digest(Path(__file__)),
        "GPU_initialized": False,
        "new_scoring_or_generation": False,
        "global_state_or_git_modified": False,
    }
    save_json(output / "PREPARATION_CHECKPOINT.json", receipt)
    save_json(output / "bounded_details_original_top_fields.json", {"original_top_keys": sorted(details)})
    (output / "README.zh.md").write_text(
        "# 九模型最终审阅包准备检查点\n\n"
        "Food v5、最新 dev/J ZIP 已核验给定 SHA 与全部成员 CRC；原内容未修改。"
        "有限四模型 details manifest 按原 SHA 字节复制。\n\n"
        "Viz 最终闭合包的实际路径和 SHA 尚待 root 提供；当前没有生成最终 ZIP，"
        "没有声明全任务完成。两份 ZIP 原尺寸和成员清单见 PREPARATION_CHECKPOINT.json。"
        "最后整合将保存旧说明历史快照、当前 dev-selected 主表、Viz 实际覆盖与精确科学缺项。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "preparation_passed": True, "final_package_complete": False,
                      "input_archive_bytes": receipt["source_archive_bytes_before_any_deduplication"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
