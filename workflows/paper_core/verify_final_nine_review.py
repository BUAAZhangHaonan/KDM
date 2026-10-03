#!/usr/bin/env python3
"""Verify a final review ZIP extraction, optionally restoring equal-SHA assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

import pandas as pd
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe(package: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "Unsafe manifest path")
    path = (package / name).resolve()
    path.relative_to(package.resolve())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = load(package / "FINAL_PACKAGE_MANIFEST.json")
    aliases = load(package / "IDENTICAL_ASSET_ALIASES.json")
    require(digest(package / "IDENTICAL_ASSET_ALIASES.json") == manifest["aliases_sha256"], "Alias map changed")
    restored = 0
    for name, record in manifest["files"].items():
        target = safe(package, name)
        if not target.exists() and name in aliases:
            binding = aliases[name]
            canonical = safe(package, binding["canonical_path"])
            require(canonical.is_file() and digest(canonical) == binding["sha256"] == record["sha256"],
                    "An explicit identical asset cannot be restored")
            if args.materialize:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(canonical, target)
                restored += 1
            else:
                target = canonical
        require(target.is_file() and target.stat().st_size == record["bytes"] and digest(target) == record["sha256"],
                f"A packaged source file differs: {name}")
    status = load(package / "FINAL_ACTUAL_STATUS.json")
    require(status["passed"] is True and status["Food_rows"] == 1170792
            and status["Food_main_conditions"] == 123 and status["dev"]["selected_model_methods"] == 18,
            "The actual Food/dev scope differs")
    metrics = pd.read_csv(package / "main/metrics_all_complete.csv")
    points = pd.read_csv(package / "dev_and_J/Food_eval_J/main69_actual_operating_points.csv")
    require(len(metrics) == 123 and len(points) == 69, "Main points or full conditions differ")
    selected = metrics.set_index("condition_id")
    for row in points.itertuples(index=False):
        source = selected.loc[row.condition_id]
        require(all(getattr(row, field) == source[field] for field in ("C", "W", "A", "TP", "FP", "FN", "J")),
                "A current point mixes different accepted conditions")
    old31 = pq.read_metadata(package / "support/vizwiz/core5_512/new_scores.parquet").num_rows
    new = pq.read_table(package / "vizwiz_final/new25_complete_records.parquet").to_pylist()
    require(old31 == 15872 and len(new) == 12800, "Viz scientific rows are missing")
    for row in new:
        original = row["complete_original_record_json_line"]
        record = json.loads(original)
        require(hashlib.sha256(original.encode()).hexdigest() == row["source_record_line_sha256"]
                and record["key"] == row["key"] and record["model"] == row["model"]
                and record["sample_id"] == row["sample_id"], "A complete original Viz record changed")
    coverage = pd.read_csv(package / "vizwiz_final/condition_coverage56.csv")
    require(len(coverage) == 56 and coverage.n.eq(512).all()
            and coverage.quality_pending.eq(0).all() and coverage.abstain_pending.eq(0).all()
            and coverage.panel_role.eq("main").sum() == 45, "Actual Viz coverage differs")
    details = load(package / "bounded_details_remaining4/completed_only_manifest.json")
    math = load(package / "bounded_details_remaining4/math_summary.json")
    require(details["passed"] is True and details["source_scopes"] == 8 and math["passed"] is True
            and math["diagnostic_samples"] == 103 and math["actual_token_candidate_pairs"] == 175
            and math["actual_registered_case_inputs"] == 11, "Bounded details scope differs")
    result = {"passed": True, "logical_files_verified": len(manifest["files"]),
              "restored_identical_asset_paths": restored, "Food_full_rows": 1170792,
              "dev_selected_model_methods": 18, "main_actual_workpoints": 69,
              "Viz_old_rows": old31, "Viz_new_rows": len(new), "Viz_main_conditions": 45,
              "Viz_mechanism_conditions": 11, "new25_complete_original_JSON_rows_equal": True,
              "new_scoring_or_GPU": False}
    if args.output:
        require(not args.output.exists(), "Verification output must be exclusive")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
