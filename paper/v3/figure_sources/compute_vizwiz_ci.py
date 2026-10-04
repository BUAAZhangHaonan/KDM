"""Compute sample-paired VizWiz J intervals from the saved score rows."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True,
                    help='Directory containing merged31_new_scores.csv and finalrows/*.jsonl.gz')
TMP = parser.parse_args().source_dir
OUT = ROOT / "statistics/vizwiz_paired_J_intervals.csv"
RECEIPT = ROOT / "figure_sources/vizwiz_ci_receipt.json"
BOOT = 2000
SEED = 20260929


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows() -> pd.DataFrame:
    core = pd.read_csv(TMP / "merged31_new_scores.csv")
    core["source_set"] = "final_review_package_support_vizwiz_core5"
    core["condition_id"] = core["condition_id"].fillna("")
    core["source_file"] = "merged_viz31_all_five_closed_20261003_0410/new_scores.csv"
    fields = ["model", "method", "marker", "sample_id", "quality_score", "abstain",
              "official_reference", "condition_id", "source_file", "source_set"]
    rows: list[dict] = []
    for path in sorted((TMP / "finalrows").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                rows.append({k: x.get(k) for k in fields[:-2]} | {
                    "source_file": path.name,
                    "source_set": "final_review_package_vizwiz_final",
                })
    final = pd.DataFrame(rows, columns=fields)
    return pd.concat([core[fields], final[fields]], ignore_index=True)


def utility(frame: pd.DataFrame) -> pd.Series:
    return frame["quality_score"].astype(float) + (
        frame["abstain"].fillna(False).astype(bool)
        & frame["official_reference"].fillna(False).astype(bool)
    ).astype(int)


def main() -> None:
    rows = load_rows()
    points = pd.read_csv(ROOT / "statistics/vizwiz_nine_models_recorded_working_points.csv")
    points = points[points.method.isin(["vcd", "instruction_vcd", "instruction_m3id"])]
    models = list(points.model.drop_duplicates())
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for _, point in points.iterrows():
        q = rows[(rows.model == point.model) & (rows.method == point.method)]
        q = q[q.marker == point.marker]
        if len(q) != 512 or q.sample_id.nunique() != 512:
            raise ValueError(f"missing or duplicated rows for {point.model}/{point.method}/{point.marker}")
        q = q.copy()
        q["u"] = utility(q)
        computed = float(q.u.mean())
        if not np.isclose(computed, float(point.J), rtol=0, atol=1e-12):
            raise ValueError(f"aggregate J mismatch for {point.model}/{point.method}: {computed} != {point.J}")
        selected[(point.model, point.method)] = q

    out: list[dict] = []
    rng = np.random.default_rng(SEED)
    for model in models:
        base = selected[(model, "vcd")].set_index("sample_id")
        for method in ("instruction_vcd", "instruction_m3id"):
            test = selected[(model, method)].set_index("sample_id")
            ids = base.index.intersection(test.index)
            if len(ids) != 512 or len(set(ids)) != 512:
                raise ValueError(f"unpaired inputs for {model}/{method}")
            delta = (test.loc[ids, "u"].to_numpy(float) - base.loc[ids, "u"].to_numpy(float))
            draws = rng.integers(0, len(delta), size=(BOOT, len(delta)))
            boot_delta_pp = delta[draws].mean(axis=1) * 100.0
            point_pp = float(delta.mean() * 100.0)
            source_sets = sorted(set(test.loc[ids, "source_set"]) | set(base.loc[ids, "source_set"]))
            source_files = sorted(set(test.loc[ids, "source_file"]) | set(base.loc[ids, "source_file"]))
            out.append({
                "task": "VizWiz",
                "model": model,
                "method": method,
                "baseline": "vcd",
                "n": len(delta),
                "delta_J_pp": point_pp,
                "ci95_lo_pp": float(np.percentile(boot_delta_pp, 2.5)),
                "ci95_hi_pp": float(np.percentile(boot_delta_pp, 97.5)),
                "bootstrap_replicates": BOOT,
                "bootstrap_seed": SEED,
                "bootstrap_unit": "VizWiz eval image/sample_id",
                "J_formula": "mean(quality_score + abstain*official_reference)",
                "source_sets": ";".join(source_sets),
                "source_files": ";".join(source_files),
                "status": "available",
            })
    result = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, float_format="%.12g")
    receipt = {
        "schema": "kdm_vizwiz_paired_J_intervals_v1",
        "task": "VizWiz",
        "rows": len(result),
        "available_rows": int((result.status == "available").sum()),
        "models": models,
        "n_per_pair": 512,
        "bootstrap_replicates": BOOT,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "VizWiz eval image/sample_id",
        "J_formula": "mean(quality_score + abstain*official_reference)",
        "aggregate_verified_against": "statistics/vizwiz_nine_models_recorded_working_points.csv",
        "aggregate_input_sha256": sha256(ROOT / "statistics/vizwiz_nine_models_recorded_working_points.csv"),
        "copied_source_sha256": {str(p.relative_to(TMP)): sha256(p) for p in sorted(TMP.rglob("*")) if p.is_file()},
        "remote_source_packages": [
            "outputs/paper_core_20261002_dev_viz/merged_viz31_all_five_closed_20261003_0410",
            "outputs/paper_core_20261002_dev_viz/final_review_package_complete_accepted_sources_v2_20261003/package/vizwiz_final",
        ],
        "new_inference": 0,
        "new_scientific_judgment": 0,
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
