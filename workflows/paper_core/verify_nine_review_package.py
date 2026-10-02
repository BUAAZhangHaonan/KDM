#!/usr/bin/env python3
"""Verify the exported review package without a checkout or server access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIVE = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
EXTRA = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
MODELS = FIVE + EXTRA
SID = {"qwen25vl", "llava16_mistral", "internvl35_8b", "onevision", "phi35", "qwen3vl"}
MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it")
FIELDS = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker",
          "guided", "reference_guided", "replicate")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def same_number(actual, expected, reason):
    if expected is None:
        require(pd.isna(actual), reason)
    else:
        require(np.isfinite(actual) and np.isclose(actual, expected, rtol=0, atol=1e-12), reason)


def expected_main():
    result = set()
    for model in MODELS:
        definitions = [("direct", "unguided", "NONE", False),
                       ("dola", "native_unguided", "NONE", False),
                       ("deco", "native_unguided", "NONE", False),
                       ("vcd", "native_unguided", "NONE", False),
                       ("cda_visual", "instruction_preserving", "UNKNOWN", True)]
        if model in SID:
            definitions.append(("sid", "native_unguided", "NONE", False))
        definitions += [(method, "instruction_preserving", marker, True)
                        for method in ("instruction_vcd", "instruction_m3id") for marker in MARKERS]
        result.update((model, "food101", "eval", method, kind, marker, marker, guided, False, 0)
                      for method, kind, marker, guided in definitions)
    return result


def verify_main(package):
    folder = package / "main"
    scores = pd.read_parquet(folder / "main_scores.parquet")
    conditions = pd.read_parquet(folder / "condition_dictionary.parquet")
    qa = pd.read_parquet(folder / "qa_dictionary.parquet")
    sources = pd.read_parquet(folder / "source_dictionary.parquet")
    require(len(scores) == 298152 and len(conditions) == 123, "Main panel size differs")
    require(not conditions.duplicated("compact_condition_id").any()
            and not qa.duplicated("qa_id").any() and not sources.duplicated("compact_source_id").any(),
            "A compact dictionary repeats identifiers")
    require(set(conditions[list(FIELDS)].itertuples(index=False, name=None)) == expected_main(),
            "The 123 mandatory native/IP condition identities differ")
    frame = scores.merge(conditions, on="compact_condition_id", validate="many_to_one")
    frame = frame.merge(qa, on="qa_id", validate="many_to_one")
    frame = frame.merge(sources, on="compact_source_id", validate="many_to_one")
    require(len(frame) == 298152 and not frame.duplicated(["condition_id", "sample_id"]).any(),
            "Main sample keys are missing or repeated")
    science = ["correct_canonical", "correct_literal", "abstain", "uniform_reference",
               "reference_complete", "accepted_complete_decision"]
    require(frame[science].notna().all().all() and frame.reference_complete.all()
            and frame.accepted_complete_decision.all(), "Main labels or reference connections are unresolved")
    for field in ("correct_canonical", "correct_literal", "abstain", "uniform_reference"):
        require(frame[field].isin([0, 1, False, True]).all(), "A Food decision is not binary: " + field)
    require(not (frame.correct_canonical.eq(1) & frame.abstain).any(), "Correct answers overlap abstention")
    require(frame.groupby(["condition_id", "target_class"]).size().eq(24).all()
            and frame.groupby("condition_id").target_class.nunique().eq(101).all(), "Food class quotas differ")
    domains = frame.groupby("condition_id").sample_id.agg(lambda values: frozenset(values))
    require(domains.nunique() == 1 and len(domains.iloc[0]) == 2424, "Methods use different eval inputs")
    require(frame.groupby(["model", "sample_id"]).uniform_reference.nunique().eq(1).all(),
            "A model/input reference changes across conditions")
    for row in qa.itertuples(index=False):
        key = hashlib.sha256(json.dumps([row.question, row.answer], ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
        require(key == row.qa_key, "A QA pointer differs from its full original text")
    require(frame.source_score_line.gt(0).all() and frame.source_score_sha256.str.len().eq(64).all(),
            "An original source path/hash/one-based line is missing")

    metrics = pd.read_csv(folder / "metrics_all_complete.csv").set_index("condition_id")
    require(len(metrics) == 123 and metrics.index.is_unique, "Main metrics repeat or omit conditions")
    for condition, data in frame.groupby("condition_id", sort=False):
        row = metrics.loc[condition]
        c = int(data.correct_canonical.eq(1).sum()); a = int(data.abstain.sum())
        r = int(data.uniform_reference.sum()); tp = int((data.abstain & data.uniform_reference).sum())
        counts = {"n": 2424, "C": c, "W": 2424-c-a, "A": a, "TP": tp, "FP": a-tp,
                  "FN": r-tp, "TN": 2424-r-a+tp, "reference_positive": r,
                  "J_numerator": c+tp, "J_denominator": 2424,
                  "literal_correct": int(data.correct_literal.eq(1).sum())}
        require(all(row[name] == value for name, value in counts.items()),
                "A metric numerator differs from complete sample decisions: " + condition)
        for name, numerator, denominator in (("J", c+tp, 2424), ("accuracy", c, 2424),
                                             ("precision", tp, a), ("recall", tp, r),
                                             ("F1", 2*tp, a+r), ("unnecessary_abstention_rate", a-tp, 2424-r)):
            same_number(row[name], numerator / denominator if denominator else None,
                        "A ratio or zero-denominator convention differs: " + name)

    observed = pd.read_csv(folder / "best_observed_joint_operating_points.csv")
    require(len(observed) == 18 and not observed.duplicated(["model", "method"]).any()
            and not observed.dev_selected.any() and observed.selection_split.eq("eval").all(),
            "The 18 eval observations are incomplete or mislabelled as dev-selected")
    for row in observed.itertuples():
        choices = metrics[(metrics.model == row.model) & (metrics.method == row.method)].copy()
        require(len(choices) == 4, "An IP observation lacks its four registered candidates")
        choices["order"] = choices.marker.map({v: i for i, v in enumerate(MARKERS)})
        winner = choices.sort_values(["J", "order"], ascending=[False, True]).iloc[0]
        require(winner.name == row.condition_id, "The observed J working point uses a different condition")
        for field in ("C", "W", "A", "TP", "FP", "FN", "J"):
            same_number(getattr(row, field), winner[field], "Working-point metrics mix different runs")

    paired = pd.read_csv(folder / "paired_main_comparisons.csv")
    joint = paired[paired.metric.eq("J")]
    require(len(joint) == 129 and joint.bootstrap_replicates.eq(2000).all()
            and joint.bootstrap_seed.eq(20260929).all(), "Registered J comparisons or bootstrap identity differ")
    for row in joint.itertuples():
        delta = metrics.loc[row.condition_id, "J"] - metrics.loc[row.baseline_condition_id, "J"]
        same_number(row.delta, delta, "A paired J difference uses a different working point")
        require(np.isfinite(row.ci95_lower) and np.isfinite(row.ci95_upper)
                and row.ci95_lower <= row.ci95_upper, "A saved paired interval is unavailable")
    return frame, metrics


def verify_references_and_selection(package, frame):
    folder = package / "support"
    core = pd.read_parquet(folder / "references/core5_references.parquet")
    extra = pd.read_parquet(folder / "references/reference4.parquet")
    require(len(core) == 24240 and len(extra) == 19392, "The frozen Food reference panels differ")
    reference = pd.concat([core[["model", "sample_id", "uniform_reference"]],
                           extra[["model", "sample_id", "reference_G"]].rename(columns={"reference_G": "uniform_reference"})])
    require(not reference.duplicated(["model", "sample_id"]).any(), "Frozen references repeat model/input keys")
    connected = frame[["model", "sample_id", "uniform_reference"]].drop_duplicates().merge(
        reference, on=["model", "sample_id"], how="left", suffixes=("", "_frozen"), validate="one_to_one")
    require(len(connected) == 21816 and connected.uniform_reference_frozen.notna().all()
            and connected.uniform_reference.eq(connected.uniform_reference_frozen).all(),
            "Main reference labels differ from their frozen authorities")
    attempts = pd.read_parquet(folder / "references/reference4_attempts.parquet")
    require(len(attempts) == 193920 and not attempts.duplicated(["model", "sample_id", "replicate"]).any(),
            "Independent attempts lose or repeat keys")
    counts = attempts.groupby(["model", "sample_id"]).canonical.agg(["size", "sum"])
    connected = extra.set_index(["model", "sample_id"]).join(counts, validate="one_to_one")
    require(connected["size"].eq(10).all() and connected["sum"].eq(connected.correct_count).all(),
            "Independent attempts do not reproduce the reference counts")
    require(extra.reference_G.eq(extra.gold_rank.gt(1) & extra.correct_count.eq(0)).all(),
            "The registered Food reference definition differs")

    dev = pd.read_parquet(folder / "dev_selection/registered_dev_decisions.parquet")
    selected = load(folder / "dev_selection/IP_VCD_selected_configs5.json")
    evaluation = pd.read_csv(package / "main/dev_selected_operating_points.csv")
    require(len(dev) == 32320 and len(selected) == len(evaluation) == 5 and dev.split.eq("dev").all(),
            "Actual dev coverage or selected models differ")
    require(not dev.duplicated(["model", "method", "marker", "sample_id"]).any(), "Dev keys repeat")
    for choice in selected:
        require(choice["selection"] == "dev_selected" and choice["selection_split"] == "dev"
                and choice["method"] == "instruction_vcd" and choice["model"] in FIVE,
                "A selection does not come from the actual core dev panel")
        options = []
        for marker in MARKERS:
            data = dev[(dev.model == choice["model"]) & (dev.method == "instruction_vcd") & (dev.marker == marker)]
            require(len(data) == 404 and data.sample_id.nunique() == 404, "A dev candidate is incomplete")
            c = int(data.correct_canonical.sum()); tp = int((data.abstain & data.uniform_reference).sum())
            options.append((-(c+tp), -c, MARKERS.index(marker), marker, c, tp))
        winner = min(options)
        require(choice["marker"] == winner[3] and choice["correct"] == winner[4]
                and choice["tp"] == winner[5], "Dev selection or its frozen counts differ")
        point = evaluation[evaluation.model.eq(choice["model"])]
        require(len(point) == 1 and point.iloc[0].marker == choice["marker"]
                and point.iloc[0].selection == "dev_selected", "Eval uses a different actual dev choice")


def verify_support_panels(package):
    mechanism = package / "mechanism"
    data = pd.read_parquet(mechanism / "mechanism_scores.parquet")
    conditions = pd.read_csv(mechanism / "conditions.csv")
    require(len(data) == 484800 and len(conditions) == 200
            and not data.duplicated(["condition_id", "sample_id"]).any(), "Frozen mechanism coverage differs")
    expected = {(model, method, kind): n for model in FIVE for method in ("vcd", "m3id")
                for kind, n in (("main", 16), ("reference_instruction_removed", 4))}
    require(conditions.groupby(["model", "method", "kind"]).size().to_dict() == expected,
            "Frozen mechanism includes different or exploratory conditions")
    require(data.groupby(["condition_id", "target_class"]).size().eq(24).all(), "Mechanism quotas differ")
    qa = pd.read_parquet(mechanism / "qa_dictionary.parquet")
    require(qa[["qa_key", "question", "answer"]].notna().all().all()
            and set(data.qa_id).issubset(set(qa.qa_id)), "A frozen mechanism QA still lacks full text")
    for row in qa.itertuples(index=False):
        key = hashlib.sha256(json.dumps([row.question, row.answer], ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
        require(key == row.qa_key, "A recovered mechanism QA differs from its frozen hash")
    viz = pd.read_parquet(package / "support/vizwiz/core5_512/new_scores.parquet")
    require(len(viz) == 15872 and viz.condition_id.nunique() == 31
            and not viz.duplicated(["condition_id", "sample_id"]).any(), "Actual VizWiz coverage differs")
    require(viz[["quality_score", "official_raw_score", "abstain", "official_reference"]].notna().all().all()
            and viz.quality_score.between(0, 1).all() and viz.official_raw_score.between(0, 1).all(),
            "Official continuous VizWiz quality is unresolved or outside its range")
    require(viz.groupby("condition_id").size().eq(512).all()
            and viz.groupby("condition_id").official_reference.sum().eq(166).all()
            and viz.official_reference.eq(viz.annotated_answerable.eq(0)).all(), "VizWiz answerability quotas differ")
    require(viz.loc[viz.abstain, "quality_score"].eq(0).all()
            and viz.loc[viz.quality_state.eq("PARTIAL"), "quality_score"].between(0, 1, inclusive="neither").all(),
            "VizWiz abstention or partial-credit semantics differ")
    require(set(viz.quality_state) == {"FULL", "PARTIAL", "ZERO", "A"}, "VizWiz states were collapsed")
    return len(data), len(viz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = load(package / "PACKAGE_MANIFEST.json")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    require(actual == set(manifest) | {"PACKAGE_MANIFEST.json"}, "Package members differ from the manifest")
    for relative, proof in manifest.items():
        path = (package / relative).resolve()
        path.relative_to(package)
        require(path.stat().st_size == proof["bytes"] and sha(path) == proof["sha256"],
                "A package file differs from its source hash: " + relative)
    proof = load(package / "PACKAGE_RECEIPT.json")
    require(proof["passed"] and proof["Food_main_complete"]
            and not proof["extension_mechanism_complete"] and not proof["all_nine_VizWiz_methods_complete"],
            "The package misstates completion scope")
    frame, _ = verify_main(package)
    verify_references_and_selection(package, frame)
    mechanism_rows, viz_rows = verify_support_panels(package)
    print(json.dumps({"passed": True, "offline_no_server_access": True, "files": len(manifest),
                      "Food_main_rows": len(frame), "Food_main_conditions": 123,
                      "frozen_mechanism_rows": mechanism_rows, "core_VizWiz_rows": viz_rows,
                      "actual_dev_selected_IP_VCD_models": 5, "new_scoring_or_annotation": 0}))


if __name__ == "__main__":
    main()
