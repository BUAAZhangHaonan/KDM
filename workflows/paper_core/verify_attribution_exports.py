"""Verify the finite four-view export without new model forwards or old raw scans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import stable_hash, stable_seed
from kdm.prompts import task_prompt

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
DIAGNOSTIC_N = dict(zip(MODELS, (48, 48, 43, 34, 25)))
CASE_N = dict(zip(MODELS, (1, 1, 5, 5, 0)))
PAIR_LABELS = {
    "guided_vcd_ip": ("vcd_unknown", "instruction_vcd_unknown"),
    "direct_guided_vcd": ("direct_guided", "vcd_unknown"),
    "direct_unguided_guided": ("direct_unguided", "direct_guided"),
}
RESIDUAL = Counter()
SOURCES = []
EVENTS = []
PAIRS = []
CASES = []
ARGMAX_TIES = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_float(a, b, name):
    a, b = float(a), float(b)
    assert math.isfinite(a) and math.isfinite(b), name
    delta = abs(a - b)
    RESIDUAL[name] = max(RESIDUAL[name], delta)
    assert delta <= max(1e-8, 1e-10 * max(abs(a), abs(b))), (name, a, b)


def ledger(path, expected):
    if not path.exists():
        assert expected == 0, path
        return []
    identity_path = path.with_suffix(".identity.json")
    identity = json.loads(identity_path.read_text())
    assert stable_hash(identity["definition"]) == identity["identity"], path
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == expected, (path, len(rows), expected)
    assert len({r["key"] for r in rows}) == len(rows), path
    assert all(r["identity"] == identity["identity"] and r["status"] == "ok" for r in rows)
    SOURCES.append({"path": str(path), "sha256": sha(path),
                    "identity_path": str(identity_path), "identity_sha256": sha(identity_path),
                    "identity": identity["identity"], "rows": len(rows)})
    return rows


def measure(m, model, sample, panel, source_path, source_line, stratum="", path_name=""):
    assert m["source_meta"]["model"] == model
    assert m["seed"] == stable_seed(sample, model, 0), (model, sample, "seed")
    assert (m["alpha"], m["beta"], m["noise_step"], m["marker"]) == (1.0, .1, 500, "UNKNOWN")
    plain = task_prompt(m["question"], guided=False)
    guided = task_prompt(m["question"], marker="UNKNOWN", guided=True)
    assert m["prompt_texts"] == {"c": plain, "r": plain, "g": guided, "h": guided}
    proof = m["actual_input_proof"]
    assert all(proof[k] for k in ("same_clean_visual_tensor", "same_noisy_visual_tensor",
                                 "registered_noise_recreated_exactly"))
    assert proof["prompt_token_ids"]["c"] == proof["prompt_token_ids"]["r"]
    assert proof["prompt_token_ids"]["g"] == proof["prompt_token_ids"]["h"]
    assert len(m["events"]) == len(m["prefixes"])
    for event, prefix in zip(m["events"], m["prefixes"]):
        assert event["prefix_tokens"] == prefix and event["position"] == len(prefix)
        if m["observed_tokens"] is not None:
            assert prefix == m["observed_tokens"][:len(prefix)]
            assert event["observed_token"] == m["observed_tokens"][len(prefix)]
        candidates = {r["token"]: r for r in event["candidate_rows"]}
        assert len(candidates) == len(event["candidate_rows"])
        assert set(candidates) == set(event["candidate_tokens"])
        for token, row in candidates.items():
            assert row["common_support"] == (row["Sc_member"] and row["Sg_member"])
            lp = row["logp"]
            assert set(lp) == {"g", "h", "c", "r"} and all(math.isfinite(x) for x in lp.values())
            for view in lp:
                equal_float(math.exp(lp[view]), row["probability"][view], "raw_probability")
            interaction = (lp["g"] - lp["c"]) - (lp["h"] - lp["r"])
            scores = {"native": 2*lp["c"] - lp["r"],
                      "guided": 2*lp["g"] - lp["h"],
                      "ip": lp["g"] + lp["c"] - lp["r"]}
            masks = {"native": row["Sc_member"], "guided": row["Sg_member"], "ip": row["Sg_member"]}
            scores["native_on_guided_support"] = scores["native"]
            masks["native_on_guided_support"] = row["Sg_member"]
            for name in ("native", "guided", "ip"):
                scores[name+"_on_common_support"] = scores[name]
                masks[name+"_on_common_support"] = row["common_support"]
            for weight in (0, .5, 1):
                name = f"interaction_{weight:g}"
                scores[name] = scores["ip"] + weight*interaction
                masks[name] = row["Sg_member"]
            for name, score in scores.items():
                observed = row["method_log_probability"][name]
                probability = row["method_probability"][name]
                if not masks[name]:
                    assert observed is None and probability in (None, 0), (name, token)
                else:
                    norm = event["masked_method_distributions"][name]["log_normalizer"]
                    equal_float(score-norm, observed, "sparse_local_normalization")
                    equal_float(math.exp(score-norm), probability, "sparse_method_probability")
            equal_float(scores["guided"]-scores["ip"], interaction, "candidate_interaction_closure")
        dist = event["masked_method_distributions"]
        for left_name, right_name in (("ip", "interaction_0"), ("guided", "interaction_1")):
            left, right = dist[left_name]["argmax"], dist[right_name]["argmax"]
            if left != right:
                # Algebraically identical formulas can select different tokens
                # at a float64 tie; preserve both actual exported argmaxes.
                gaps = []
                for method_name in (left_name, right_name):
                    left_value = candidates[left]["method_log_probability"][method_name]
                    right_value = candidates[right]["method_log_probability"][method_name]
                    equal_float(left_value, right_value, "endpoint_argmax_tie_gap")
                    gaps.append(abs(left_value-right_value))
                ARGMAX_TIES.append({"model": model, "sample_id": sample, "panel": panel,
                                    "position": event["position"], "left_method": left_name,
                                    "right_method": right_name, "left_token": left,
                                    "right_token": right, "maximum_gap": max(gaps)})
        equal_float(dist["ip"]["log_normalizer"], dist["interaction_0"]["log_normalizer"],
                    "lambda0_endpoint")
        equal_float(dist["guided"]["log_normalizer"], dist["interaction_1"]["log_normalizer"],
                    "lambda1_endpoint")
        record = {"model": model, "sample_id": sample, "panel": panel, "stratum": stratum,
                  "path_condition": path_name, "position": event["position"],
                  "prefix_tokens": prefix, "seed": m["seed"], "source_path": str(source_path),
                  "source_line": source_line, "support": event["support"], "argmax": event["argmax"],
                  "distributions": dist, "candidate_pairs": event["math_pairs"],
                  "semantic_abstention_inferred": False}
        EVENTS.append(record)
        for pair in event["math_pairs"]:
            a, b = pair["candidate_a"], pair["candidate_b"]
            assert a != b and a in candidates and b in candidates
            margins = {v: candidates[a]["logp"][v] - candidates[b]["logp"][v] for v in ("g", "h", "c", "r")}
            visual = margins["c"] - margins["r"]
            normal_instruction = margins["g"] - margins["c"]
            reference_instruction = margins["h"] - margins["r"]
            interaction = normal_instruction - reference_instruction
            guided_margin = 2*margins["g"] - margins["h"]
            ip_margin = margins["g"] + visual
            first_residual = guided_margin - (margins["c"] + visual + normal_instruction + interaction)
            second_residual = (ip_margin-guided_margin) + interaction
            equal_float(first_residual, 0, "decomposition_closure")
            equal_float(second_residual, 0, "ip_difference_closure")
            equal_float(pair["guided_pair_margin"], guided_margin, "exported_guided_margin")
            equal_float(pair["ip_pair_margin"], ip_margin, "exported_ip_margin")
            equal_float(pair["interaction_pair_margin"], interaction, "exported_interaction")
            equal_float(pair["decomposition_closure"], first_residual, "saved_decomposition_residual")
            equal_float(pair["ip_difference_closure"], second_residual, "saved_ip_residual")
            for item in pair["interpolation"]:
                equal_float(item["pair_margin"], ip_margin+item["interaction_weight"]*interaction,
                            "pair_interpolation")
            pair_support = candidates[a]["Sg_member"] and candidates[b]["Sg_member"]
            assert pair_support == pair["pair_in_guided_support"]
            assert (candidates[a]["Sc_member"] and candidates[b]["Sc_member"]) == pair["pair_in_native_support"]
            marker_ids = {ids[0] for ids in event["token_proxy_marker_ids"].values() if ids}
            PAIRS.append({"model": model, "sample_id": sample, "panel": panel, "stratum": stratum,
                          "position": event["position"], "candidate_a": a, "candidate_b": b,
                          "candidate_a_text": candidates[a]["decoded"], "candidate_b_text": candidates[b]["decoded"],
                          "a_nonmarker_b_marker_proxy": a not in marker_ids and b in marker_ids,
                          "natural_visual": visual, "normal_instruction": normal_instruction,
                          "reference_instruction": reference_instruction, "interaction": interaction,
                          "ip_minus_guided": ip_margin-guided_margin,
                          "pair_in_guided_support": pair_support,
                          "pair_in_common_support": pair_support and candidates[a]["Sc_member"] and candidates[b]["Sc_member"],
                          "source_path": str(source_path), "source_line": source_line})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--representative", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    coverage = []
    budget = 0
    for model in MODELS:
        host = "K100" if model == "qwen35_4b" else "6403"
        path = args.representative / host / f"{model}.representative_events.jsonl"
        rows = ledger(path, 101)
        assert len({r["sample_id"] for r in rows}) == len({r["target_class"] for r in rows}) == 101
        for line, row in enumerate(rows, 1):
            assert row["model"] == model and row["split"] == "eval"
            assert row["measured"]["prefixes"] == [[]]
            measure(row["measured"], model, row["sample_id"], "representative101", path, line)
        detail = args.details / model
        path = detail / "diagnostic.events.jsonl"
        diagnostics = ledger(path, DIAGNOSTIC_N[model])
        no_divergence = 0
        for line, row in enumerate(diagnostics, 1):
            config = row["config_row"]
            assert config["model"] == model
            if row.get("measurement_status") == "no_divergence":
                sequences = list(row["source_tokens"].values())
                shortest = min(map(len, sequences))
                assert all(x[:shortest] == sequences[0][:shortest] for x in sequences)
                assert row["lcp"] == shortest and row["positions"] == []
                no_divergence += 1
                continue
            assert 1 <= len(row["measured_pairs"]) <= 2
            for m in row["measured_pairs"]:
                meta = m["source_meta"]
                left_name, right_name = PAIR_LABELS[meta["pair_label"]]
                left = meta["source_branches"][left_name]["tokens"]
                right = meta["source_branches"][right_name]["tokens"]
                position = meta["position"]
                assert left[:position] == right[:position] and left[position] != right[position]
                assert m["prefixes"] == [left[:position]]
                assert m["candidate_pairs"] == [[left[position], right[position]]]
                measure(m, model, config["sample_id"], "diagnostic198", path, line, config["stratum"])
        path = detail / "case12.events.jsonl"
        cases = ledger(path, CASE_N[model])
        for line, row in enumerate(cases, 1):
            assert len(row["paths"]) == 2
            for m in row["paths"]:
                source = m["source_meta"]["source"]
                assert m["response_tokens"] == m["observed_tokens"] == source["tokens"]
                assert m["response_text"] == m["observed_text"] == source["text"]
                assert m["prefixes"] == [source["tokens"][:i] for i in range(len(source["tokens"]))]
                assert m["terminated"] == source["terminated"]
                assert m["truncated"] == (not m["terminated"] and len(source["tokens"]) >= 32)
                measure(m, model, row["case"]["sample_id"], "case12", path, line,
                        path_name=source["condition"])
                CASES.append({"model": model, "sample_id": row["case"]["sample_id"],
                              "condition": source["condition"], "tokens": json.dumps(source["tokens"]),
                              "text": source["text"], "terminated": m["terminated"], "truncated": m["truncated"],
                              "positions": len(m["events"]), "source_key": source["source_key"],
                              "source_path": str(path), "source_line": line})
        natural = ledger(detail / "natural_reference.events.jsonl", 101)
        assert {r["sample_id"] for r in natural} == {r["sample_id"] for r in rows}
        for row in natural:
            assert row["seed"] == stable_seed(row["sample_id"], model, 0)
            assert row["noise_generated_r"]["status"] == "ok"
        used = json.loads((detail / "budget.json").read_text())["used_gpu_seconds"]
        assert 0 <= used <= 2880
        budget += used
        coverage.append({"model": model, "representative": len(rows), "diagnostic_rows": len(diagnostics),
                         "diagnostic_no_divergence": no_divergence, "cases": len(cases),
                         "natural_reference": len(natural), "cumulative_gpu_seconds": used})
    assert sum(r["representative"] for r in coverage) == 505
    assert sum(r["diagnostic_rows"] for r in coverage) == 198
    assert len(CASES) == 24 and sum(r["cases"] for r in coverage) == 12
    assert budget <= 14400
    for name, rows in (("four_view_pair_margins.csv", PAIRS), ("coverage.csv", coverage),
                       ("case_replay_summary.csv", CASES)):
        with (args.out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.out / "four_view_events.jsonl").open("w") as stream:
        for row in EVENTS:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    summaries = []
    for model in MODELS:
        for panel in ("representative101", "diagnostic198", "case12"):
            selected = [r for r in PAIRS if r["model"] == model and r["panel"] == panel]
            proxy = [r for r in selected if r["a_nonmarker_b_marker_proxy"]]
            summaries.append({"model": model, "panel": panel, "pairs": len(selected),
                              "answer_marker_proxy_pairs": len(proxy),
                              "E_negative_IP_raises_answer_margin": sum(r["interaction"] < -1e-8 for r in proxy),
                              "E_positive_IP_lowers_answer_margin": sum(r["interaction"] > 1e-8 for r in proxy),
                              "E_zero": sum(abs(r["interaction"]) <= 1e-8 for r in proxy),
                              "both_in_guided_support": sum(r["pair_in_guided_support"] for r in proxy),
                              "both_in_common_support": sum(r["pair_in_common_support"] for r in proxy),
                              "interpretation": "token-proxy margins, not semantic abstention rates"})
    with (args.out / "four_view_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    receipt = {"status": "complete", "coverage": coverage, "sources": SOURCES,
               "events": len(EVENTS), "pairs": len(PAIRS), "case_paths": len(CASES),
               "maximum_float64_residuals": dict(RESIDUAL), "gpu_seconds": budget,
               "actual_argmax_disagreements_at_verified_float64_ties": ARGMAX_TIES,
               "source_script_sha256": sha(Path(__file__)), "new_GPU_forwards": 0,
               "limits": ["Sparse values verify exported algebra and local normalization; full vocabulary is not reloaded.",
                          "Literal first-token markers are proxies; no semantic abstention is inferred.",
                          "Outcome-conditioned diagnostic rows and representative rows remain separate.",
                          "No eval operating point is chosen by this audit."]}
    (args.out / "verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("status", "events", "pairs", "case_paths",
                                           "maximum_float64_residuals", "gpu_seconds")}, indent=2))


if __name__ == "__main__":
    main()
