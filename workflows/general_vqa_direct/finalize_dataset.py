"""Freeze one approved benchmark roster and its answer-only Direct prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODELS = ["qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl"]
COUNTS = {"mmmu": 1000, "scienceqa": 1000, "pope": 1000, "hallusionbench": 951}
SOURCES = {
    "mmmu": "https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/configs/llava1.5.yaml",
    "scienceqa": "https://github.com/lupantech/ScienceQA/blob/main/models/base_prompt.py",
    "pope": "https://github.com/RUCAIBox/POPE",
    "hallusionbench": "https://github.com/tianyi-lab/HallusionBench",
}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_prompt(row):
    dataset = row["dataset"]
    question = row["question"]
    if dataset in {"pope", "hallusionbench"}:
        return question
    options = row.get("options", [])
    if not isinstance(options, list):
        raise ValueError("Options must already be parsed from the official dataset")
    if dataset == "scienceqa":
        choices = " ".join(f"({chr(65+i)}) {value}" for i, value in enumerate(options))
        context = row.get("hint", "").strip() or "N/A"
        return f"Question: {question}\nContext: {context}\nOptions: {choices}\nAnswer:".replace("  ", " ").strip()
    if dataset == "mmmu":
        if row["question_type"] == "multiple-choice":
            choices = "\n".join(f"({chr(65+i)}) {value}" for i, value in enumerate(options))
            return question + "\n\n" + choices + "\n\nAnswer with the option's letter from the given choices directly."
        return question + "\n\nAnswer the question using a single word or phrase."
    raise ValueError(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=COUNTS, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--budget-source", required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_tokens == 32:
        raise ValueError("A positive, separately justified benchmark budget is required; the old 32-token budget is withdrawn")
    root = args.root.resolve()
    base = root / "data/general_vqa_direct_20261004"
    source = base / f"manifest_{args.dataset}.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) != COUNTS[args.dataset] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate roster")
    for row in rows:
        if row["dataset"] != args.dataset:
            raise ValueError("Dataset mismatch")
        paths, hashes = row["image_paths"], row["image_sha256"]
        if not paths or len(paths) != len(hashes):
            raise ValueError("All image paths and byte hashes must be present")
        for path, expected in zip(paths, hashes):
            actual_path = (root / path).resolve()
            if Path(path).is_absolute() or not actual_path.is_relative_to(root):
                raise ValueError("Image path must be project relative")
            if file_sha(actual_path) != expected:
                raise ValueError(f"Image bytes do not match: {path}")
        row["prompt"] = format_prompt(row)
        row["prompt_source"] = SOURCES[args.dataset]
        row["guidance"] = "native_dataset_format_no_abstention_instruction"
    frozen = base / "frozen"
    frozen.mkdir(exist_ok=True)
    dest = frozen / f"manifest_{args.dataset}.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    if dest.exists() and dest.read_text() != payload:
        raise FileExistsError(f"Frozen input differs: {dest}")
    if not dest.exists():
        dest.write_text(payload, encoding="utf-8")
    protocol_path = frozen / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {
        "schema": "kdm_general_vqa_direct_protocol_v1", "models": MODELS,
        "method": "direct", "temperature": 0.0, "top_p": 1.0,
        "datasets": {}, "sampling_seed": 20261004,
        "scoring": "dataset_correctness_and_separate_full_response_semantic_abstention",
    }
    entry = {"manifest": str(dest.relative_to(root)), "manifest_sha256": file_sha(dest),
             "expected_rows": len(rows), "max_tokens": args.max_tokens,
             "budget_source": args.budget_source, "source_roster": str(source.relative_to(root)),
             "source_roster_sha256": file_sha(source), "prompt_source": SOURCES[args.dataset]}
    if args.dataset in protocol["datasets"] and protocol["datasets"][args.dataset] != entry:
        raise FileExistsError("An existing frozen dataset protocol cannot change")
    protocol["datasets"][args.dataset] = entry
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, **entry}, ensure_ascii=False))


if __name__ == "__main__":
    main()
