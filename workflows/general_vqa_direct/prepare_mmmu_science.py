#!/usr/bin/env python3
"""Prepare source-bound, deterministic MMMU and ScienceQA evaluation assets.

CPU only. This file neither constructs final prompts nor invokes a model.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any

SEED = 20261004
SCIENCE_REVISION = "2cbf8318e07b9ece895bb2ae605e71e38d623264"
MMMU_CODE_REVISION = "268471d0d488258990025331c7528359c324aa25"
MMMU_DATA_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SCIENCE_IMAGES = "https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip"
RELATIVE_DATA = Path("data/general_vqa_direct_20261004")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stable_key(dataset: str, source_id: str) -> str:
    return digest(f"{SEED}:{dataset}:{source_id}".encode("utf-8"))


def save_frozen(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to change frozen file: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace an existing different manifest: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def source_record(path: Path, uri: str, revision: str) -> dict[str, Any]:
    return {"filename": path.name, "uri": uri, "revision": revision,
            "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verify_image(data: bytes) -> dict[str, Any]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        result = {"width": image.width, "height": image.height, "format": image.format}
        image.verify()
    return result


def write_image(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing image differs from source bytes: {path}")
    else:
        with path.open("xb") as f:
            f.write(data)


def prepare_scienceqa(root: Path, workers: int) -> None:
    dest = root / RELATIVE_DATA
    meta = dest / "source_metadata"
    problems_path = meta / "scienceqa_problems.json"
    splits_path = meta / "scienceqa_pid_splits.json"
    problems = load_json(problems_path)
    splits = load_json(splits_path)
    test_ids = list(map(str, splits["test"]))
    assert len(test_ids) == len(set(test_ids)) == 4241
    assert all(problems[q]["split"] == "test" for q in test_ids)
    candidates = [q for q in test_ids if problems[q].get("image") is not None]
    assert len(candidates) == 2017
    selected = sorted(candidates, key=lambda q: (stable_key("scienceqa", q), q))[:1000]
    frozen = {
        "dataset": "scienceqa", "source_split": "test", "seed": SEED,
        "eligible": "official test IDs with non-null image", "candidate_count": len(candidates),
        "selection_rule": "ascending SHA256(UTF8('20261004:scienceqa:' + official_id)), then official_id; first 1000",
        "selected_ids": selected, "candidate_ids_sha256": digest("\n".join(sorted(candidates)).encode()),
        "sources": [
            source_record(problems_path, f"https://raw.githubusercontent.com/lupantech/ScienceQA/{SCIENCE_REVISION}/data/scienceqa/problems.json", SCIENCE_REVISION),
            source_record(splits_path, f"https://raw.githubusercontent.com/lupantech/ScienceQA/{SCIENCE_REVISION}/data/scienceqa/pid_splits.json", SCIENCE_REVISION),
        ],
    }
    save_frozen(meta / "selection_scienceqa_1000.json", frozen)
    print(json.dumps({"event": "selection_frozen", "dataset": "scienceqa", "count": len(selected), "sha256": file_sha(meta / "selection_scienceqa_1000.json")}), flush=True)

    wheel = meta / "remotezip-0.12.6-py3-none-any.whl"
    assert file_sha(wheel) == "ffbfe0d8658fd56b8de540686ae647e50150f6ee7ec25a2cd96fa6fbadb84418"
    sys.path.insert(0, str(wheel))
    from remotezip import RemoteZip
    zip_members = {r["member"]: r for r in load_json(meta / "scienceqa_test_zip_members.json")}
    image_results: dict[str, dict[str, Any]] = {}

    def read_images(qids: list[str]) -> dict[str, dict[str, Any]]:
        output = {}
        with RemoteZip(SCIENCE_IMAGES, timeout=(20, 60)) as archive:
            for qid in qids:
                member = f"test/{qid}/{problems[qid]['image']}"
                expected = zip_members[member]
                path = dest / "images/scienceqa" / qid / problems[qid]["image"]
                if path.exists():
                    image_bytes = path.read_bytes()
                    import zlib
                    assert len(image_bytes) == expected["bytes"]
                    assert f"{zlib.crc32(image_bytes) & 0xffffffff:08x}" == expected["crc32"]
                else:
                    image_bytes = archive.read(member)  # zipfile validates the member CRC.
                    assert len(image_bytes) == expected["bytes"]
                    write_image(path, image_bytes)
                details = verify_image(image_bytes)
                output[qid] = {"path": path.relative_to(root).as_posix(), "sha256": digest(image_bytes),
                               "source_member": member, "source_crc32": expected["crc32"],
                               "bytes": len(image_bytes), **details}
        return output

    groups = [selected[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_images, group) for group in groups if group]
        for future in as_completed(futures):
            image_results.update(future.result())
            print(json.dumps({"event": "images_verified", "dataset": "scienceqa", "count": len(image_results)}), flush=True)
    assert set(image_results) == set(selected)
    rows = []
    for qid in selected:
        problem = problems[qid]
        image = image_results[qid]
        choices = problem["choices"]
        assert 0 <= problem["answer"] < len(choices)
        rows.append({
            "id": f"scienceqa:{qid}", "sample_id": f"scienceqa:{qid}", "source_id": qid,
            "dataset": "scienceqa", "split": "eval", "source_split": "test", "seed": SEED,
            "question": problem["question"], "options": choices, "hint": problem["hint"],
            "gold": chr(65 + problem["answer"]), "gold_index": problem["answer"],
            "question_type": "multiple-choice", "subject": problem["subject"],
            "image_paths": [image["path"]], "image_sha256": [image["sha256"]],
            "image_slots": [1], "image_metadata": [image],
            "source_uri": frozen["sources"][0]["uri"], "source_revision": SCIENCE_REVISION,
            "source_metadata_sha256": frozen["sources"][0]["sha256"],
            "image_source_uri": SCIENCE_IMAGES,
        })
    # Lecture and solution stay only in the immutable source file, never the model manifest.
    assert len(rows) == len({r["id"] for r in rows}) == 1000
    assert all("lecture" not in r and "solution" not in r and "prompt" not in r for r in rows)
    manifest = dest / "manifest_scienceqa.jsonl"
    save_jsonl(manifest, rows)
    summary = {"dataset": "scienceqa", "status": "READY_FOR_PROMPT_FINALIZATION", "questions": 1000,
               "official_test_count": 4241, "eligible_image_test_count": 2017, "images": 1000,
               "multi_image_questions": 0, "image_bytes": sum(x["bytes"] for x in image_results.values()),
               "manifest": manifest.relative_to(root).as_posix(), "manifest_sha256": file_sha(manifest),
               "selection_sha256": file_sha(meta / "selection_scienceqa_1000.json"),
               "subject_counts": dict(Counter(r["subject"] for r in rows)),
               "image_acquisition": "official S3 ZIP; remotezip 0.12.6 HTTP ranges; only selected members",
               "validation": ["1000 unique official test IDs", "2017 image eligibility", "source ZIP CRC for every image", "PIL image decode verification", "SHA256 for every original image byte stream", "lecture/solution absent from manifests"]}
    save_frozen(meta / "prepared_scienceqa_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def prepare_mmmu(root: Path) -> None:
    import pyarrow.parquet as pq
    dest = root / RELATIVE_DATA
    meta = dest / "source_metadata"
    info = load_json(meta / "mmmu_hf_file_metadata.json")
    assert info["sha"] == MMMU_DATA_REVISION
    files = sorted((f for f in info["siblings"] if re.search(r"/test-.*\.parquet$", f["rfilename"])), key=lambda f: f["rfilename"])
    subjects = sorted({f["rfilename"].split("/")[0] for f in files})
    assert len(subjects) == 30 and len(files) == 32
    answer_path = meta / "mmmu_answer_dict_test.json"
    answers = load_json(answer_path)
    assert len(answers) == 10500
    # Only the official key strings enter selection; answer values are not consulted.
    candidates = {subject: sorted(q for q in answers if q.rsplit("_", 1)[0] == f"test_{subject}") for subject in subjects}
    assert sum(map(len, candidates.values())) == 10500
    extra = set(sorted(subjects, key=lambda s: (stable_key("mmmu_subject", s), s))[:10])
    quotas = {subject: 33 + int(subject in extra) for subject in subjects}
    selected_by_subject = {subject: sorted(candidates[subject], key=lambda q: (stable_key("mmmu", q), q))[:quotas[subject]] for subject in subjects}
    selected = [q for subject in subjects for q in selected_by_subject[subject]]
    assert len(selected) == len(set(selected)) == 1000
    answer_source = source_record(answer_path, f"https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/{MMMU_CODE_REVISION}/mmmu/answer_dict_test.json", MMMU_CODE_REVISION)
    frozen = {
        "dataset": "mmmu", "source_split": "test", "seed": SEED, "candidate_count": 10500,
        "source_revision": MMMU_DATA_REVISION, "official_subjects": subjects,
        "subject_quotas": quotas, "candidate_counts": {s: len(candidates[s]) for s in subjects},
        "subject_remainder_rule": "33 per subject; add 1 to first 10 subjects ordered by SHA256(UTF8('20261004:mmmu_subject:' + subject)), then subject",
        "selection_rule": "within each subject, ascending SHA256(UTF8('20261004:mmmu:' + official_id)), then official_id; take subject quota",
        "selected_ids": selected, "candidate_ids_sha256": digest("\n".join(sorted(answers)).encode()),
        "answer_source": answer_source,
    }
    save_frozen(meta / "selection_mmmu_1000.json", frozen)
    print(json.dumps({"event": "selection_frozen", "dataset": "mmmu", "count": len(selected), "sha256": file_sha(meta / "selection_mmmu_1000.json")}), flush=True)
    wanted = set(selected)
    rows_by_id = {}
    all_ids = set()
    sources = []
    issues = {"repeated_reference_ids": [], "unreferenced_image_ids": [], "legacy_image_tag_ids": []}
    for file_info in files:
        name = file_info["rfilename"]
        subject = name.split("/")[0]
        path = meta / "mmmu_parquet" / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned official HF test shard is not staged: {path}")
        assert path.stat().st_size == file_info["size"], name
        actual_sha = file_sha(path)
        assert actual_sha == file_info["lfs"]["sha256"], name
        source_uri = f"https://huggingface.co/datasets/MMMU/MMMU/resolve/{MMMU_DATA_REVISION}/{name}"
        sources.append({"filename": name, "uri": source_uri, "revision": MMMU_DATA_REVISION,
                        "bytes": file_info["size"], "sha256": actual_sha, "official_lfs_sha256_matched": True})
        parquet = pq.ParquetFile(path)
        columns = ["id", "question", "options", "answer", "question_type", "subfield", "img_type"] + [f"image_{i}" for i in range(1, 8)]
        row_number = -1
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for raw in batch.to_pylist():
                row_number += 1
                qid = raw["id"]
                assert qid not in all_ids, qid
                all_ids.add(qid)
                assert qid in answers and qid.rsplit("_", 1)[0] == f"test_{subject}", qid
                if qid not in wanted:
                    continue
                options = ast.literal_eval(raw["options"]) if isinstance(raw["options"], str) else raw["options"]
                assert isinstance(options, list), qid
                assert all(isinstance(x, str) for x in options), qid
                gold = answers[qid]["ground_truth"]
                raw_gold = raw["answer"]
                if isinstance(gold, list) and isinstance(raw_gold, str):
                    raw_gold = ast.literal_eval(raw_gold)
                assert raw_gold == gold, f"HF and official answer file differ: {qid}"
                assert raw["question_type"] == answers[qid]["question_type"], qid
                images = []
                for slot in range(1, 8):
                    field = raw[f"image_{slot}"]
                    if field is None:
                        continue
                    data = field["bytes"]
                    assert isinstance(data, bytes) and data, (qid, slot)
                    details = verify_image(data)
                    suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}.get(details["format"])
                    if suffix is None:
                        raise ValueError(f"Unexpected original image format {qid}: {details['format']}")
                    image_path = dest / "images/mmmu" / qid / f"image_{slot}{suffix}"
                    write_image(image_path, data)
                    images.append({"slot": slot, "path": image_path.relative_to(root).as_posix(),
                                   "sha256": digest(data), "bytes": len(data), "original_path": field.get("path"),
                                   "source_field": f"image_{slot}", "source_parquet_row": row_number, **details})
                assert images, qid
                slots = [image["slot"] for image in images]
                refs = [int(n) for n in re.findall(r"<image\s+(\d+)>", raw["question"] + "\n" + "\n".join(options))]
                missing = sorted(set(refs) - set(slots))
                assert not missing, f"Image placeholder without original image field: {qid} {missing}"
                repeated = sorted(n for n, count in Counter(refs).items() if count > 1)
                unreferenced = sorted(set(slots) - set(refs))
                if repeated:
                    issues["repeated_reference_ids"].append(qid)
                if unreferenced:
                    issues["unreferenced_image_ids"].append(qid)
                if "<img=" in raw["question"] or any("<img=" in o for o in options):
                    issues["legacy_image_tag_ids"].append(qid)
                rows_by_id[qid] = {
                    "id": f"mmmu:{qid}", "sample_id": f"mmmu:{qid}", "source_id": qid,
                    "dataset": "mmmu", "split": "eval", "source_split": "test", "seed": SEED,
                    "question": raw["question"], "options": options, "hint": "", "gold": gold,
                    "question_type": raw["question_type"], "subject": subject, "subfield": raw["subfield"],
                    "image_paths": [image["path"] for image in images], "image_sha256": [image["sha256"] for image in images],
                    "image_slots": slots, "image_metadata": images,
                    "image_placeholder_map": {f"<image {slot}>": i for i, slot in enumerate(slots)},
                    "image_reference_slots": refs, "repeated_image_reference_slots": repeated,
                    "unreferenced_image_slots": unreferenced, "source_uri": source_uri,
                    "source_revision": MMMU_DATA_REVISION, "source_sha256": actual_sha,
                    "source_parquet_row": row_number, "answer_source_uri": answer_source["uri"],
                    "answer_source_sha256": answer_source["sha256"], "answer_source_revision": MMMU_CODE_REVISION,
                }
        print(json.dumps({"event": "mmmu_shard_verified", "file": name, "selected_ready": len(rows_by_id)}), flush=True)
    assert all_ids == set(answers), "HF test and official answer-file key sets differ"
    assert set(rows_by_id) == wanted
    rows = [rows_by_id[qid] for qid in selected]
    assert dict(Counter(row["subject"] for row in rows)) == quotas
    assert all("explanation" not in row and "prompt" not in row for row in rows)
    manifest = dest / "manifest_mmmu.jsonl"
    save_jsonl(manifest, rows)
    save_frozen(meta / "mmmu_source_shards_verified.json", sources)
    histogram = dict(sorted(Counter(len(row["image_paths"]) for row in rows).items()))
    summary = {
        "dataset": "mmmu", "status": "READY_FOR_PROMPT_FINALIZATION", "questions": 1000,
        "official_test_count": len(all_ids), "subjects": 30, "subject_counts": quotas,
        "question_type_counts": dict(Counter(row["question_type"] for row in rows)),
        "images": sum(len(row["image_paths"]) for row in rows),
        "multi_image_questions": sum(len(row["image_paths"]) > 1 for row in rows),
        "image_count_histogram": histogram,
        "image_bytes": sum(image["bytes"] for row in rows for image in row["image_metadata"]),
        "manifest": manifest.relative_to(root).as_posix(), "manifest_sha256": file_sha(manifest),
        "selection_sha256": file_sha(meta / "selection_mmmu_1000.json"),
        "source_parquet_count": len(sources), "source_parquet_bytes": sum(x["bytes"] for x in sources),
        "image_acquisition": "all non-null image_1..image_7 fields, original encoded bytes, ascending slot order",
        "placeholder_audit": issues,
        "validation": ["1000 unique deterministic source IDs", "30 official subject quotas of 33 or 34", "10500 official test-key equality", "every parquet SHA matches official HF LFS", "HF gold equals released GitHub gold", "all selected original image fields retained", "PIL image decode verification", "SHA256 per image", "no explanation in model manifest"],
    }
    save_frozen(meta / "prepared_mmmu_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["scienceqa", "mmmu"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if args.dataset == "scienceqa":
        prepare_scienceqa(args.root.resolve(), args.workers)
    else:
        prepare_mmmu(args.root.resolve())


if __name__ == "__main__":
    main()
