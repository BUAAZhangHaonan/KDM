"""Package the already checked paper export and record the current Git receipt."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_20260929"
NAME = "KDM_Paper_Data_Export_20260929.zip"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def payload_files():
    return sorted(
        p for p in OUT.rglob("*")
        if p.is_file()
        and "work" not in p.relative_to(OUT).parts
        and "__pycache__" not in p.relative_to(OUT).parts
        and p.name not in (NAME, "version_before.json", "git_inventory.md", "package_receipt.json")
        and not p.name.endswith(".pyc")
    )


def main():
    check = json.loads((OUT / "export_checks.json").read_text())
    before = json.loads((OUT / "version_before.json").read_text())
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "origin", "refs/heads/master").split()[0]
    if branch != "master" or remote != head:
        raise RuntimeError(f"Publication receipt mismatch: branch={branch}, head={head}, remote={remote}")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    status = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    status_counts = dict(Counter(line[:2] for line in status))
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    top_counts = dict(Counter("/".join(path.split("/")[:2]) for path in untracked))
    commits = git("log", "--reverse", "--format=%H %s", f"{before['head']}..HEAD").splitlines()
    initial_remote = before["origin_master_live"]
    if isinstance(initial_remote, dict):
        initial_remote = initial_remote.get("sha", initial_remote)
    (OUT / "version_receipt.md").write_text(f"""# Version receipt

Generated UTC: {now}

- Canonical project and actual server-local checkout: `{ROOT}`
- Branch: `{branch}`
- Export start local HEAD: `{before['head']}`
- Export start live remote master: `{initial_remote}`
- Published local HEAD: `{head}`
- Verified current remote master: `{remote}`
- Remote: `{git('remote', 'get-url', 'origin')}`
- User-uploaded paper-start package date: 2026-09-29.
- Current completed results: `outputs/annotations/main_results/`, `outputs/annotations/reference_gt/`, `outputs/analysis/main_results/`, `data/responses/`.
- New export: `outputs/paper_20260929/`; actual scripts: `scripts/paper_20260929/`.

## Relationship to the connector snapshot

The user-reported connector snapshot was `1684809caf5593798cee049f13f0714931d7cf25`, dated 2026-09-24. At export start the actual server checkout already contained 12 later local commits, plus the completed canonical file cleanup, final manuscript/result tables and local large result files. This task recorded that completed checkout in six bounded commits, then committed the CPU export scripts and compact handoff data. A normal fast-forward push published the resulting master. The current published HEAD above was checked directly against the remote after push.

## Commits created during this export

""" + "\n".join(f"- `{line.split(' ', 1)[0]}` — {line.split(' ', 1)[1]}" for line in commits) + f"""

## Local files and source retention

Git working-tree status counts at receipt capture: `{json.dumps(status_counts, ensure_ascii=False)}`. Untracked paths by first two components: `{json.dumps(top_counts, ensure_ascii=False)}`. The original 77 large local result payloads (1,030,098,652 bytes) were retained; their earlier byte-size inventory is recorded in git_commit_receipt.json. The export workspace contains the reusable SQLite join index and package artifacts. The final version receipt, manifest and archive are generated after the published code/data commit.

Source mappings are in sources.csv, prompts_and_configs.json and context/LARGE_ASSETS.json. The raw source and scoring files retained their existing paths/content. The export stores direct mappings and copied values; checks use counts, keys, source-line binding and image-byte comparisons.

Export checks passed: `{check['passed']}`. Detailed CPU checks: export_checks.json.
""", encoding="utf-8")
    script_dest = OUT / "scripts"
    script_dest.mkdir(exist_ok=True)
    for script in (ROOT / "scripts/paper_20260929").glob("*.py"):
        target = script_dest / script.name
        if not target.exists() or target.read_bytes() != script.read_bytes():
            shutil.copyfile(script, target)
    files = payload_files()
    # Inventory lists byte sizes and roles; source content hashes are not recomputed.
    manifest = {"generated_utc": now, "root": str(ROOT), "published_head": head, "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size} for p in files if p.name != "file_manifest.json"]}
    write_json(OUT / "file_manifest.json", manifest)
    files = payload_files()
    total = sum(p.stat().st_size for p in files)
    archive = OUT / NAME
    if archive.exists():
        raise FileExistsError(f"Completed package already exists: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in files:
            package.write(path, path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as package:
        error = package.testzip()
        if error:
            raise RuntimeError(f"ZIP CRC check failed at {error}")
        required = {"README.md", "schema.md", "conditions.csv", "sources.csv", "scores.parquet", "references.parquet", "control_selections.parquet", "cases.jsonl", "runtime_records.csv", "availability.csv", "version_receipt.md", "export_checks.json"}
        missing = sorted(required-set(package.namelist()))
        if missing:
            raise RuntimeError(f"Missing package entries: {missing}")
    receipt = {"generated_utc": now, "archive": str(archive), "archive_bytes": archive.stat().st_size, "uncompressed_bytes": total, "files": len(files), "zip_crc_check": "passed", "required_entries": "complete", "published_master": head, "export_checks_passed": check["passed"]}
    write_json(OUT / "package_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
