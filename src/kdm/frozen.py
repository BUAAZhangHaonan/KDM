#!/usr/bin/env python3
"""Verify canonical frozen copies and the original study/runtime proof chain."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BUNDLE = Path("data/provenance/frozen_contract")
SOURCE_RETIREMENT_FILES = {"src/kdm/migration.py"}
ACTIVE_MIGRATION_FILES = {
    "src/kdm/cli.py", "src/kdm/protocol.py", "src/kdm/provenance.py",
    "src/kdm/task_provenance.py", "src/kdm/frozen.py",
    "workflows/formal/generate.py", "workflows/formal/verify_provenance.py",
    "scripts/run_census_panel.py", "scripts/complete_response_audit.py", "scripts/render_figures.py",
    "verification/check_full_visual_counts.py", "verification/max_input_resource_check.py",
    "docs/PROTOCOL_REGISTER.md", "docs/PROTOCOL.md", "docs/THEORY.md",
    "docs/REPRODUCIBILITY.md",
}

def frozen_path(root: Path, original_path: str) -> Path:
    """Resolve an original frozen path to its content-addressed canonical blob."""
    root = Path(root).resolve()
    manifest_path = root / BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        if root == ROOT:
            raise ValueError("Canonical frozen-proof manifest is missing")
        return root / original_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["entries"]:
        if item["original_path"] == original_path:
            path = root / item["canonical_blob_path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"Canonical frozen blob SHA mismatch: {original_path}")
            return path
    return root / original_path

def canonical_contract_sha256(root: Path) -> str:
    root = Path(root).resolve()
    manifest_path = root / BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        if root == ROOT:
            raise ValueError("Canonical frozen-proof manifest is missing")
        legacy = root / "outputs/records/preregistration_freeze.json"
        return sha(legacy)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / manifest["canonical_contract_copy"]
    actual = sha(path)
    if actual != manifest["original_contract_sha256"]:
        raise ValueError("Canonical freeze contract SHA mismatch")
    return actual

def canonical_runtime_spec(root: Path, spec: dict) -> dict:
    manifest_path = Path(root) / BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        if Path(root).resolve() == ROOT:
            raise ValueError("Canonical frozen-proof manifest is missing")
        return copy.deepcopy(spec)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = {item["original_path"]: item["canonical_blob_path"] for item in manifest["entries"]}
    return canonical_spec(Path(root), spec, mapping)

def supporting_path(root: Path, original_path: str) -> Path:
    """Resolve a supplemental reproducibility input outside the frozen contract."""
    root = Path(root).resolve()
    manifest = json.loads((root / BUNDLE / "supporting_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["entries"]:
        if item["original_path"] == original_path:
            path = root / item["canonical_blob_path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"Supporting proof blob SHA mismatch: {original_path}")
            return path
    raise ValueError(f"No canonical supporting proof registered for: {original_path}")


def current_code_identity(root: Path) -> list[dict]:
    root = Path(root).resolve()
    paths = {*root.joinpath("src/kdm").rglob("*.py"), *root.joinpath("configs/kdm").glob("*.json"),
             *root.joinpath("workflows/formal").glob("*.py")}
    paths.update(root / relative for relative in ACTIVE_MIGRATION_FILES if (root / relative).is_file())
    files = sorted(paths)
    return [{"path": str(path.relative_to(root)), "sha256": sha(path)} for path in files if path.is_file()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _contract(root: Path):
    folder = root / BUNDLE
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    contract_copy = root / manifest["canonical_contract_copy"]
    if sha(contract_copy) != manifest["canonical_contract_sha256"]:
        raise ValueError("Canonical freeze contract copy SHA mismatch")
    freeze = json.loads(contract_copy.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or not isinstance(freeze.get("files"), dict):
        raise ValueError("Canonical contract is not a frozen contract")
    original_contract = root / manifest["original_contract_path"]
    if original_contract.exists() and sha(original_contract) != manifest["original_contract_sha256"]:
        raise ValueError("Original freeze contract SHA mismatch")
    if sha(contract_copy) != manifest["original_contract_sha256"]:
        raise ValueError("Canonical contract does not match archived original identity")
    return manifest, freeze

def load_contract(root: Path) -> tuple[dict, dict]:
    """Load the canonical contract; minimal temporary-root fixtures may use a local contract file."""
    root = Path(root).resolve()
    manifest_path = root / BUNDLE / "manifest.json"
    if manifest_path.is_file():
        return _contract(root)
    if root == ROOT:
        raise ValueError("Canonical frozen-proof manifest is missing")
    path = root / "outputs/records/preregistration_freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    return {"original_contract_sha256": sha(path)}, freeze


def verify_bundle(root: Path, *, require_originals: bool = False, check_originals: bool = True) -> dict:
    """Check canonical copies and, while present, original source-path copies."""
    root = Path(root).resolve()
    manifest, freeze = _contract(root)
    seen = set(); original_verified = 0
    for item in manifest["entries"]:
        source_rel = item["original_path"]
        if source_rel in seen or freeze["files"].get(source_rel) != item["sha256"]:
            raise ValueError(f"Manifest does not bind uniquely to freeze contract: {source_rel}")
        seen.add(source_rel)
        copy_path = root / item["canonical_blob_path"]
        if sha(copy_path) != item["sha256"]:
            raise ValueError(f"Canonical proof copy SHA mismatch: {source_rel}")
        source_path = root / source_rel
        if check_originals and source_path.exists():
            if sha(source_path) != item["sha256"]:
                raise ValueError(f"Original proof SHA mismatch: {source_rel}")
            original_verified += 1
        elif check_originals and require_originals:
            raise ValueError(f"Original proof path is required by caller: {source_rel}")
    if seen != set(freeze["files"]):
        raise ValueError("Canonical package omits or adds frozen proof files")
    contract_path = root / manifest["original_contract_path"]
    if require_originals and not contract_path.exists():
        raise ValueError("Original freeze contract is required by caller")
    return {"files_verified": len(seen), "original_files_still_present": original_verified,
            "contract_sha256": manifest["original_contract_sha256"],
            "frozen_source_bytes": sum(item["size_bytes"] for item in manifest["entries"]),
            "canonical_copies_verified": True}

def _canonical_rel(root: Path, rel: str, mapping: dict[str, str]) -> str:
    if rel in mapping:
        return mapping[rel]
    if (root / rel).is_file():
        return rel
    raise ValueError(f"No canonical copy registered for proof path: {rel}")

def canonical_spec(root: Path, spec: dict, mapping: dict[str, str]) -> dict:
    """Copy a runtime spec and relocate only its proof-record path fields."""
    result = copy.deepcopy(spec)
    visual_record = result.get("visual_count_verification", {}).get("record")
    def walk(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "record" and isinstance(item, str):
                    value[key] = _canonical_rel(root, item, mapping)
                elif key == "records" and isinstance(item, dict):
                    value[key] = {stage: _canonical_rel(root, path, mapping) if isinstance(path, str) else walk(path)
                                  for stage, path in item.items()}
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(result)
    # Resource proofs compare this declared original path to the path embedded
    # in each historic proof; the mapped within() resolves bytes canonically.
    if visual_record is not None:
        result["visual_count_verification"]["record"] = visual_record
    return result

def _verify_source_commit(root: Path, manifest: dict, freeze: dict) -> bool:
    """Verify historical source identities from the recorded Git commit."""
    import subprocess
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or commit != freeze.get("source_commit"):
        raise ValueError("Canonical bundle source commit differs from frozen contract")
    for line in freeze.get("source_blobs", []):
        # Git ls-files format is: mode SP blob SP stage TAB path.
        meta, path = line.split("\t", 1)
        mode, blob, stage = meta.split(" ", 2)
        resolved = subprocess.check_output(["git", "rev-parse", f"{commit}:{path}"], cwd=root, text=True).strip()
        if resolved != blob:
            raise ValueError(f"Frozen source commit blob mismatch: {path}")
    return True


def _verify_current_frozen_inputs(root: Path, freeze: dict) -> None:
    """Bind active config/data files to the original frozen byte identities."""
    prefixes = ("configs/kdm/", "configs/runtime/")
    paths = {path for path in freeze["files"] if path.startswith(prefixes)}
    paths.update({"data/current/all.jsonl", "data/current/interface16.jsonl"})
    for relative in sorted(paths):
        expected = freeze["files"].get(relative)
        path = root / relative
        if not expected or not path.is_file() or sha(path) != expected:
            raise ValueError(f"Active frozen input differs from registered bytes: {relative}")


def _verify_current_algorithms(root: Path, freeze: dict) -> None:
    """Keep original decoding/model algorithm blobs byte-identical to the source commit."""
    import subprocess
    for line in freeze.get("source_blobs", []):
        meta, relative = line.split("\t", 1)
        _mode, expected_blob, _stage = meta.split(" ", 2)
        if not relative.startswith("src/kdm/") or relative in ACTIVE_MIGRATION_FILES or relative in SOURCE_RETIREMENT_FILES:
            continue
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Frozen algorithm source is missing: {relative}")
        actual_blob = subprocess.check_output(["git", "hash-object", str(path)], cwd=root, text=True).strip()
        if actual_blob != expected_blob:
            raise ValueError(f"Algorithm source differs from the frozen source commit: {relative}")


def validate_canonical_freeze(root: Path):
    """Run frozen checks through canonical proof blobs and current active inputs.

    Algorithm modules remain byte-identical to the registered source commit.
    The listed provenance and entrypoint files implement the path migration;
    runtime specifications retain their original model and method values.
    """
    root = Path(root).resolve()
    proof_receipt = verify_bundle(root, check_originals=False)
    support_path = root / BUNDLE / 'supporting_manifest.json'
    support_sha = sha(support_path)
    support = json.loads(support_path.read_text(encoding='utf-8'))
    for item in support['entries']:
        if sha(root / item['canonical_blob_path']) != item['sha256']:
            raise ValueError('Supporting proof blob SHA mismatch: ' + item['original_path'])
    manifest, freeze = _contract(root)
    mapping = {x["original_path"]: x["canonical_blob_path"] for x in manifest["entries"]}
    from kdm.io import read_jsonl
    from kdm.protocol import (validate_native_runtime_files, validate_method_runtime,
                              validate_resource_runtime)
    import kdm.io as kdm_io
    from kdm.reports import validated_method_plan
    from kdm.execution import read_registry

    panel = json.loads((root / "configs/kdm/models.json").read_text())
    keys = [row["key"] for row in panel]
    if len(keys) != 16 or len(set(keys)) != 16:
        raise ValueError("Freeze requires the fixed sixteen-model inventory")
    required = {'docs/current/PREREGISTER.md','data/current/all.jsonl','data/current/interface16.jsonl',
        'configs/kdm/models.json','configs/kdm/food_aliases.json','configs/kdm/method_plan.json',
        'outputs/records/protocol_user_decisions_20260919.json','configs/runtime/semantic_judge.json',
        'scripts/run_census_panel.py','scripts/worker.sh','scripts/verify_complete.py',
        'configs/runtime/hosts.json','outputs/records/image_content_catalog.json'}
    samples = list(read_jsonl(root / 'data/current/all.jsonl'))
    if len(samples) != 9167 or len({row['id'] for row in samples}) != 9167 or Counter(row['dataset'] for row in samples) != {'food101':4848,'vizwiz':4319}:
        raise ValueError("Freeze requires all original 9167 Food-101/VizWiz samples")
    catalog = json.loads(frozen_path(root, 'outputs/records/image_content_catalog.json').read_text())
    from kdm.io import file_hash
    if catalog.get('schema') != 1 or catalog.get('manifest_sha256') != file_hash(root/'data/current/all.jsonl') or set(catalog.get('images',{})) != {row['image_path'] for row in samples}:
        raise ValueError("Image content catalog must cover the exact full original manifest")
    if any(not isinstance(v.get('sha256'),str) or len(v['sha256']) != 64 or type(v.get('size_bytes')) is not int or v['size_bytes'] <= 0 for v in catalog['images'].values()):
        raise ValueError("Image content catalog lacks complete byte identities")
    plan = json.loads((root/'configs/kdm/method_plan.json').read_text())
    methods = validated_method_plan(plan, [(key,dataset) for key in keys for dataset in ('food101','vizwiz')])
    registry = read_registry(root)
    if set(registry['model_hosts']) != set(keys):
        raise ValueError("Host registry must assign all sixteen models exactly once")
    specs = {}
    for key in keys:
        relative = f'configs/runtime/{key}.json'; required.add(relative)
        original_spec = json.loads((root/relative).read_text()); specs[key] = original_spec
        if original_spec.get('key') != key or original_spec.get('availability') != 'resolved':
            raise ValueError('Unresolved frozen runtime: '+key)
        declaration = original_spec.get('interface_verification',{})
        if not isinstance(declaration,dict) or declaration.get('status') != 'passed':
            raise ValueError('Native interface is incomplete: '+key)
        required.add(declaration['record'])
        methods_requested = {method for values in methods[key].values() for method in values}
        if 'sid' in methods_requested:
            sid = original_spec.get('mechanism_validation',{}).get('sid_reference')
            if not isinstance(sid,dict) or sid.get('status') != 'passed':
                raise ValueError('Frozen SID method proof is incomplete: '+key)
            required.add(sid['record'])
    reverse_mapping = {canonical: original for original, canonical in mapping.items()}
    original_within = kdm_io.within
    def mapped_within(check_root, relative):
        rel = str(relative)
        return original_within(check_root, mapping.get(rel, rel))
    kdm_io.within = mapped_within
    try:
      for key, original_spec in specs.items():
        proof_spec = canonical_spec(root, original_spec, mapping)
        validate_native_runtime_files(root, proof_spec, key)
        validate_method_runtime(root, proof_spec, {m for values in methods[key].values() for m in values})
        if registry['model_hosts'][key] == '6403':
            resource_paths = validate_resource_runtime(root, proof_spec)
            required.update(reverse_mapping.get(rel, rel) for rel in resource_paths)
    finally:
        kdm_io.within = original_within
    if not required <= set(freeze['files']):
        raise ValueError('Freeze contract omits required protocol/data/runtime/proof identities: ' + ', '.join(sorted(required-set(freeze['files']))))
    if not _verify_source_commit(root, manifest, freeze):
        raise ValueError('Frozen historical source blobs differ from the recorded source commit')
    _verify_current_frozen_inputs(root, freeze)
    _verify_current_algorithms(root, freeze)
    freeze = copy.deepcopy(freeze)
    freeze['_canonical_contract_sha256'] = manifest['original_contract_sha256']
    freeze['_proof_manifest_sha256'] = sha(root / BUNDLE / 'manifest.json')
    freeze['_supporting_manifest_sha256'] = support_sha
    freeze['_proof_path_map'] = mapping
    freeze['_current_code_identity'] = current_code_identity(root)
    freeze['_active_migration_files'] = sorted(ACTIVE_MIGRATION_FILES)
    freeze['_retired_source_files'] = sorted(SOURCE_RETIREMENT_FILES)
    return freeze, proof_receipt

