"""Validate the fixed census and dataset selection before interventions."""
from collections import Counter
from .io import read_jsonl
from .pipeline import census_tasks, task_id


def validate_census_collection(paths, samples, candidates):
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Invalid fixed candidate inventory")
    expected_samples = {s["id"]: s for s in samples}
    if len(expected_samples) != len(samples):
        raise ValueError("Duplicate manifest samples")
    observed_models = set()
    for path in paths:
        rows = list(read_jsonl(path))
        models = {r["model"] for r in rows}
        if len(models) != 1:
            raise ValueError("Each merged census file must contain exactly one model")
        model = next(iter(models))
        if model not in candidates or model in observed_models:
            raise ValueError("Unexpected or duplicate model census")
        observed_models.add(model)
        expected = {task_id(model, t): t for t in census_tasks(samples)}
        seen = set()
        for r in rows:
            key = r["key"]
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate census task")
            seen.add(key)
            if any(r.get(k) != v for k, v in expected[key].items()):
                raise ValueError("Census task contents differ from the frozen manifest")
            if r.get("status") != "ok" or not r.get("tokens") or not isinstance(r.get("terminated"), bool):
                raise ValueError("Census lacks successful real token and termination evidence")
        if seen != set(expected):
            raise ValueError("Incomplete guided or unguided census")
    if observed_models != set(candidates):
        raise ValueError("Missing fixed candidate census: " + ", ".join(sorted(set(candidates)-observed_models)))


def selected_samples(samples, selection, model):
    rows = [r for r in selection if r["model"] == model]
    datasets = Counter(s["dataset"] for s in samples)
    if len(rows) != len(datasets) or {r["dataset"] for r in rows} != set(datasets):
        raise ValueError("Selection must record every dataset condition for this model")
    for row in rows:
        if row["n"] != datasets[row["dataset"]] or type(row["selected"]) is not bool:
            raise ValueError("Selection denominator or boolean status is invalid")
        if not 0 <= row["n_abstain"] <= row["n"] or row["selected"] != (row["n_abstain"] > 0):
            raise ValueError("Selection must depend solely on original semantic abstention")
    selected = {r["dataset"] for r in rows if r["selected"]}
    return [s for s in samples if s["dataset"] in selected]


def code_identity(root):
    """Record committed source blobs; output-only commits do not alter a run."""
    import subprocess
    paths = ["src/kdm", "configs/kdm", "docs/current/PREREGISTER.md"]
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=root)
    if result.returncode:
        raise ValueError("Commit active source/protocol changes before formal model execution")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=root, text=True)
    if untracked.strip():
        raise ValueError("Untracked active source/protocol must be committed before execution")
    lines = subprocess.check_output(["git", "ls-files", "--stage", "--", *paths], cwd=root, text=True).splitlines()
    if not lines:
        raise ValueError("No committed research source identity")
    return lines


def validate_runtime(root, spec, model, cards):
    """Admit only the registered model and a complete native-interface check."""
    import json
    import os
    import sys
    from pathlib import Path
    from importlib.metadata import version
    from .io import within, file_hash
    if spec.get("key") != model or spec.get("availability") != "resolved":
        raise ValueError("Model spec is unresolved or belongs to another candidate")
    if len(cards) != spec.get("gpu_count") or len(cards) != len(set(cards)):
        raise ValueError("Allocated physical GPU count differs from model spec")
    for i, card in enumerate(sorted(cards, key=int)):
        lock = Path(root) / "outputs/locks" / f"gpu_{card}.lock"
        try:
            owned = Path(os.readlink(f"/proc/self/fd/{20+i}"))
        except OSError as exc:
            raise ValueError("Formal execution requires inherited worker.sh GPU locks") from exc
        if owned != lock:
            raise ValueError("Worker lock does not match allocated physical GPU")
    if os.path.abspath(sys.executable) != os.path.abspath(spec["environment_python"]):
        raise ValueError("Use the model spec's environment Python")
    for package, expected in spec["versions"].items():
        if version(package) != expected:
            raise ValueError("Runtime package version differs from registered spec: " + package)
    check = spec.get("interface_verification", {})
    if not isinstance(check, dict) or check.get("status") != "passed":
        raise ValueError("Native 16-image interface verification has not passed")
    result = json.loads(within(root, check["record"]).read_text())
    if not result.get("passed") or result.get("completed") != 16 or result.get("expected") != 16:
        raise ValueError("Incomplete native-interface evidence")
    verified = result.get("spec", {})
    fields = ("key", "factory", "kwargs", "environment_python", "versions", "dtype", "thinking_mode", "processor", "weights")
    if any(verified.get(field) != spec.get(field) for field in fields):
        raise ValueError("Model, processor or environment changed after interface verification")
    if result["manifest_sha256"] != file_hash(Path(root) / "data/current/interface16.jsonl"):
        raise ValueError("Native-interface sample identity changed")
    dependencies = {"hf.py", "backbone.py", "sid.py"}
    if model in {"minicpm26", "minicpm45", "phi35"}:
        dependencies.add("remote.py")
    if model == "internvl35_8b":
        dependencies.add("internvl_preprocessing.py")
    recorded_sources = result.get("runtime_adapter_sha256", {})
    allowed_sources = {"hf.py", "backbone.py", "sid.py", "remote.py", "internvl_preprocessing.py"}
    if not dependencies <= set(recorded_sources) or not set(recorded_sources) <= allowed_sources:
        raise ValueError("Native-interface evidence omits required adapter source identities")
    for filename, digest in recorded_sources.items():
        if file_hash(Path(root) / "src/kdm/models" / filename) != digest:
            raise ValueError("Adapter changed after native-interface verification: " + filename)
    for weight in spec["weights"]:
        path = Path(spec["kwargs"]["model_path"]) / weight["filename"]
        if not path.is_file() or path.stat().st_size != weight["size_bytes"]:
            raise ValueError("Missing or incomplete registered model weight: " + str(path))
