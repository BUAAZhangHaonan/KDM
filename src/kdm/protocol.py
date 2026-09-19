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
