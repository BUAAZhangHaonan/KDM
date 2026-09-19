"""Keep test fixtures and temporary files inside the project write boundary."""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "cache" / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)

def pytest_configure(config):
    base = config.option.basetemp
    if base is None:
        config.option.basetemp = ROOT / "cache" / "pytest" / f"run-{os.getpid()}"
    elif not Path(base).resolve().is_relative_to(ROOT):
        raise ValueError("pytest basetemp must remain inside the project")
