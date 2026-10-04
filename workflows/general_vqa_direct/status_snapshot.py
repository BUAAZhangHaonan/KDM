"""One compact read of this experiment's ledgers, claims and timing evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/general_vqa_direct/run_20261004")
    args = parser.parse_args()
    base = ROOT / args.output
    models = []
    for model in sorted(base.iterdir()):
        if not model.is_dir() or model.name == "launches":
            continue
        ledger = model / "completed_keys.jsonl"
        completed = Counter()
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                if line.strip():
                    completed[json.loads(line)["dataset"]] += 1
        claims = []
        for path in sorted((model / "claims").glob("*/owner.json")):
            owner = json.loads(path.read_text())
            run = path.parent
            proc = Path(f"/proc/{owner['pid']}/stat")
            alive = proc.exists() and int(proc.read_text().rsplit(")", 1)[1].split()[19]) == owner["start_tick"]
            timing = {}
            for f in run.glob("first8_*.json"):
                measurement = json.loads(f.read_text())
                timing[measurement["dataset"]] = round(measurement["mean_input_wall_s"], 4)
            failures = json.loads((run / "failed.json").read_text()) if (run / "failed.json").exists() else None
            pending = sum(len(p.read_text().splitlines()) for p in run.glob("*.pending.jsonl"))
            claims.append({"claim_id": owner["claim_id"], "pid": owner["pid"], "alive": alive,
                           "expected": len(owner["keys"]), "pending_unsealed_rows": pending,
                           "first8_seconds_per_input": timing, "complete": (run / "complete.json").exists(),
                           "released": (run / "released.json").exists(),
                           "failed": None if failures is None else {"key": failures["key"], "error": failures["error"]}})
        models.append({"model": model.name, "sealed_by_dataset": dict(completed),
                       "sealed_total": sum(completed.values()), "claims": claims})
    early_failures = []
    for receipt_path in sorted((base / "launches").glob("*.launch.json")):
        rec = json.loads(receipt_path.read_text())
        if not (base / rec["model"] / "claims" / rec["claim_id"] / "owner.json").exists():
            early_failures.append({"model": rec["model"], "pid_alive": Path(f"/proc/{rec['pid']}").exists(),
                "log_tail": (ROOT / rec["log"]).read_text(errors="replace").splitlines()[-8:]})
    print(json.dumps({"host": socket.gethostname(), "utc": datetime.now(timezone.utc).isoformat(),
                      "models": models, "before_claim": early_failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
