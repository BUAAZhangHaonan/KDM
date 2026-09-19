#!/usr/bin/env python3
"""Retain every original sample in selected model--dataset conditions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdm.io import read_jsonl, within
from kdm.data import write_manifest
from kdm.protocol import selected_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    samples = list(read_jsonl(a.manifest))
    selection = json.loads(Path(a.selection).read_text())
    for model in sorted({r["model"] for r in selection}):
        rows = selected_samples(samples, selection, model)
        if not rows:
            continue
        out = within(a.root, Path(a.out_dir) / (model + ".jsonl"))
        if out.exists():
            if list(read_jsonl(out)) != rows:
                raise ValueError("Cannot overwrite a different selected manifest")
        else:
            write_manifest(rows, out)
        print(model, len(rows), sum(s["split"] == "eval" for s in rows), out)


if __name__ == "__main__":
    main()
