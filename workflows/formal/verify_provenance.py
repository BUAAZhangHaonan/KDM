#!/usr/bin/env python3
"""Command-line access to the canonical frozen-proof verifier."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.frozen import verify_bundle, validate_canonical_freeze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--full-contract", action="store_true")
    parser.add_argument("--verify-present-originals", action="store_true")
    args = parser.parse_args()
    result = verify_bundle(args.root, check_originals=args.verify_present_originals)
    if args.full_contract:
        _, result = validate_canonical_freeze(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
