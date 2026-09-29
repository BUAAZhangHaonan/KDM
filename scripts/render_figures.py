#!/usr/bin/env python3
"""Render registered main-results figures from explicit result tables."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kdm.visualization import plot_semantic_matrix, plot_method_tradeoff
from kdm.io import atomic_json, within


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out-dir', default='figures')
    parser.add_argument('--matrix')
    parser.add_argument('--tradeoff')
    parser.add_argument('--source-note', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = within(root, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    if args.matrix:
        files.extend(plot_semantic_matrix(json.load(open(args.matrix)), out / 'semantic_matrix', args.source_note))
    if args.tradeoff:
        files.extend(plot_method_tradeoff(json.load(open(args.tradeoff)), out / 'method_tradeoff', args.source_note))
    if not files:
        raise ValueError('Provide a main-results matrix or tradeoff table')
    atomic_json(out / 'FIGURE_MANIFEST.json', {'files': files, 'source_note': args.source_note})


if __name__ == '__main__':
    main()
