#!/usr/bin/env python3
"""Plot four independently observed endpoints without mixing operating points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json, file_hash, within


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    source, out = within(ROOT, a.input), within(ROOT, a.out)
    out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(source)
    models = ['qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b']
    labels = ['Qwen2.5-VL-7B', 'Qwen3.5-4B', 'LLaVA-v1.6-Mistral-7B', 'MiniCPM-V-2.6', 'Gemma-3-4B']
    endpoints = ['accuracy', 'reasonable_abstention_retention', 'abstention_precision', 'abstention_recall']
    titles = ['Accuracy', 'Retention on the frozen reasonable-abstention set', 'Abstention precision', 'Abstention recall']
    if len(data) != 20 or data.duplicated(['model', 'endpoint']).any():
        raise ValueError('The verified five-by-four endpoint table is required')
    colors = ['#606875', '#267b96']
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), layout='constrained')
    for ax, endpoint, title in zip(axes.flat, endpoints, titles):
        block = data[data.endpoint.eq(endpoint)].set_index('model').loc[models]
        y = np.arange(5)
        native, ip = block.native_value.to_numpy() * 100, block.ip_value.to_numpy() * 100
        ax.barh(y - .19, native, .34, color=colors[0], label='Native unguided VCD')
        ax.barh(y + .19, ip, .34, color=colors[1], label='IP-VCD observed endpoint')
        for i, (_, row) in enumerate(block.iterrows()):
            marker = {'UNKNOWN': 'U', 'UNCLEAR': 'C', 'UNSURE': 'S', 'I cannot identify it': 'I'}[row.marker]
            native_text, ip_text = f'{native[i]:.2f}', f'{ip[i]:.2f}'
            if row.native_denominator == 1:
                native_text += ' [1/1]'
            if row.ip_denominator == 1:
                ip_text += ' [1/1]'
            ip_text += f' ({marker})'
            for value, offset, text in ((native[i], -.19, native_text),
                                       (ip[i], .19, ip_text)):
                if np.isfinite(value):
                    ax.text(value + .7, i + offset, text, va='center', fontsize=8.5)
                else:
                    ax.text(.7, i + offset, 'undefined (0 abstentions)', va='center', fontsize=8, color=colors[0])
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontsize=10.5)
        ax.set_xlim(0, 124 if endpoint in ('abstention_precision', 'reasonable_abstention_retention') else 109)
        ax.set_xlabel('Percent')
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Food-101 eval: native VCD vs independent IP-VCD endpoints (2,424 inputs/model)', fontsize=13)
    fig.supxlabel('Independent observed eval endpoints; [1/1] has one observation. U=UNKNOWN, C=UNCLEAR, I=I cannot identify it.', fontsize=8)
    axes[0, 0].legend(loc='lower right', fontsize=8, frameon=False)
    fig.savefig(out / 'native_vs_IP_four_endpoints.png', dpi=200)
    fig.savefig(out / 'native_vs_IP_four_endpoints.pdf')
    plt.close(fig)
    data.to_csv(out / 'figure_data.csv', index=False)
    atomic_json(out / 'receipt.json', {'source': str(source.relative_to(ROOT)), 'source_sha256': file_hash(source),
                                    'independent_endpoint_choices': True, 'N_per_model': 2424,
                                    'native_precision_100_percent_denominators': {'minicpm26': 1, 'gemma3_4b': 1},
                                    'retention_fixed_set': 'frozen guided Direct abstain intersect frozen uniformR at chosen marker',
                                    'GPU_initialized': False, 'new_generations': 0, 'renderer_sha256': file_hash(Path(__file__))})
    print(json.dumps({'status': 'complete', 'source_endpoint_rows': 20, 'figures': 2}))


if __name__ == '__main__':
    main()
