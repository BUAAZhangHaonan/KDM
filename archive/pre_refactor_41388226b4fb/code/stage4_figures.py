"""Stage 4 figures: fig7 (three-probe R-share), fig8 (confirmatory pairs)."""
import sys, json
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / 'outputs' / 'tables'
F = ROOT / 'outputs' / 'figures'
NICE = {'q4b': 'Qwen3.5-4B', 'q9b': 'Qwen3.5-9B', 'q3vl4b': 'Qwen3-VL-4B',
        'llava16': 'LLaVA-v1.6-7B', 'internvl4b': 'InternVL3.5-4B', 'glm46v': 'GLM-4.6V'}
PNICE = {'ll_probe_v4': 'LL probe (v4, primary)', 'full_number_choice': 'number choice (full, v3)',
         'near10_choice': 'neighbour 10-choice (v3)'}
COLOR = {'ll_probe_v4': '#c0504d', 'full_number_choice': '#4f81bd', 'near10_choice': '#9bbb59'}


def rows(name):
    with open(T / name) as f:
        return list(csv.DictReader(f))


def pnice(pid):
    m, d = pid.split('@')
    return f"{NICE.get(m, m)}\n({d})"


def fig7():
    rb = rows('robustness_probe.csv')
    pairs = []
    for r in rb:
        if r['pair'] not in pairs:
            pairs.append(r['pair'])
    probes = ['ll_probe_v4', 'full_number_choice', 'near10_choice']
    fig, ax = plt.subplots(figsize=(1.9 + 1.5 * len(pairs), 4.0))
    x = np.arange(len(pairs))
    w = 0.26
    for i, pr in enumerate(probes):
        ys, los, his = [], [], []
        for p in pairs:
            r = next((r for r in rb if r['pair'] == p and r['probe'] == pr), None)
            if r and r['R_share'] != '':
                v = float(r['R_share'])
                ys.append(v)
                los.append(v - float(r['lo']) if r['lo'] != '' else 0)
                his.append(float(r['hi']) - v if r['hi'] != '' else 0)
            else:
                ys.append(0); los.append(0); his.append(0)
        ax.bar(x + (i - 1) * w, ys, w, color=COLOR[pr], label=PNICE[pr],
               yerr=[los, his], capsize=2, error_kw={'lw': 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels([pnice(p) for p in pairs], fontsize=8)
    ax.set_ylabel('retrieval-failure share of\ndeficient-stratum errors')
    ax.set_ylim(0, 1)
    ax.axhline(0.20, color='grey', ls=':', lw=1, label='J1 threshold (20%)')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(F / 'fig7.pdf')
    plt.close(fig)


def fig8():
    bc = [r for r in rows('bycause_v4.csv') if r['role'] == 'confirmatory'
          and r['cause'] in ('retrieval', 'recognition')]
    dd = [r for r in rows('bycause_v4.csv') if r['role'] == 'confirmatory'
          and r['cause'] == 'D_dECE_R_minus_K']
    pairs = sorted({r['pair'] for r in bc} | {r['pair'] for r in dd})
    methods = ['vcd', 'mib', 'lcd']
    mn = {'vcd': 'VCD', 'mib': 'MIB', 'lcd': 'LCD'}
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.0 * len(pairs), 6.6), squeeze=False)
    for c, pid in enumerate(pairs):
        x = np.arange(len(methods))
        w = 0.38
        ax = axes[0][c]
        for off, cause, color, lbl in [(-w/2, 'retrieval', '#c0504d', 'retrieval failure'),
                                       (w/2, 'recognition', '#4f81bd', 'recognition failure')]:
            ys, los, his = [], [], []
            for m in methods:
                r = next((r for r in bc if r['pair'] == pid and r['method'] == m and r['cause'] == cause), None)
                ys.append(float(r['acc']) if r else 0)
                los.append(float(r['acc']) - float(r['acc_lo']) if r else 0)
                his.append(float(r['acc_hi']) - float(r['acc']) if r else 0)
            ax.bar(x + off, ys, w, color=color, label=lbl if c == 0 else None,
                   yerr=[los, his], capsize=3, error_kw={'lw': 1})
        ax.set_xticks(x); ax.set_xticklabels([mn[m] for m in methods])
        ax.set_title(f'{pnice(pid)}\n(confirmatory)', fontsize=10)
        ax.set_ylim(0, 1)
        if c == 0:
            ax.set_ylabel('open-set accuracy (direct = 0)')
        ax2 = axes[1][c]
        ys, los, his = [], [], []
        for m in ['vcd', 'lcd']:
            r = next(r for r in dd if r['pair'] == pid and r['method'] == m)
            ys.append(float(r['acc'])); los.append(float(r['acc']) - float(r['acc_lo']))
            his.append(float(r['acc_hi']) - float(r['acc']))
        ax2.bar(np.arange(2), ys, 0.5, color=['#c0504d', '#4f81bd'],
                yerr=[los, his], capsize=3, error_kw={'lw': 1})
        ax2.axhline(0, color='k', lw=0.8)
        ax2.set_xticks(np.arange(2)); ax2.set_xticklabels(['VCD', 'LCD'])
        if c == 0:
            ax2.set_ylabel('D = ΔECE(R) − ΔECE(K)\n(negative ⇒ cost on recognition failures)')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(F / 'fig8.pdf')
    plt.close(fig)


if __name__ == '__main__':
    fig7(); fig8()
    print('figures written: fig7.pdf fig8.pdf')
