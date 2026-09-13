"""Stage 3 figures (pair-based): fig4/fig5/fig6."""
import sys, json
from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / 'outputs' / 'tables'
F = ROOT / 'outputs' / 'figures'
NICE = {'q4b': 'Qwen3.5-4B', 'q9b': 'Qwen3.5-9B', 'internvl4b': 'InternVL3.5-4B',
        'q3vl4b': 'Qwen3-VL-4B', 'q3vl8b': 'Qwen3-VL-8B', 'llava16': 'LLaVA-v1.6-7B',
        'glm46v': 'GLM-4.6V-Flash'}
DNICE = {'food101': 'food101', 'dogs': 'dogs'}
MNICE = {'vcd': 'VCD', 'mib': 'MIB', 'lcd': 'LCD'}


def rows(name):
    with open(T / name) as f:
        return list(csv.DictReader(f))


def pairs_of(rs):
    seen = []
    for r in rs:
        if r['pair'] not in seen:
            seen.append(r['pair'])
    return seen


def pnice(pid):
    m, d = pid.split('@')
    return f"{NICE.get(m, m)}\n({DNICE.get(d, d)})"


def grid(n, per_row=3):
    return math.ceil(n / per_row), per_row


def fig4():
    data = [r for r in rows('bycause.csv') if r['cause'] != 'diff_R-K']
    pairs = pairs_of(data)
    methods = ['vcd', 'mib', 'lcd']
    nr, nc = grid(len(pairs))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.6 * nr), squeeze=False, sharey=True)
    for i, pid in enumerate(pairs):
        ax = axes[i // nc][i % nc]
        x = np.arange(len(methods))
        w = 0.38
        for off, cause, color, lbl in [(-w / 2, 'retrieval', '#c0504d', 'retrieval failure'),
                                       (w / 2, 'recognition', '#4f81bd', 'recognition failure')]:
            ys, los, his = [], [], []
            for m in methods:
                r = next((r for r in data if r['pair'] == pid and r['method'] == m and r['cause'] == cause), None)
                ys.append(float(r['acc']) if r else 0)
                los.append(float(r['acc']) - float(r['acc_lo']) if r else 0)
                his.append(float(r['acc_hi']) - float(r['acc']) if r else 0)
            ax.bar(x + off, ys, w, color=color, label=lbl if i == 0 else None,
                   yerr=[los, his], capsize=3, error_kw={'lw': 1})
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MNICE[m] for m in methods])
        ax.set_title(pnice(pid), fontsize=10)
        ax.set_ylim(0, 1.0)
    for axrow in axes:
        axrow[0].set_ylabel('open-set accuracy (direct = 0)')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(F / 'fig4.pdf')
    plt.close(fig)


def fig5():
    eq = rows('equivalence.csv')
    pairs = pairs_of(eq)
    methods = ['vcd', 'lcd']
    nr, nc = grid(len(pairs))
    fig, axes = plt.subplots(2 * nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False, sharey='row')
    for i, pid in enumerate(pairs):
        x = np.arange(len(methods))
        w = 0.38
        for row, metric in [(0, 'd_wrong_conf'), (1, 'd_ece')]:
            ax = axes[2 * (i // nc) + row][i % nc]
            ax.axhspan(-0.05, 0.05, color='grey', alpha=0.15, lw=0)
            ax.axhline(0, color='k', lw=0.8)
            for off, cause, color in [(-w / 2, 'R', '#c0504d'), (w / 2, 'K', '#4f81bd')]:
                ys = []
                for m in methods:
                    r = next(r for r in eq if r['pair'] == pid and r['method'] == m and r['metric'] == metric)
                    ys.append(float(r['mean_' + cause]))
                ax.bar(x + off, ys, w, color=color,
                       label=('retrieval failure' if cause == 'R' else 'recognition failure')
                       if (i == 0 and row == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels([MNICE[m] for m in methods])
            if i % nc == 0:
                ax.set_ylabel('Δ wrong-answer conf.' if row == 0 else 'Δ ECE')
            if row == 0:
                ax.set_title(pnice(pid), fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(F / 'fig5.pdf')
    plt.close(fig)


def fig6():
    ss = rows('signal_split.csv')
    pairs = pairs_of(ss)
    signals = ['entropy', 'jsd_blank', 'logistic_combo']
    snice = {'entropy': 'entropy', 'jsd_blank': 'JSD(blank)', 'logistic_combo': 'combination'}
    nr, nc = grid(len(pairs))
    fig, axes = plt.subplots(nr, nc, figsize=(5.0 * nc, 3.6 * nr), squeeze=False)
    for i, pid in enumerate(pairs):
        ax = axes[i // nc][i % nc]
        x = np.arange(len(signals))
        ys = []
        for s in signals:
            r = next((r for r in ss if r['pair'] == pid and r['signal'] == s), None)
            ys.append(float(r['auc']) if r else 0)
        ax.bar(x, ys, 0.55, color=['#7f7f7f', '#4f81bd', '#c0504d'])
        ax.axhline(0.5, color='grey', ls=':', lw=1)
        ax.axhline(0.70, color='red', ls='--', lw=1, label='0.70 usability bound')
        ax.set_xticks(x)
        ax.set_xticklabels([snice[s] for s in signals], fontsize=9)
        ax.set_title(pnice(pid), fontsize=10)
        ax.set_ylim(0, 1)
        if i % nc == 0:
            ax.set_ylabel('AUC: retrieval vs recognition')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / 'fig6.pdf')
    plt.close(fig)


if __name__ == '__main__':
    fig4(); fig5(); fig6()
    print('figures written: fig4.pdf fig5.pdf fig6.pdf')
