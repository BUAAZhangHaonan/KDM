"""Paper figures from the analysis tables.

figure1: certain-error rate per method x tier (naming) + hallucination on negatives
         (existence), with paired-delta CIs — effects across frequency tiers.
figure2: low-tier performance before/after remedy rule and vs ideal rule.
figure3: judgment-2 signal discrimination (ROC-style AUC summary).
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

TAB = ROOT / 'outputs' / 'tables'
FIG = ROOT / 'outputs' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
METHODS = ['direct', 'vcd', 'mib', 'lcd']
LBL = {'direct': 'Direct', 'vcd': 'VCD', 'mib': 'MIB', 'lcd': 'LCD'}
C_HIGH, C_LOW = '#3b6fb6', '#c25b4e'


def read_csv(name):
    with open(TAB / name) as f:
        return list(csv.DictReader(f))


def fig1():
    main = read_csv('main.csv')
    eff = read_csv('effects.csv')
    models = sorted({r['model'] for r in main})
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 7.2), squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0][j]
        w = 0.38
        xs = np.arange(len(METHODS))
        for k, (tier, c) in enumerate([('high', C_HIGH), ('low', C_LOW)]):
            vals = []
            for m in METHODS:
                row = next((r for r in main if r['model'] == model and r['task'] == 'naming'
                            and r['method'] == m and r['tier'] == tier), None)
                vals.append(float(row['certain_error_rate']) if row else np.nan)
            ax.bar(xs + (k - 0.5) * w, vals, w, color=c, label=f'{tier}-frequency')
        # CI whiskers for delta vs direct on low tier
        for i, m in enumerate(METHODS[1:], start=1):
            er = next((r for r in eff if r['model'] == model and r['task'] == 'naming'
                       and r['method'] == m and r['tier'] == 'low'), None)
            if er:
                base = next(r for r in main if r['model'] == model and r['task'] == 'naming'
                            and r['method'] == m and r['tier'] == 'low')
                y = float(base['certain_error_rate'])
                lo, hi = float(er['err_lo']), float(er['err_hi'])
                ax.plot([i, i], [y + lo, y + hi], color='k', lw=1.2)
                ax.plot([i - 0.08, i + 0.08], [y + hi] * 2, color='k', lw=1.2)
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in METHODS])
        ax.set_ylabel('certain-error rate' if j == 0 else '')
        ax.set_title(f'{model}: object naming')
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, 1)

        ax = axes[1][j]
        w = 0.38
        for k, (tier, c) in enumerate([('high', C_HIGH), ('low', C_LOW)]):
            vals = []
            for m in METHODS:
                row = next((r for r in main if r['model'] == model and r['task'] == 'existence'
                            and r['method'] == m and r['tier'] == f'neg_{tier}'), None)
                vals.append(float(row['hallucination_on_neg']) if row else np.nan)
            ax.bar(xs + (k - 0.5) * w, vals, w, color=c, label=f'{tier}-frequency')
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in METHODS])
        ax.set_ylabel('"Yes" rate on negatives' if j == 0 else '')
        ax.set_title(f'{model}: existence (negatives)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / 'figure1.pdf')
    plt.close(fig)
    print('figure1 done')


def fig2():
    rem = read_csv('remedy.csv')
    main = read_csv('main.csv')
    models = sorted({r['model'] for r in rem})
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6), squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0][j]
        d_low = next(r for r in main if r['model'] == model and r['method'] == 'direct'
                     and r['task'] == 'naming' and r['tier'] == 'low')
        bars = [('direct', float(d_low['certain_error_rate']), '#666666'),
                ('remedy', float(next(r for r in rem if r['model'] == model and r['rule'] == 'remedy_high_tuned'
                                      and r['tier'] == 'low')['certain_error_rate']), C_LOW),
                ('median', float(next(r for r in rem if r['model'] == model and r['rule'] == 'remedy_median'
                                      and r['tier'] == 'low')['certain_error_rate']), '#d99a91'),
                ('ideal', float(next(r for r in rem if r['model'] == model and r['rule'] == 'ideal'
                                     and r['tier'] == 'low')['certain_error_rate']), '#4c9f70')]
        xs = np.arange(len(bars))
        ax.bar(xs, [b[1] for b in bars], 0.55, color=[b[2] for b in bars])
        for x, (_, v, _) in zip(xs, bars):
            ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        # 30% reduction target line
        target = float(d_low['certain_error_rate']) * 0.7
        ax.axhline(target, ls='--', c='k', lw=1)
        ax.text(len(bars) - 0.4, target + 0.02, '30% reduction', fontsize=8, ha='right')
        ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars])
        ax.set_ylabel('low-freq certain-error rate' if j == 0 else '')
        ax.set_title(model)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / 'figure2.pdf')
    plt.close(fig)
    print('figure2 done')


def fig3():
    j2 = read_csv('judgment2_auc.csv')
    models = sorted({r['model'] for r in j2})
    sigs = ['entropy', 'jsd', 'logistic_combo']
    sig_lbl = {'entropy': 'entropy', 'jsd': 'JSD(blank)', 'logistic_combo': 'combination'}
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4), squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0][j]
        methods = METHODS[1:]
        xs = np.arange(len(methods))
        w = 0.26
        for k, sig in enumerate(sigs):
            vals = [float(next((r for r in j2 if r['model'] == model and r['method'] == m
                                and r['tier'] == 'all' and r['signal'] == sig), {'auc': 'nan'})['auc'])
                    for m in methods]
            ax.bar(xs + (k - 1) * w, vals, w, label=sig_lbl[sig])
        ax.axhline(0.5, c='k', lw=0.8, ls=':')
        ax.axhline(0.7, c='#4c9f70', lw=1, ls='--')
        ax.text(len(methods) - 0.45, 0.705, '0.70 target', fontsize=8, color='#4c9f70', ha='right')
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in methods])
        ax.set_ylabel('AUC (harmed vs helped)' if j == 0 else '')
        ax.set_title(model)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'figure3.pdf')
    plt.close(fig)
    print('figure3 done')


if __name__ == '__main__':
    fig1(); fig2(); fig3()
