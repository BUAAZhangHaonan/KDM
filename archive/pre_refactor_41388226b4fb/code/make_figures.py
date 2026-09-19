"""v2 paper figures.
figure1: per-method effects on the two strata (naming): certain-error rate with
         CI whiskers + wrong-answer confidence.
figure2: wrong-answer confidence and calibration across configs per stratum.
figure3: judgment-2 signal AUC summary.
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
C_KN, C_DEF = '#3b6fb6', '#c25b4e'


def read_csv(name):
    with open(TAB / name) as f:
        return list(csv.DictReader(f))


def _num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fig1():
    main = read_csv('main.csv')
    eff = read_csv('effects.csv')
    models = sorted({r['model'] for r in main})
    fig, axes = plt.subplots(2, len(models), figsize=(4.4 * len(models), 7.0), squeeze=False)
    xs = np.arange(len(METHODS))
    w = 0.38
    for j, model in enumerate(models):
        ax = axes[0][j]
        for k, (st, c) in enumerate([('known', C_KN), ('deficient', C_DEF)]):
            vals = [_num(next((r['certain_error_rate'] for r in main if r['model'] == model and r['task'] == 'naming'
                               and r['method'] == m and r['stratum'] == st), '')) for m in METHODS]
            ax.bar(xs + (k - 0.5) * w, vals, w, color=c, label=f'{st}')
        for i, m in enumerate(METHODS[1:], start=1):
            er = next((r for r in eff if r['model'] == model and r['task'] == 'naming'
                       and r['method'] == m and r['stratum'] == 'deficient'), None)
            if er and er.get('err_lo'):
                base = _num(next(r['certain_error_rate'] for r in main if r['model'] == model and r['task'] == 'naming'
                                 and r['method'] == m and r['stratum'] == 'deficient'))
                lo, hi = _num(er['err_lo']), _num(er['err_hi'])
                ax.plot([i, i], [base + lo, base + hi], color='k', lw=1.2)
                ax.plot([i - .08, i + .08], [base + hi] * 2, color='k', lw=1.2)
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in METHODS])
        ax.set_ylabel('certain-error rate' if j == 0 else '')
        ax.set_title(f'{model}: object naming')
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylim(0, 1)
        ax = axes[1][j]
        for k, (st, c) in enumerate([('known', C_KN), ('deficient', C_DEF)]):
            vals = [_num(next((r['wrong_answer_conf'] for r in main if r['model'] == model and r['task'] == 'naming'
                               and r['method'] == m and r['stratum'] == st), '')) for m in METHODS]
            ax.bar(xs + (k - 0.5) * w, vals, w, color=c)
        for i, m in enumerate(METHODS[1:], start=1):
            er = next((r for r in eff if r['model'] == model and r['task'] == 'naming'
                       and r['method'] == m and r['stratum'] == 'deficient'), None)
            if er and er.get('conf_lo') != '' and er['conf_lo'] != '':
                base = _num(next(r['wrong_answer_conf'] for r in main if r['model'] == model and r['task'] == 'naming'
                                 and r['method'] == m and r['stratum'] == 'deficient'))
                lo, hi = _num(er['conf_lo']), _num(er['conf_hi'])
                dc = _num(er.get('d_wrong_conf', ''))
                ax.plot([i, i], [base - dc + lo, base - dc + hi], color='k', lw=1.2)
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in METHODS])
        ax.set_ylabel('wrong-answer confidence' if j == 0 else '')
        ax.set_title(f'{model}: confidence when wrong')
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / 'figure1.pdf')
    plt.close(fig)
    print('figure1 done')


def fig2():
    main = read_csv('main.csv')
    models = sorted({r['model'] for r in main})
    fig, axes = plt.subplots(2, len(models), figsize=(4.4 * len(models), 6.6), squeeze=False)
    xs = np.arange(len(METHODS))
    w = 0.38
    for j, model in enumerate(models):
        for row_i, (metric, ttl) in enumerate([('wrong_answer_conf', 'mean wrong-answer confidence'),
                                               ('calibration_ece', 'calibration error (ECE)')]):
            ax = axes[row_i][j]
            for k, (st, c) in enumerate([('known', C_KN), ('deficient', C_DEF)]):
                vals = [_num(next((r[metric] for r in main if r['model'] == model and r['task'] == 'naming'
                                   and r['method'] == m and r['stratum'] == st), '')) for m in METHODS]
                ax.bar(xs + (k - 0.5) * w, vals, w, color=c, label=f'{st}' if row_i == 0 and j == 0 else None)
            ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in METHODS])
            ax.set_ylabel(ttl if j == 0 else '')
            ax.set_title(f'{model}')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(0, 1 if row_i == 0 else 0.8)
    fig.tight_layout()
    fig.savefig(FIG / 'figure2.pdf')
    plt.close(fig)
    print('figure2 done')


def fig3():
    j2 = read_csv('judgment2_auc.csv')
    models = sorted({r['model'] for r in j2})
    sigs = ['entropy', 'jsd', 'logistic_combo']
    lbl = {'entropy': 'entropy', 'jsd': 'JSD(blank)', 'logistic_combo': 'combination'}
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4), squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0][j]
        ms = METHODS[1:]
        xs = np.arange(len(ms))
        w = 0.26
        for k, s in enumerate(sigs):
            vals = [_num(next((r['auc'] for r in j2 if r['model'] == model and r['method'] == m
                               and r['stratum'] == 'all' and r['signal'] == s), '')) for m in ms]
            ax.bar(xs + (k - 1) * w, vals, w, label=lbl[s])
        ax.axhline(0.5, c='k', lw=0.8, ls=':')
        ax.axhline(0.7, c='#4c9f70', lw=1, ls='--')
        ax.text(len(ms) - 0.45, 0.705, '0.70 target', fontsize=8, color='#4c9f70', ha='right')
        ax.set_xticks(xs); ax.set_xticklabels([LBL[m] for m in ms])
        ax.set_ylabel('AUC (harmed vs helped)' if j == 0 else '')
        ax.set_title(model); ax.set_ylim(0, 1)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'figure3.pdf')
    plt.close(fig)
    print('figure3 done')


if __name__ == '__main__':
    fig1(); fig2(); fig3()
