"""Stage-6 figures.

fig1  core paired figure: correction change (rescue vs break) and confidence
      change (unchanged-answer increment with CI) side by side - no fitted
      curves, no crossing points here (task sec 7)
fig2  four-cell decomposition: mean confidence increment per cell with CIs
fig3  stratification: difference-in-differences forest (low minus high)
fig4  dose-response (q4b VCD alpha 0.5/1.0/2.0): increment + dECE vs alpha
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})

METHODS = ['vcd', 'm3id', 'dola', 'deco']
COLOR = {'vcd': '#c44e52', 'm3id': '#4c72b0', 'dola': '#55a868', 'deco': '#8172b2'}
TDIR = ROOT / 'outputs' / 'tables'
FDIR = ROOT / 'outputs' / 'figures'
FDIR.mkdir(exist_ok=True)


def read(name):
    import csv
    p = TDIR / f'stage6_{name}.csv'
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fig1():
    core = read('core')
    if not core:
        return
    models = sorted({r['model'] for r in core})
    strata = ['low_acc', 'high_acc']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4),
                             constrained_layout=True)
    w = 0.19
    for si, st in enumerate(strata):
        ax = axes[si]
        for mi, m in enumerate(METHODS):
            xs, ys_lo, ys_hi, ys, res, brk = [], [], [], [], [], []
            for gi, g in enumerate(models):
                r = next((r for r in core
                          if r['model'] == g and r['stratum'] == st and r['method'] == m), None)
                if r:
                    xs.append(gi + (mi - 2 + 0.5 * si) * w * 0.98)
                    ys.append(f(r['dconf_unchanged']))
                    ys_lo.append(f(r['dconf_un_lo']))
                    ys_hi.append(f(r['dconf_un_hi']))
                    res.append(f(r['rescue_rate']))
                    brk.append(f(r['break_rate']))
            if not xs:
                continue
            x = np.array(xs)
            ax.errorbar(x, ys, yerr=[np.array(ys) - np.array(ys_lo),
                                     np.array(ys_hi) - np.array(ys)],
                        fmt='o', ms=4, lw=1.2, color=COLOR[m], label=m)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_title('low-accuracy stratum' if st == 'low_acc' else 'high-accuracy stratum')
        ax.set_ylabel('confidence increment (answer unchanged)' if si == 0 else '')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.savefig(FDIR / 'stage6_fig1_core.pdf', bbox_inches='tight')
    plt.close(fig)

    # panel version b: rescue vs break (correction change), same grid
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    for si, st in enumerate(strata):
        ax = axes[si]
        for mi, m in enumerate(METHODS):
            xs, res, brk = [], [], []
            for gi, g in enumerate(models):
                r = next((r for r in core
                          if r['model'] == g and r['stratum'] == st and r['method'] == m), None)
                if r:
                    xs.append(gi + (mi - 1.5) * 0.16)
                    res.append(f(r['rescue_rate']))
                    brk.append(f(r['break_rate']))
            if xs:
                ax.plot(xs, res, 'o', ms=4, color=COLOR[m], label=f'{m} rescue')
                ax.plot(xs, brk, 'x', ms=4, mfc='none', color=COLOR[m], alpha=0.55)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_title('low-accuracy stratum' if st == 'low_acc' else 'high-accuracy stratum')
        ax.set_ylabel('rescue (o) / break (x) rate' if si == 0 else '')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.savefig(FDIR / 'stage6_fig1b_correction.pdf', bbox_inches='tight')
    plt.close(fig)


def fig2():
    fc = read('fourcell')
    if not fc:
        return
    models = sorted({r['model'] for r in fc})
    cells = ['always_right', 'always_wrong', 'corrected', 'broken']
    labels = ['always right', 'always wrong', 'corrected', 'broken']
    fig, axes = plt.subplots(len(models), 2, figsize=(8.4, 2.3 * len(models)),
                             constrained_layout=True, squeeze=False)
    for ri, g in enumerate(models):
        for si, st in enumerate(['low_acc', 'high_acc']):
            ax = axes[ri][si]
            for mi, m in enumerate(METHODS):
                r = next((r for r in fc if r['model'] == g
                          and r['stratum'] == st and r['method'] == m), None)
                if not r:
                    continue
                ys = [f(r[f'{c}_dconf']) for c in cells]
                lo = [f(r[f'{c}_lo']) for c in cells]
                hi = [f(r[f'{c}_hi']) for c in cells]
                x = np.arange(len(cells)) + (mi - 1.5) * 0.2
                ax.errorbar(x, ys, yerr=[np.array(ys) - np.array(lo),
                                         np.array(hi) - np.array(ys)],
                            fmt='o', ms=3.5, lw=1, color=COLOR[m], label=m)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, rotation=20, ha='right')
            if si == 0:
                ax.set_ylabel(f'{g}\nconf. incr.')
            if ri == 0:
                ax.set_title('low-accuracy stratum' if st == 'low_acc' else 'high-accuracy stratum')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False)
    fig.savefig(FDIR / 'stage6_fig2_fourcell.pdf', bbox_inches='tight')
    plt.close(fig)


def fig3():
    did = read('did')
    if not did:
        return
    models = sorted({r['model'] for r in did})
    fig, ax = plt.subplots(figsize=(5.6, 0.75 * len(models) + 1.2),
                           constrained_layout=True)
    y = 0
    yticks, ylabels = [], []
    for g in models:
        for m in METHODS:
            r = next((r for r in did if r['model'] == g and r['method'] == m), None)
            if not r:
                continue
            v, lo, hi = f(r['did']), f(r['did_lo']), f(r['did_hi'])
            col = COLOR[m]
            ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt='o', ms=4,
                        lw=1.2, color=col, label=m if g == models[0] else None)
            yticks.append(y)
            ylabels.append(f'{g} {m}')
            y += 1
        y += 0.5
    ax.axvline(0, color='k', lw=0.7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('DiD = dECE(low-acc) - dECE(high-acc)')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(FDIR / 'stage6_fig3_did.pdf', bbox_inches='tight')
    plt.close(fig)


def fig4():
    dose = [r for r in read('dose') if r.get('model') == 'q4b']
    cross = [r for r in read('dose') if r.get('model') == 'q4b_class_crossing']
    if not dose:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)
    for si, st in enumerate(['low_acc', 'high_acc']):
        ax = axes[si]
        for m_label, col, mk in [('conf. increment', '#c44e52', 'o'),
                                 ('dECE', '#4c72b0', 's')]:
            xs = [f(r['alpha']) for r in dose if r['stratum'] == st]
            ys = [f(r['dconf_all']) if mk == 'o' else f(r['dece'])
                  for r in dose if r['stratum'] == st]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], mk, ms=4, color=col,
                    label=m_label)
        ax.set_xlabel('VCD strength alpha')
        ax.set_title('low-accuracy stratum' if st == 'low_acc' else 'high-accuracy stratum')
    ax = axes[2]
    xs = [f(r['alpha']) for r in cross]
    ys = [f(r['crossing']) for r in cross]
    lo = [f(r['cross_lo']) for r in cross]
    hi = [f(r['cross_hi']) for r in cross]
    o = np.argsort(xs)
    ax.errorbar(np.array(xs)[o], np.array(ys)[o],
                yerr=[np.array(ys)[o] - np.array(lo)[o],
                      np.array(hi)[o] - np.array(ys)[o]],
                fmt='o', ms=4, lw=1.2, color='#55a868')
    ax.set_xlabel('α')
    ax.set_ylabel('crossing point (class-level)')
    fig.savefig(FDIR / 'stage6_fig4_dose.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    print('figures written:', sorted(p.name for p in FDIR.glob('stage6_*')))
