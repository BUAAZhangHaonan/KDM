"""Stage-5 figure: fig9 (ΔECE vs pre-intervention accuracy / calibration gap).

Reads outputs/tables/baserate.csv + crossing.csv (written by stage5_baserate.py).
Left: ΔECE vs pre accuracy, class-accuracy display bins (solid) and mixture
populations (dashed), per model x method; OLS crossing points marked.
Right: same quantities against the pre calibration gap.
"""
import sys, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / 'outputs' / 'tables'
F = ROOT / 'outputs' / 'figures'
NICE = {'q4b': 'Qwen3.5-4B', 'q9b': 'Qwen3.5-9B',
        'vcd': 'VCD', 'mib': 'MIB', 'lcd': 'LCD'}
MCOLOR = {'vcd': '#c0504d', 'mib': '#4f81bd', 'lcd': '#9bbb59'}
MSTYLE = {'vcd': 'o', 'mib': 's', 'lcd': '^'}
DOMLS = {'food101': '-', 'dogs': '-'}


def rows(name):
    with open(T / name) as f:
        return list(csv.DictReader(f))


def fig9():
    base = rows('baserate.csv')
    cross = rows('crossing.csv')
    pairs = []
    for r in base:
        k = (r['model'], r['domain'])
        if k not in pairs:
            pairs.append(k)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, reg, xlabel in ((axes[0], 'acc', 'pre-intervention accuracy (direct)'),
                            (axes[1], 'gap', 'pre-intervention calibration gap (confidence − accuracy)')):
        ax.axhline(0, color='0.4', lw=0.8, zorder=1)
        for (model, domain) in pairs:
            for m in ('vcd', 'mib', 'lcd'):
                bins = sorted([r for r in base if r['population'] == 'bin'
                               and r['model'] == model and r['domain'] == domain
                               and r['method'] == m],
                              key=lambda r: float(r['acc_direct']))
                if not bins:
                    continue
                xk = 'acc_direct' if reg == 'acc' else 'gap_direct'
                x = [float(r[xk]) for r in bins]
                y = [float(r['d_ece']) for r in bins]
                c = MCOLOR[m]
                lab = f'{NICE[model]} {NICE[m]}' if domain == 'food101' else f'{NICE[model]} {NICE[m]} (dogs)'
                ax.plot(x, y, DOMLS[domain], marker=MSTYLE[m], ms=4.5, lw=1.2,
                        color=c, alpha=0.85, label=lab, zorder=3)
                mixes = sorted([r for r in base if r['population'] == 'mixture'
                                and r['model'] == model and r['domain'] == domain
                                and r['method'] == m],
                               key=lambda r: float(r['acc_direct']))
                if mixes:
                    xm = [float(r[xk]) for r in mixes]
                    ym = [float(r['d_ece']) for r in mixes]
                    ax.plot(xm, ym, ':', lw=1.1, color=c, alpha=0.55, zorder=2)
        # crossing markers on the acc panel (per model, method-averaged x0)
        if reg == 'acc':
            for (model, domain) in pairs:
                x0s = [float(r['x0']) for r in cross
                       if r['model'] == model and r['domain'] == domain
                       and r['regressor'] == 'acc' and r['x0'] not in ('', None)]
                if x0s:
                    ax.axvline(float(np.mean(x0s)), color='0.3', lw=0.9, ls='--', alpha=0.6, zorder=1)
                    ax.annotate(f'{NICE[model]} crossing\n' + r'$\approx$' + f'{np.mean(x0s):.2f}',
                                (float(np.mean(x0s)), ax.get_ylim()[1]), fontsize=7,
                                ha='center', va='top', color='0.25',
                                xytext=(float(np.mean(x0s)), ax.get_ylim()[1] - 0.02 *
                                        (ax.get_ylim()[1] - ax.get_ylim()[0])))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\Delta$ECE (method $-$ direct)')
    axes[0].legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9)
    axes[1].legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9)
    fig.suptitle('Calibration change of decoding interventions is predicted by the population base rate '
                 '(solid: class-accuracy bins; dotted: mixed populations; dashed vertical: OLS crossing)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(F / 'fig9.pdf')
    print('wrote fig9.pdf')


if __name__ == '__main__':
    fig9()
