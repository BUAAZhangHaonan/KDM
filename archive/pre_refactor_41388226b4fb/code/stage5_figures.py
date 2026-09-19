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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    handles = []
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
                line, = ax.plot(x, y, DOMLS[domain], marker=MSTYLE[m], ms=4.5, lw=1.2,
                                color=c, alpha=0.85, label=lab, zorder=3)
                if reg == 'acc':
                    handles.append(line)
                mixes = sorted([r for r in base if r['population'] == 'mixture'
                                and r['model'] == model and r['domain'] == domain
                                and r['method'] == m],
                               key=lambda r: float(r['acc_direct']))
                if mixes:
                    xm = [float(r[xk]) for r in mixes]
                    ym = [float(r['d_ece']) for r in mixes]
                    ax.plot(xm, ym, ':', lw=1.1, color=c, alpha=0.55, zorder=2)
        # crossing markers on the acc panel: VCD cells whose direction is supported
        if reg == 'acc':
            ylim = ax.get_ylim()
            for (model, domain) in pairs:
                x0s = [float(r['x0']) for r in cross
                       if r['model'] == model and r['domain'] == domain
                       and r['regressor'] == 'acc' and r['method'] == 'vcd'
                       and r['direction_supported'] == 'True' and r['x0'] not in ('', None)]
                if x0s:
                    xv = float(np.mean(x0s))
                    ax.axvline(xv, color='0.3', lw=0.9, ls='--', alpha=0.6, zorder=1)
                    ax.annotate(f'{NICE[model].split("-")[-1]} {domain}: ' + r'$x_0\approx$' + f'{np.mean(x0s):.2f}',
                                (xv, ylim[1]), fontsize=7.5, ha='center', va='top', color='0.25',
                                xytext=(xv, ylim[1] - 0.03 * (ylim[1] - ylim[0])))
                    ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\Delta$ECE (method $-$ direct)')
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Calibration change of decoding interventions is predicted by the population base rate\n'
                 '(solid: class-accuracy bins; dotted: mixed populations; dashed vertical: OLS crossing)',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(F / 'fig9.pdf', bbox_inches='tight')
    print('wrote fig9.pdf')


if __name__ == '__main__':
    fig9()
