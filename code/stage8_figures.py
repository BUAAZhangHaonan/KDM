"""Stage-8 figure: residual-regression presentation (replaces the cell-count
scatter as the paper's second figure per PREREGISTER_STAGE8 sec 6).

Panel A: per-cell within-cell residual gap gamma_m (mean r among wrong minus
         among right answers) vs pre-intervention accuracy A_m, 32 faithful
         cells; the three meta-regression lines (FE inverse-variance, DL
         random-effects weights, unweighted OLS) over the judged-cell range -
         the weighting sensitivity is the honest headline.
Panel B: within-cell demeaned residuals by answer correctness (the pooled
         gamma shift with all clustering CIs annotated).
Data: outputs/tables/stage8_{residual_regression, meta, metareg}.csv.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import stage8_analysis as sa  # noqa: E402

os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
TDIR = ROOT / 'outputs' / 'tables'
FDIR = ROOT / 'outputs' / 'figures'
COLOR = {'vcd': '#c44e52', 'm3id': '#4c72b0', 'dola': '#55a868', 'deco': '#8172b2'}
LBL = {'vcd': 'VCD', 'm3id': 'M3ID', 'dola': 'DoLa', 'deco': 'DeCo'}


def main():
    summary = json.load(open(TDIR / 'stage8_summary.json'))
    cells = [r for r in csv.DictReader(open(TDIR / 'stage8_residual_regression.csv'))
             if r['row_type'] == 'per_cell']
    meta = [r for r in csv.DictReader(open(TDIR / 'stage8_meta.csv'))
            if r['model'] != 'POOLED_DL']

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.6, 3.5),
                                   constrained_layout=True)

    # ---- panel A: gamma_m vs A_m + three metareg lines ----------------------
    for meth in COLOR:
        xs = [float(r['A_pre']) for r in cells if r['method'] == meth]
        ys = [float(r['beta']) for r in cells if r['method'] == meth]
        judged = [r['stage7_verdict'] != 'insufficient_n'
                  for r in cells if r['method'] == meth]
        axa.scatter([x for x, j in zip(xs, judged) if j],
                    [y for y, j in zip(ys, judged) if j], s=26,
                    color=COLOR[meth], label=LBL[meth], zorder=3)
        axa.scatter([x for x, j in zip(xs, judged) if not j],
                    [y for y, j in zip(ys, judged) if not j], s=26,
                    color=COLOR[meth], alpha=0.35, marker='x', zorder=3)
    axa.axhline(0, color='0.4', lw=0.8, ls=':')
    mr = json.load(open(TDIR / 'stage8_summary.json'))['metareg']
    A = np.array([float(r['A_pre']) for r in meta])
    xs = np.linspace(A.min(), A.max(), 50)
    fe = [r for r in csv.DictReader(open(TDIR / 'stage8_metareg.csv'))
          if r['weights'].startswith('inverse')][0]
    re_ = [r for r in csv.DictReader(open(TDIR / 'stage8_metareg.csv'))
           if r['weights'].startswith('random')][0]
    axa.plot(xs, float(fe['intercept']) + float(fe['slope']) * xs,
             color='0.2', lw=1.2, ls='-',
             label='metareg FE (1/se$^2$)')
    axa.plot(xs, float(re_['intercept']) + float(re_['slope']) * xs,
             color='0.5', lw=1.2, ls='--',
             label='metareg RE (DL)')
    un_b1 = mr['slope_unweighted']
    un_b0 = float(np.mean([float(r['d_m']) for r in meta])) - un_b1 * float(np.mean(A))
    axa.plot(xs, un_b0 + un_b1 * xs, color='0.7', lw=1.2, ls=':',
             label='unweighted OLS')
    axa.set_xlabel('pre-intervention accuracy of the cell population $A_m$')
    axa.set_ylabel('within-cell residual gap $\\gamma_m$ (wrong $-$ right)')
    axa.set_title('(a) per-cell residual gap vs population accuracy\n'
                  '(x = stage-7 insufficient cells)', fontsize=9)
    axa.legend(fontsize=7, loc='upper right', frameon=False)

    # ---- panel B: within-cell demeaned residuals by correctness -------------
    rows = sa.build_samples()
    r_all = np.array([r['r'] for r in rows])
    cell_code = {}
    dem = np.zeros(len(rows))
    for c in sa.CELLS:
        idx = np.array([i for i, r in enumerate(rows) if r['cell'] == c])
        if len(idx):
            dem[idx] = r_all[idx] - r_all[idx].mean()
    wv = np.array([r['w'] for r in rows])
    parts = [dem[wv == v] for v in (0, 1)]
    bp = axb.boxplot(parts, positions=[1, 2], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops={'color': '0.15'})
    axb.set_xticks([1, 2])
    axb.set_xticklabels(['correct\n(w=0)', 'wrong\n(w=1)'])
    for patch in bp['boxes']:
        patch.set_facecolor('#dde6f0')
    for i, p in enumerate(parts):
        rngj = np.random.RandomState(7 + i)
        xj = rngj.uniform(-0.14, 0.14, len(p))
        axb.scatter(1 + i + xj, p, s=1.4, alpha=0.12, color='#4c72b0', zorder=1)
    g = summary['gamma']['beta']
    axb.annotate('', xy=(2, dem[wv == 1].mean()), xytext=(2, dem[wv == 0].mean()),
                 arrowprops=dict(arrowstyle='->', color='#c44e52', lw=1.4))
    axb.text(2.06, (dem[wv == 1].mean() + dem[wv == 0].mean()) / 2,
             f'pooled $\\gamma$ = {g:+.4f}\n95% CI '
             f'[{summary["gamma"]["z_ci"][0]:+.4f}, {summary["gamma"]["z_ci"][1]:+.4f}]'
             f' (model-cl.)\nall 6 CI variants contain 0',
             fontsize=7.5, va='center')
    axb.axhline(0, color='0.4', lw=0.8, ls=':')
    axb.set_ylabel('within-cell demeaned residual $r_i - \\bar{r}_{m(i)}$')
    axb.set_title('(b) residual distribution by answer correctness\n'
                  '(cell fixed effects removed)', fontsize=9)

    out = FDIR / 'stage8_fig_residual.pdf'
    fig.savefig(out)
    print('saved', out)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'code'))
    main()
