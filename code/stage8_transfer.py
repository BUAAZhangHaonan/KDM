"""Stage-8 part 2: dogs magnitude-consistency test (preregistered caliber A)
and the M3ID gate/rescue recheck. Zero new inference: reads stage6_core.csv,
stage7_dogspredict.csv, stage7_m3idgate_q4b/q9b.csv.

Caliber A (PREREGISTER_STAGE8 sec 4): OLS of dece on pre-intervention accuracy
over the 32 faithful food101 cells; per-dogs-cell 95% prediction band; hit =
observed dogs dece inside the band; per-method sensitivity fits; extrapolation
flag; the word "prediction" is NOT used for the claim (consistency test only).
Outputs: outputs/tables/stage8_{dogs_transfer, m3id_gate_recheck}.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def ols_pi(xs, ys, x_star):
    """OLS fit + 95% prediction interval half-width at x_star."""
    n = len(xs)
    b1, b0 = np.polyfit(xs, ys, 1)
    yhat = np.array(b0 + b1 * np.array(xs))
    s = float(np.sqrt(((np.array(ys) - yhat) ** 2).sum() / (n - 2)))
    xbar = float(np.mean(xs))
    Sxx = float(((np.array(xs) - xbar) ** 2).sum())
    t = float(stats.t.ppf(0.975, n - 2))
    x_star = np.atleast_1d(np.asarray(x_star, float))
    hw = t * s * np.sqrt(1 + 1 / n + (x_star - xbar) ** 2 / Sxx)
    ystar = b0 + b1 * x_star
    r2 = 1 - ((np.array(ys) - yhat) ** 2).sum() / \
        ((np.array(ys) - np.mean(ys)) ** 2).sum()
    return {'b0': b0, 'b1': b1, 'r2': r2, 's': s, 'n': n,
            'pred': ystar, 'hw': hw,
            'lo': ystar - hw, 'hi': ystar + hw}


def main():
    core = list(csv.DictReader(open(ROOT / 'outputs/tables/stage6_core.csv')))
    fx = np.array([float(r['acc_direct']) for r in core])
    fy = np.array([float(r['dece']) for r in core])

    dogs = list(csv.DictReader(open(ROOT / 'outputs/tables/stage7_dogspredict.csv')))
    rows = []
    # pooled primary fit
    fit = ols_pi(fx, fy, [float(r['acc_d']) for r in dogs])
    for i, r in enumerate(dogs):
        acc = float(r['acc_d'])
        obs = float(r['dece'])
        rows.append({
            'fit': 'pooled_32cells', 'model': r['model'], 'stratum': r['stratum'],
            'method': r['method'], 'acc_pre_dogs': acc, 'dece_obs': obs,
            'pred': float(fit['pred'][i]), 'band_lo': float(fit['lo'][i]),
            'band_hi': float(fit['hi'][i]),
            'hit': bool(fit['lo'][i] <= obs <= fit['hi'][i]),
            'extrapolated': bool(acc < fx.min() or acc > fx.max())})
    pooled = {'fit': 'pooled_32cells', 'b0': fit['b0'], 'b1': fit['b1'],
              'r2': fit['r2'], 's': fit['s'], 'n': fit['n'],
              'x_min': float(fx.min()), 'x_max': float(fx.max())}
    # per-method sensitivity
    for meth in ('vcd', 'm3id', 'dola', 'deco'):
        sub = [r for r in core if r['method'] == meth]
        mx = np.array([float(r['acc_direct']) for r in sub])
        my = np.array([float(r['dece']) for r in sub])
        tgt = [r for r in dogs if r['method'] == meth]
        fm = ols_pi(mx, my, [float(r['acc_d']) for r in tgt])
        for i, r in enumerate(tgt):
            acc = float(r['acc_d'])
            obs = float(r['dece'])
            j = next(k for k, rr in enumerate(rows)
                     if rr['fit'] == 'pooled_32cells' and rr['model'] == r['model']
                     and rr['stratum'] == r['stratum'] and rr['method'] == meth)
            rows[j].update({'fit_per_method_pred': float(fm['pred'][i]),
                            'fit_per_method_lo': float(fm['lo'][i]),
                            'fit_per_method_hi': float(fm['hi'][i]),
                            'fit_per_method_hit': bool(fm['lo'][i] <= obs <= fm['hi'][i])})
        pooled[f'b0_{meth}'], pooled[f'b1_{meth}'] = fm['b0'], fm['b1']
        pooled[f'r2_{meth}'] = fm['r2']

    outdir = ROOT / 'outputs' / 'tables'
    keys = sorted({k for r in rows for k in r})
    with open(outdir / 'stage8_dogs_transfer.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    hits = sum(1 for r in rows if r['hit'])
    hits_pm = sum(1 for r in rows if r.get('fit_per_method_hit'))
    n_ex = sum(1 for r in rows if r['extrapolated'])
    print(f'dogs_transfer: pooled band hits {hits}/16 (extrapolated {n_ex}); '
          f'per-method band hits {hits_pm}/16')
    print('fit:', {k: round(v, 4) for k, v in pooled.items() if isinstance(v, float)})

    # ---- M3ID gate + rescue recheck -----------------------------------------
    gate = {}
    for q in ('q4b', 'q9b'):
        for r in csv.DictReader(open(ROOT / 'outputs' / 'tables' / f'stage7_m3idgate_{q}.csv')):
            gate[(r['model'], r['stratum'])] = r
    m3 = [r for r in core if r['method'] == 'm3id']
    m3rows = []
    low_all = [float(r['rescue_rate']) for r in core if r['stratum'] == 'low_acc']
    lo, hi = min(low_all), max(low_all)
    for r in m3:
        key = (r['model'], r['stratum'])
        rr = float(r['rescue_rate'])
        m3rows.append({
            'model': r['model'], 'stratum': r['stratum'],
            'gate_closed_ratio': gate[key]['gate_closed_ratio'] if key in gate else '',
            'rescue_rate': rr,
            'in_low_acc_rescue_range': '',
            'acc_direct': float(r['acc_direct']),
            'dconf_unchanged': float(r['dconf_unchanged']),
            'dece': float(r['dece']),
            'wording_condition_holds':
                bool(lo <= rr <= lo + (hi - lo) / 2) if r['stratum'] == 'low_acc' else ''})
    with open(outdir / 'stage8_m3id_gate_recheck.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(m3rows[0]))
        w.writeheader()
        w.writerows(m3rows)
    print('m3id_gate_recheck:',
          [(r['model'], r['stratum'], round(r['rescue_rate'], 3),
            r['gate_closed_ratio'][:5] if r['gate_closed_ratio'] else '',
            r['wording_condition_holds'])
           for r in m3rows if r['stratum'] == 'low_acc'])

    json.dump({'dogs': {'pooled_hits': hits, 'per_method_hits': hits_pm,
                        'n': len(rows), 'n_extrapolated': n_ex, 'fit': pooled},
               'm3id_low_rescue_range': [lo, hi]},
              open(outdir / 'stage8_transfer_summary.json', 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main()
