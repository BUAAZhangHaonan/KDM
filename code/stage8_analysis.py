"""Stage-8 part 1: branch-decision redo on existing caches, zero new inference.

Implements PREREGISTER_STAGE8 sections 1-3:
  1. per-sample residual regression  r_i = a_m + gamma*w_i (+ theta*w_i*A_m)
     on unchanged-answer eval-half samples of all 32 faithful cells;
     cluster-robust SE by model (primary) and by cell; model-level cluster
     bootstrap; per-cell gaps gamma_m with n_wrong/n_right (coverage of the
     9 stage-7 insufficient cells).
  2. per-cell d_m = delta_m - delta_ctrl,m with paired-bootstrap SE (B=2000,
     seed 20260917), cross-checked against stage7_cellconsistency.csv CIs;
     DerSimonian-Laird random-effects meta; weighted meta-regression on the
     pre-intervention accuracy A_m; TOST-style equivalence at margin 0.02
     (90% bootstrap CI inside (-0.02, +0.02)).
  3. frozen 3-way branch rule + archival sign test of the stage-7 11:12 count.
Outputs: outputs/tables/stage8_{residual_regression, meta, metareg,
equivalence}.csv + stage8_summary.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from stage6_analysis import load, norm_text, _temp_conf  # noqa: E402

MODELS = ['q4b', 'q9b', 'llava16', 'internvl4b']
METHODS = ['vcd', 'm3id', 'dola', 'deco']
B, SEED = 2000, 20260917
EQ_MARGIN = 0.02

CELLS = [(m, s, k) for m in MODELS for s in ('low_acc', 'high_acc')
         for k in METHODS]


def build_samples():
    """Per-sample rows (cell, model, w, r, im, ic) under the frozen scope."""
    import csv as _csv
    T_fit = {(r['model'], r['stratum'], r['method']): float(r['T_fit'])
             for r in _csv.DictReader(open(ROOT / 'outputs/tables/stage6_sharpening.csv'))}
    acc = {(r['model'], r['stratum'], r['method']): float(r['acc_direct'])
           for r in _csv.DictReader(open(ROOT / 'outputs/tables/stage6_core.csv'))}
    rows = []
    for model in MODELS:
        ev = load(model, 'eval')
        fd = np.load(ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_eval.npz')
        files = sorted({f for f, m in ev if m == 'direct'})
        for stratum in ('low_acc', 'high_acc'):
            for method in METHODS:
                cell = (model, stratum, method)
                for f in files:
                    d, mm = ev.get((f, 'direct')), ev.get((f, method))
                    if d is None or mm is None or f not in fd:
                        continue
                    if d['stratum'] != stratum:
                        continue
                    if d.get('answer_maxp') is None or mm.get('answer_maxp') is None:
                        continue
                    if norm_text(mm['text']) != norm_text(d['text']) or \
                            mm['tokens'][0] != d['tokens'][0]:
                        continue
                    v = np.asarray(fd[f], dtype=np.float64)
                    tok = mm['tokens'][0]
                    p0 = float(v[tok])
                    im = mm['answer_maxp'] - p0
                    ic = _temp_conf(v, tok, T_fit[cell]) - p0
                    rows.append({'cell': cell, 'model': model, 'file': f,
                                 'cls': d['class'],
                                 'w': int(d['outcome'] != 'correct'),
                                 'r': im - ic, 'im': im, 'ic': ic,
                                 'A': acc[cell]})
    return rows


def ols_cluster(y, X, clusters, names):
    """OLS with cluster-robust (sandwich, small-sample corrected) covariance."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    G = len(set(clusters))
    N, K = X.shape
    c = G / (G - 1) * (N - 1) / (N - K)
    meat = np.zeros((K, K))
    for g in set(clusters):
        idx = clusters == g
        Xg, ug = X[idx], u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    V = c * XtX_inv @ meat @ XtX_inv
    out = {}
    for j, nm in enumerate(names):
        se = float(np.sqrt(V[j, j]))
        out[nm] = {'beta': float(beta[j]), 'se': se,
                   'z_ci': (float(beta[j] - 1.959964 * se),
                            float(beta[j] + 1.959964 * se))}
    return out, beta


def cell_dummies(rows, extra):
    """Design [intercept-less 32 dummies | extra columns] + cluster vectors."""
    idx = {c: i for i, c in enumerate(CELLS)}
    X = np.zeros((len(rows), len(CELLS) + len(extra)))
    cl_model = np.empty(len(rows), dtype=object)
    cl_cell = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        X[i, idx[r['cell']]] = 1.0
        for j, col in enumerate(extra):
            X[i, len(CELLS) + j] = r[col[0]] if isinstance(col, tuple) else r[col]
        cl_model[i] = r['model']
        cl_cell[i] = '%s|%s|%s' % r['cell']
    return X, cl_model, cl_cell


def main():
    rows = build_samples()
    y = np.array([r['r'] for r in rows])
    w = np.array([r['w'] for r in rows], float)
    print(f'N samples = {len(rows)}, wrong = {int(w.sum())}')

    # ---- 1. main regression: r ~ cell FE + w ------------------------------
    X1, cm, cc = cell_dummies(rows, ['w'])
    cl_class = np.array([r['cls'] for r in rows], dtype=object)
    cl_method = np.array([r['cell'][2] for r in rows], dtype=object)
    names1 = [f'cell:{m}|{s}|{k}' for m, s, k in CELLS] + ['gamma']
    res_m, beta1 = ols_cluster(y, X1, cm, names1)
    res_c, _ = ols_cluster(y, X1, cc, names1)
    res_class, _ = ols_cluster(y, X1, cl_class, names1)
    res_meth, _ = ols_cluster(y, X1, cl_method, names1)
    g = res_m['gamma']
    t3 = stats.t.ppf(0.975, 3)
    g_t3ci = (g['beta'] - t3 * g['se'], g['beta'] + t3 * g['se'])

    # model-level cluster bootstrap (resample the 4 models with replacement)
    rng = np.random.RandomState(SEED)
    by_model = {m: [i for i, r in enumerate(rows) if r['model'] == m]
                for m in MODELS}
    boot_g = []
    for _ in range(B):
        pick = [MODELS[j] for j in rng.randint(0, len(MODELS), len(MODELS))]
        idx = np.concatenate([by_model[m] for m in pick])
        Xb = np.zeros((len(idx), len(CELLS) + 1))
        idxc = {c: i for i, c in enumerate(CELLS)}
        for rj, i in enumerate(idx):
            Xb[rj, idxc[rows[i]['cell']]] = 1.0
            Xb[rj, -1] = rows[i]['w']
        bb, *_ = np.linalg.lstsq(Xb, y[idx], rcond=None)
        boot_g.append(bb[-1])
    g_boot_ci = (float(np.percentile(boot_g, 2.5)),
                 float(np.percentile(boot_g, 97.5)))

    # ---- extended regression: + theta * w * A ------------------------------
    for r in rows:
        r['wA'] = r['w'] * r['A']
    X2, cm2, cc2 = cell_dummies(rows, ['w', 'wA'])
    names2 = names1 + ['theta']
    res2_m, _ = ols_cluster(y, X2, cm2, names2)
    res2_c, _ = ols_cluster(y, X2, cc2, names2)
    res2_class, _ = ols_cluster(y, X2, cl_class, names2)

    # ---- per-cell table (coverage: gamma_m, n_wrong, n_right) -------------
    rng2 = np.random.RandomState(SEED)
    s7 = {(r['model'], r['stratum'], r['method']): r for r in
          csv.DictReader(open(ROOT / 'outputs/tables/stage7_cellconsistency.csv'))}
    cell_rows, meta_rows = [], []
    for cell in CELLS:
        cr = [r for r in rows if r['cell'] == cell]
        rw = np.array([r['r'] for r in cr if r['w'] == 1])
        rr_ = np.array([r['r'] for r in cr if r['w'] == 0])
        gamma_m = float(rw.mean() - rr_.mean()) if len(rw) and len(rr_) else float('nan')
        row = {'model': cell[0], 'stratum': cell[1], 'method': cell[2],
               'A_pre': cr[0]['A'], 'n': len(cr), 'n_wrong': len(rw),
               'n_right': len(rr_), 'gamma_m': gamma_m,
               'mean_r_wrong': float(rw.mean()) if len(rw) else float('nan'),
               'mean_r_right': float(rr_.mean()) if len(rr_) else float('nan'),
               'stage7_verdict': s7[cell]['verdict'] if cell in s7 else ''}
        # replicate the stage-7 paired bootstrap of d_m (own seed stream)
        if s7[cell]['verdict'] != 'insufficient_n':
            mw = np.array([r['im'] for r in cr if r['w'] == 1])
            mr = np.array([r['im'] for r in cr if r['w'] == 0])
            cw = np.array([r['ic'] for r in cr if r['w'] == 1])
            crr = np.array([r['ic'] for r in cr if r['w'] == 0])
            nw, nr = len(mw), len(mr)
            gaps = []
            for _ in range(B):
                iw = rng2.randint(0, nw, nw)
                ir = rng2.randint(0, nr, nr)
                gaps.append((mw[iw].mean() - mr[ir].mean()) -
                            (cw[iw].mean() - crr[ir].mean()))
            gaps = np.array(gaps)
            d_m = float((mw.mean() - mr.mean()) - (cw.mean() - crr.mean()))
            se = float(gaps.std(ddof=1))
            ci95 = (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))
            ci90 = (float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95)))
            row.update({'d_m': d_m, 'se_m': se, 'd_lo95': ci95[0], 'd_hi95': ci95[1],
                        'd_lo90': ci90[0], 'd_hi90': ci90[1],
                        'equiv_002': bool(ci90[0] > -EQ_MARGIN and ci90[1] < EQ_MARGIN),
                        's7_gap_lo': float(s7[cell]['gap_lo']),
                        's7_gap_hi': float(s7[cell]['gap_hi'])})
            meta_rows.append({'model': cell[0], 'stratum': cell[1], 'method': cell[2],
                              'A_pre': cr[0]['A'], 'd_m': d_m, 'se_m': se,
                              'lo95': ci95[0], 'hi95': ci95[1]})
        cell_rows.append(row)

    # ---- 2. DL random-effects meta over the judged cells -------------------
    d = np.array([r['d_m'] for r in meta_rows])
    se = np.array([r['se_m'] for r in meta_rows])
    A = np.array([r['A_pre'] for r in meta_rows])
    k = len(d)
    wv = 1.0 / se ** 2
    d_f = float(wv @ d / wv.sum())
    Q = float(wv @ (d - d_f) ** 2)
    C = float(wv.sum() - (wv ** 2).sum() / wv.sum())
    tau2 = max(0.0, (Q - (k - 1)) / C)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    d_re = float(ws @ d / ws.sum())
    se_re = float(np.sqrt(1.0 / ws.sum()))

    # ---- meta-regression: WLS d ~ 1 + A ------------------------------------
    def wls(dd, AA, ww):
        Xd = np.column_stack([np.ones_like(AA), AA])
        Wm = np.diag(ww)
        XtWX = Xd.T @ Wm @ Xd
        beta = np.linalg.solve(XtWX, Xd.T @ Wm @ dd)
        V = np.linalg.inv(XtWX)
        return beta, V

    b_fe, V_fe = wls(d, A, wv)
    b_re, V_re = wls(d, A, ws)
    slope_fe, slope_se_fe = float(b_fe[1]), float(np.sqrt(V_fe[1, 1]))
    slope_re, slope_se_re = float(b_re[1]), float(np.sqrt(V_re[1, 1]))
    Amin, Amax = float(A.min()), float(A.max())
    fit_lo, fit_hi = float(b_fe[0] + b_fe[1] * Amin), float(b_fe[0] + b_fe[1] * Amax)
    slope_ci = (slope_fe - 1.959964 * slope_se_fe,
                slope_fe + 1.959964 * slope_se_fe)
    reversal = (np.sign(fit_lo) != np.sign(fit_hi)) and (slope_ci[0] > 0 or slope_ci[1] < 0)
    decay = slope_ci[0] > 0  # residual shrinks as accuracy decreases

    # ---- 3. frozen branch rule ----------------------------------------------
    g_ci = g['z_ci']
    g_sig = not (g_ci[0] <= 0 <= g_ci[1])
    if not g_sig:
        branch = 'weak'
    elif decay or reversal:
        branch = 'strong_scoped'
    else:
        branch = 'strong'

    # ---- archival: stage-7 count verdict + sign test ------------------------
    st = stats.binomtest(12, 23, 0.5, alternative='two-sided')
    verdicts_by_stratum = {}
    for r in csv.DictReader(open(ROOT / 'outputs/tables/stage7_cellconsistency.csv')):
        if r['verdict'] in ('sharpening_explains', 'residual_structure'):
            verdicts_by_stratum.setdefault(r['stratum'], {'E': 0, 'R': 0})
            verdicts_by_stratum[r['stratum']][
                'E' if r['verdict'] == 'sharpening_explains' else 'R'] += 1

    # ---- write tables --------------------------------------------------------
    outdir = ROOT / 'outputs' / 'tables'

    def wcsv(name, rows_):
        keys = sorted({k for r in rows_ for k in r})
        with open(outdir / f'stage8_{name}.csv', 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows_)
        print(f'stage8_{name}.csv: {len(rows_)} rows')

    reg_rows = []
    for nm, res, clus in [('gamma', res_m, 'model'), ('gamma', res_c, 'cell'),
                          ('gamma', res_class, 'class'),
                          ('gamma', res_meth, 'method')]:
        reg_rows.append({'row_type': 'main_model', 'term': nm, 'cluster': clus,
                         'beta': res['gamma']['beta'], 'se': res['gamma']['se'],
                         'ci_lo': res['gamma']['z_ci'][0], 'ci_hi': res['gamma']['z_ci'][1]})
    reg_rows.append({'row_type': 'main_model', 'term': 'gamma', 'cluster': 'model_t(G-1)',
                     'beta': g['beta'], 'se': g['se'], 'ci_lo': g_t3ci[0], 'ci_hi': g_t3ci[1]})
    reg_rows.append({'row_type': 'main_model', 'term': 'gamma', 'cluster': 'model_cluster_bootstrap',
                     'beta': g['beta'], 'se': float(np.std(boot_g, ddof=1)),
                     'ci_lo': g_boot_ci[0], 'ci_hi': g_boot_ci[1]})
    for clus, res in [('model', res2_m), ('cell', res2_c), ('class', res2_class)]:
        reg_rows.append({'row_type': 'extended_model', 'term': 'gamma', 'cluster': clus,
                         'beta': res['gamma']['beta'], 'se': res['gamma']['se'],
                         'ci_lo': res['gamma']['z_ci'][0], 'ci_hi': res['gamma']['z_ci'][1]})
        reg_rows.append({'row_type': 'extended_model', 'term': 'theta', 'cluster': clus,
                         'beta': res['theta']['beta'], 'se': res['theta']['se'],
                         'ci_lo': res['theta']['z_ci'][0], 'ci_hi': res['theta']['z_ci'][1]})
    for r in cell_rows:
        reg_rows.append({'row_type': 'per_cell', 'term': 'gamma_m', 'cluster': '',
                         'beta': r['gamma_m'], 'se': '', 'ci_lo': '', 'ci_hi': '',
                         'model': r['model'], 'stratum': r['stratum'], 'method': r['method'],
                         'A_pre': r['A_pre'], 'n': r['n'], 'n_wrong': r['n_wrong'],
                         'n_right': r['n_right'], 'mean_r_wrong': r['mean_r_wrong'],
                         'mean_r_right': r['mean_r_right'], 'stage7_verdict': r['stage7_verdict'],
                         **{k: r[k] for k in ('d_m', 'se_m', 'd_lo95', 'd_hi95',
                                             'd_lo90', 'd_hi90', 'equiv_002',
                                             's7_gap_lo', 's7_gap_hi') if k in r}})
    wcsv('residual_regression', reg_rows)

    wcsv('meta', meta_rows + [{'model': 'POOLED_DL', 'stratum': '', 'method': '',
                               'A_pre': '', 'd_m': d_re, 'se_m': se_re,
                               'lo95': d_re - 1.959964 * se_re,
                               'hi95': d_re + 1.959964 * se_re}])
    wcsv('metareg', [
        {'weights': 'inverse_variance(FE)', 'intercept': float(b_fe[0]),
         'slope': slope_fe, 'slope_se': slope_se_fe,
         'slope_ci_lo': slope_ci[0], 'slope_ci_hi': slope_ci[1],
         'A_min': Amin, 'A_max': Amax, 'fit_at_Amin': fit_lo, 'fit_at_Amax': fit_hi,
         'sign_reversal_in_range': bool(reversal)},
        {'weights': 'random_effects(DL)', 'intercept': float(b_re[0]),
         'slope': slope_re, 'slope_se': slope_se_re,
         'slope_ci_lo': slope_re - 1.959964 * slope_se_re,
         'slope_ci_hi': slope_re + 1.959964 * slope_se_re,
         'A_min': Amin, 'A_max': Amax,
         'fit_at_Amin': float(b_re[0] + b_re[1] * Amin),
         'fit_at_Amax': float(b_re[0] + b_re[1] * Amax),
         'sign_reversal_in_range': ''}])
    eq_rows = [{'model': r['model'], 'stratum': r['stratum'], 'method': r['method'],
                'd_m': r['d_m'], 'lo90': r['d_lo90'], 'hi90': r['d_hi90'],
                'equivalent_within_0.02': r['equiv_002']} for r in cell_rows
               if 'd_m' in r]
    wcsv('equivalence', eq_rows)

    # ---- summary -------------------------------------------------------------
    n_eq = sum(1 for r in eq_rows if r['equivalent_within_0.02'])
    summary = {
        'N_samples': len(rows), 'N_wrong': int(w.sum()),
        'gamma': {kk: (g[kk] if not isinstance(g[kk], tuple) else list(g[kk]))
                  for kk in ('beta', 'se', 'z_ci')},
        'gamma_ci_t3': list(g_t3ci), 'gamma_ci_cluster_bootstrap': list(g_boot_ci),
        'gamma_cell_clustered_ci': list(res_c['gamma']['z_ci']),
        'gamma_class_clustered_ci': list(res_class['gamma']['z_ci']),
        'gamma_method_clustered_ci': list(res_meth['gamma']['z_ci']),
        'theta': {'beta': res2_m['theta']['beta'], 'se': res2_m['theta']['se'],
                  'ci': list(res2_m['theta']['z_ci']),
                  'ci_cell_clustered': list(res2_c['theta']['z_ci']),
                  'ci_class_clustered': list(res2_class['theta']['z_ci'])},
        'gamma_extended': {'beta': res2_m['gamma']['beta'],
                           'ci': list(res2_m['gamma']['z_ci'])},
        'meta': {'k': k, 'pooled_d_DL': d_re, 'se': se_re,
                 'ci': [d_re - 1.959964 * se_re, d_re + 1.959964 * se_re],
                 'Q': Q, 'I2': I2, 'tau2': tau2},
        'metareg': {'slope_fe': slope_fe, 'slope_ci': list(slope_ci),
                    'slope_re_weights': slope_re,
                    'slope_re_ci': [slope_re - 1.959964 * slope_se_re,
                                    slope_re + 1.959964 * slope_se_re],
                    'slope_unweighted': float(np.polyfit(A, d, 1)[0]),
                    'fit_at_Amin': fit_lo, 'fit_at_Amax': fit_hi,
                    'sign_reversal_in_range': bool(reversal)},
        'equivalence': {'margin': EQ_MARGIN, 'n_equivalent': n_eq,
                        'k': len(eq_rows)},
        'branch': branch,
        'stage7_archive': {'E': 11, 'R': 12, 'insufficient': 9,
                           'sign_test_p': float(st.pvalue),
                           'verdicts_by_stratum': verdicts_by_stratum},
    }
    json.dump(summary, open(outdir / 'stage8_summary.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float)[:2600])


if __name__ == '__main__':
    main()
