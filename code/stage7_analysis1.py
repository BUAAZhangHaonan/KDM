"""Stage-7 part 1: wording-deciding analyses, zero new inference.

  1. cell-wise sharpening consistency (decisive, PREREGISTER_STAGE7 sec 1.1):
     per (model, stratum, method): delta = mean increment(always-wrong) minus
     mean increment(always-right) among unchanged-answer samples, for the method
     and for the temperature control fitted in stage 6; verdict per cell;
     E-vs-R count decides the wording branch.
  2. distribution-level agreement: Spearman rank correlation and
     Wasserstein-1 distance between per-sample method and control increments.
  3. strength positive control: legacy MIB variant (blur branch + linear decay),
     explicitly NOT a published method.
  4. dual-caliber counts of the stage-6 reproduction verdicts.
  5. offset drift: dev-half offset minus eval-half mean increment, per cell.
Outputs: outputs/tables/stage7_{cellconsistency, distlevel, poscontrol,
dualcaliber, drift}.csv + stage7_summary.json (branch decision).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import scoring  # noqa: E402
from stage6_analysis import load, ece, norm_text, _temp_conf  # noqa: E402

MODELS = ['q4b', 'q9b', 'llava16', 'internvl4b']
ROUND_A = {'q4b'}
METHODS = ['vcd', 'm3id', 'dola', 'deco']
B, SEED = 2000, 20260916
rng = np.random.RandomState(SEED)
LEGACY = {'q4b': 'outputs/raw/q4b_main_naming.jsonl',
          'q9b': 'outputs/raw/q9b_main_naming.jsonl'}


def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def wasserstein1(a, b):
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    ai = np.searchsorted(a, allv, side='right') / len(a)
    bi = np.searchsorted(b, allv, side='right') / len(b)
    return float(np.sum(np.abs(ai - bi)) / len(allv))


def main():
    sharp = {}
    import csv
    for r in csv.DictReader(open(ROOT / 'outputs/tables/stage6_sharpening.csv')):
        sharp[(r['model'], r['stratum'], r['method'])] = r
    corr = {(r['model'], r['stratum'], r['method']): r
            for r in csv.DictReader(open(ROOT / 'outputs/tables/stage6_correction.csv'))}

    rows_c, rows_d, drifts = [], [], []
    E = R = 0
    for model in MODELS:
        ev = load(model, 'eval')
        fd = np.load(ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_eval.npz')
        files = sorted({f for f, m in ev if m == 'direct'})
        for stratum in ('low_acc', 'high_acc'):
            for method in METHODS:
                s = sharp.get((model, stratum, method))
                if s is None:
                    continue
                T = float(s['T_fit'])
                inc_m_w, inc_m_r, inc_c_w, inc_c_r = [], [], [], []
                for f in files:
                    d, m = ev.get((f, 'direct')), ev.get((f, method))
                    if d is None or m is None or f not in fd:
                        continue
                    if d['stratum'] != stratum:
                        continue
                    if d.get('answer_maxp') is None or m.get('answer_maxp') is None:
                        continue
                    if norm_text(m['text']) != norm_text(d['text']) or \
                            m['tokens'][0] != d['tokens'][0]:
                        continue
                    v = np.asarray(fd[f], dtype=np.float64)
                    tok = m['tokens'][0]
                    p0 = float(v[tok])
                    im = m['answer_maxp'] - p0
                    ic = _temp_conf(v, tok, T) - p0
                    (inc_m_w if d['outcome'] != 'correct' else inc_m_r).append(im)
                    (inc_c_w if d['outcome'] != 'correct' else inc_c_r).append(ic)
                n_w, n_r = len(inc_m_w), len(inc_m_r)
                if n_w < 20 or n_r < 20:
                    rows_c.append({'model': model, 'stratum': stratum,
                                   'method': method, 'n_wrong': n_w, 'n_right': n_r,
                                   'verdict': 'insufficient_n'})
                    continue
                mw, mr = np.array(inc_m_w), np.array(inc_m_r)
                cw, cr = np.array(inc_c_w), np.array(inc_c_r)
                delta = float(mw.mean() - mr.mean())
                delta_c = float(cw.mean() - cr.mean())

                def boot_delta():
                    out = []
                    for _ in range(B):
                        iw = rng.randint(0, n_w, n_w)
                        ir = rng.randint(0, n_r, n_r)
                        out.append((mw[iw].mean() - mr[ir].mean()) -
                                   (cw[iw].mean() - cr[ir].mean()))
                    return out

                gaps = boot_delta()
                dlo, dhi = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
                same_sign = (delta > 0) == (delta_c > 0)
                ci0 = dlo <= 0 <= dhi
                verdict = ('sharpening_explains' if (same_sign and ci0)
                           else 'residual_structure')
                E += verdict == 'sharpening_explains'
                R += verdict == 'residual_structure'
                inc_all_m = np.concatenate([mw, mr])
                inc_all_c = np.concatenate([cw, cr])
                rows_c.append({
                    'model': model, 'stratum': stratum, 'method': method,
                    'n_wrong': n_w, 'n_right': n_r,
                    'delta_method': delta, 'delta_ctrl': delta_c,
                    'gap_lo': dlo, 'gap_hi': dhi,
                    'same_sign': same_sign, 'ci_contains0': ci0,
                    'verdict': verdict})
                rows_d.append({
                    'model': model, 'stratum': stratum, 'method': method,
                    'T_fit': T, 'n': n_w + n_r,
                    'spearman': spearman(inc_all_m, inc_all_c),
                    'wasserstein1': wasserstein1(inc_all_m, inc_all_c),
                    'mean_method': float(inc_all_m.mean()),
                    'mean_ctrl': float(inc_all_c.mean())})
                c = corr.get((model, stratum, method))
                if c is not None:
                    drifts.append({
                        'model': model, 'stratum': stratum, 'method': method,
                        'offset_dev': float(c['offset_dev']),
                        'increment_eval': float(s['dconf_method']),
                        'drift': float(c['offset_dev']) - float(s['dconf_method'])})

    # ------- 3. positive control (legacy MIB variant) -------------------------
    pos = []
    for model in ('q4b', 'q9b'):
        legacy = {'direct': {}, 'mib': {}, 'roundA': {}}
        for line in open(ROOT / LEGACY[model]):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task') == 'naming' and r.get('method') in legacy:
                legacy[r['method']][r['file']] = r
        for stratum in ('low_acc', 'high_acc'):
            old_st = 'deficient' if stratum == 'low_acc' else 'known'
            rows = [(d, m) for f, m in legacy['mib'].items()
                    if (d := legacy['direct'].get(f)) is not None
                    and m.get('stratum') == old_st and d.get('stratum') == old_st
                    and not (model in ROUND_A and
                             (legacy['roundA'].get(f, {}).get('abstained'))) ]
            if not rows:
                continue
            y_d = [d['outcome'] == 'correct' for d, _ in rows]
            y_m = [m['outcome'] == 'correct' for _, m in rows]
            c_d = [d['answer_maxp'] for d, _ in rows]
            c_m = [m['answer_maxp'] for _, m in rows]
            pos.append({
                'model': model, 'stratum': stratum,
                'config': 'MIB-variant (blur branch + linear decay; NOT a '
                          'published method; positive strength control)',
                'n': len(rows), 'acc_direct': float(np.mean(y_d)),
                'acc_method': float(np.mean(y_m)),
                'dconf': float(np.mean(np.array(c_m) - np.array(c_d))),
                'ece_direct': ece(c_d, y_d), 'ece_method': ece(c_m, y_m),
                'dece': ece(c_m, y_m) - ece(c_d, y_d),
                'rescue_rate': float(np.array(y_m)[~np.array(y_d)].mean())})

    # ------- 4. dual-caliber counts -------------------------------------------
    import collections
    vc = collections.Counter(r['verdict'] for r in
                             csv.DictReader(open(ROOT / 'outputs/tables/stage6_sharpening.csv')))
    dual = [{'caliber': 'all_32_cells', **{k: vc.get(k, 0) for k in
                                           ('sharpening', 'partial', 'structure')}},
            {'caliber': 'excluding_low_resolution(method increment < 0.04)'}]

    def verdict_cn(v):
        return {'sharpening': '完整复现', 'partial': '部分复现',
                'structure': '未复现（存在残余结构）'}.get(v, v)

    dual[1] = {'caliber': 'matchable_only(>=0.04)',
               **{verdict_cn(k) if False else k: sum(
                   1 for r in csv.DictReader(open(ROOT / 'outputs/tables/stage6_sharpening.csv'))
                   if r['verdict'] == k and float(r['dconf_method']) >= 0.04)
                  for k in ('sharpening', 'partial', 'structure')}}

    # ------- write -------------------------------------------------------------
    outdir = ROOT / 'outputs' / 'tables'
    for name, rows in [('cellconsistency', rows_c), ('distlevel', rows_d),
                       ('poscontrol', pos), ('dualcaliber', dual), ('drift', drifts)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(outdir / f'stage7_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'stage7_{name}.csv: {len(rows)} rows')
    drift_vals = [d['drift'] for d in drifts]
    summary = {
        'cellwise': {'sharpening_explains': E, 'residual_structure': R,
                     'branch': 'strong (R > E)' if R > E else 'weak (E >= R)'},
        'drift': {'mean': float(np.mean(drift_vals)), 'sd': float(np.std(drift_vals)),
                  'min': float(np.min(drift_vals)), 'max': float(np.max(drift_vals)),
                  'n_cells': len(drift_vals)},
        'poscontrol': pos,
    }
    json.dump(summary, open(outdir / 'stage7_summary.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print('BRANCH DECISION:', summary['cellwise'])
    print('drift:', summary['drift'])


if __name__ == '__main__':
    main()
