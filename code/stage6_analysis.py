"""Stage-6 analysis: builds all experiment tables from the faithful runs.

Reads outputs/raw/{model}_s6_{eval,dev,dose}_naming.jsonl (+ firstdist npz) and
produces outputs/tables/stage6_*.csv + outputs/tables/stage6_summary.json.

Pre-registered judgments (PREREGISTER_STAGE6.md section 4):
  H-A  paired confidence increment on unchanged-answer samples, bootstrap CI
  H-B  sharpening controls (truncation / temperature) reproduction ratio rho
  C    four-cell decomposition, pairwise increment-difference CIs
  D    name-level confidence (product of step probs) on wrong answers
  E    difference-in-differences of ECE (low_acc minus high_acc) + rescue rates
  F    one-shot constant-offset correction estimated on the dev subset
  G    VCD dose-response on q4b (alpha in {0.5, 1.0, 2.0})
Bootstrap: paired resampling of files, B=2000, seed 20260915.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import scoring  # noqa: E402

MODELS = ['q4b', 'q9b', 'llava16', 'internvl4b']
METHODS = ['vcd', 'm3id', 'dola', 'deco']
LEGACY = {'q4b': 'outputs/raw/q4b_main_naming.jsonl',
          'q9b': 'outputs/raw/q9b_main_naming.jsonl',
          'llava16': 'outputs/raw/llava16_food101_main.jsonl',
          'internvl4b': 'outputs/raw/internvl4b_food101_main.jsonl'}
LEGACY_MAP = {'vcd': 'vcd', 'mib': 'm3id', 'lcd': 'dola'}  # variant -> faithful
B, SEED = 2000, 20260915
T_GRID = np.round(np.arange(0.20, 1.001, 0.05), 2)
TRUNC_BETA = 0.1
DOSE_ALPHAS = {'vcd_a05': 0.5, 'vcd': 1.0, 'vcd_a20': 2.0}

rng = np.random.RandomState(SEED)


def load(model, part):
    """(file, method) -> record, ignoring errors and round-A rows."""
    recs = {}
    p = ROOT / 'outputs' / 'raw' / f'{model}_s6_{part}_naming.jsonl'
    if not p.exists():
        return recs
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'error' in r or r.get('method') in (None, 'roundA'):
            continue
        recs[(r['file'], r['method'])] = r
    return recs


def ece(confs, corrects):
    return scoring.calibration_error(confs, corrects, 10)


def norm_text(t):
    return scoring._norm(t or '')


def boot_ci(vals, stat=None):
    """Mean bootstrap CI, or CI of a custom stat over resampled index lists."""
    vals = np.asarray(vals, float)
    n = len(vals)
    if n == 0:
        return (float('nan'),) * 3
    if stat is None:
        out = [vals[rng.randint(0, n, n)].mean() for _ in range(B)]
    else:
        out = [stat(rng.randint(0, n, n)) for _ in range(B)]
    return float(vals.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_ece_diff(direct_rows, m_rows, mask=None):
    """Delta ECE (m - direct) point estimate."""
    if mask is None:
        mask = np.ones(len(direct_rows), bool)
    c_d = [r['answer_maxp'] for r, k in zip(direct_rows, mask) if k]
    y_d = [r['outcome'] == 'correct' for r, k in zip(direct_rows, mask) if k]
    c_m = [r['answer_maxp'] for r, k in zip(m_rows, mask) if k]
    y_m = [r['outcome'] == 'correct' for r, k in zip(m_rows, mask) if k]
    return ece(c_m, y_m) - ece(c_d, y_d)


def main(models=None, parts=('eval', 'dev')):
    models = models or MODELS
    tables = {k: [] for k in ['core', 'fourcell', 'sharpening', 'correction',
                              'did', 'dose', 'dname', 'variant', 'consistency']}
    summary = {}

    for model in models:
        ev = load(model, 'eval')
        dv = load(model, 'dev')
        if not ev:
            continue
        files = sorted({f for f, m in ev if m == 'direct'})
        firstdist_eval = np.load(ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_eval.npz') \
            if (ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_eval.npz').exists() else {}
        firstdist_dev = np.load(ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_dev.npz') \
            if (ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_dev.npz').exists() else {}

        # -------- legacy direct consistency check (pre-registered gate) --------
        legacy_out = {}
        lp = ROOT / LEGACY[model]
        if lp.exists():
            for line in open(lp):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('method') == 'direct' and r.get('task') == 'naming':
                    legacy_out[r['file']] = r.get('outcome')
        n_cmp = n_mis = 0
        for f in files:
            if f in legacy_out:
                n_cmp += 1
                n_mis += legacy_out[f] != ev[(f, 'direct')]['outcome']
        mismatch_rate = n_mis / n_cmp if n_cmp else float('nan')
        tables['consistency'].append({'model': model, 'n_compared': n_cmp,
                                      'n_mismatch': n_mis,
                                      'mismatch_rate': round(mismatch_rate, 5),
                                      'gate_pass': mismatch_rate <= 0.005})
        summary.setdefault(model, {})['direct_consistency'] = {
            'n': n_cmp, 'mismatch': n_mis, 'rate': mismatch_rate}

        for stratum in ('low_acc', 'high_acc'):
            sfiles = [f for f in files
                      if ev[(f, 'direct')]['stratum'] == stratum]
            d_rows = [ev[(f, 'direct')] for f in sfiles]
            y_d = np.array([r['outcome'] == 'correct' for r in d_rows])
            c_d = np.array([r['answer_maxp'] for r in d_rows], float)
            ece_d = ece(c_d, y_d)
            summary.setdefault(model, {}).setdefault(stratum, {})['direct'] = {
                'n': len(sfiles), 'acc': float(y_d.mean()),
                'mean_conf': float(c_d.mean()), 'ece': ece_d}

            for method in METHODS:
                m_rows, kept = [], []
                for f, dr in zip(sfiles, d_rows):
                    r = ev.get((f, method))
                    if r is None or r.get('answer_maxp') is None:
                        continue
                    m_rows.append(r)
                    kept.append(dr)
                if not m_rows:
                    continue
                y_m = np.array([r['outcome'] == 'correct' for r in m_rows])
                c_m = np.array([r['answer_maxp'] for r in m_rows], float)
                unchanged = np.array([
                    norm_text(r['text']) == norm_text(d['text']) and
                    r['tokens'][0] == d['tokens'][0]
                    for r, d in zip(m_rows, kept)])
                ece_m = ece(c_m, y_m)
                dconf_all = c_m - np.array([d['answer_maxp'] for d in kept])
                dconf_unc = dconf_all[unchanged]
                m_ci = boot_ci(dconf_unc)
                dece_point = ece_m - ece_d
                dece_boots = []
                n = len(m_rows)
                for _ in range(B):
                    idx = rng.randint(0, n, n)
                    dece_boots.append(paired_ece_diff(
                        [kept[i] for i in idx], [m_rows[i] for i in idx]))
                dece_ci = (float(np.percentile(dece_boots, 2.5)),
                           float(np.percentile(dece_boots, 97.5)))
                rescue = float(y_m[~y_d_kept(kept)].mean()) if (~y_d_kept(kept)).any() else float('nan')
                brk = float((~y_m)[y_d_kept(kept)].mean()) if y_d_kept(kept).any() else float('nan')

                tables['core'].append({
                    'model': model, 'stratum': stratum, 'method': method,
                    'n': len(m_rows), 'acc_direct': float(y_d_kept(kept).mean()),
                    'acc_method': float(y_m.mean()),
                    'abstain_direct': float(np.mean([d['outcome'] == 'abstain' for d in kept])),
                    'abstain_method': float(np.mean([r['outcome'] == 'abstain' for r in m_rows])),
                    'conf_direct': float(np.mean([d['answer_maxp'] for d in kept])),
                    'conf_method': float(c_m.mean()),
                    'dconf_all': float(dconf_all.mean()),
                    'dconf_unchanged': m_ci[0], 'dconf_un_lo': m_ci[1], 'dconf_un_hi': m_ci[2],
                    'n_unchanged': int(unchanged.sum()),
                    'ece_direct': ece_d, 'ece_method': ece_m,
                    'dece': dece_point, 'dece_lo': dece_ci[0], 'dece_hi': dece_ci[1],
                    'rescue_rate': rescue, 'break_rate': brk,
                })

                # ---------------- experiment C: four-cell decomposition ----------
                both_wrong_same = 0
                both_wrong = 0
                cells = {'always_right': [], 'always_wrong': [],
                         'corrected': [], 'broken': []}
                for r, d in zip(m_rows, kept):
                    dc, mc = d['outcome'] == 'correct', r['outcome'] == 'correct'
                    inc = r['answer_maxp'] - d['answer_maxp']
                    if dc and mc:
                        cells['always_right'].append(inc)
                    elif (not dc) and (not mc):
                        cells['always_wrong'].append(inc)
                        both_wrong += 1
                        both_wrong_same += norm_text(r['text']) == norm_text(d['text'])
                    elif (not dc) and mc:
                        cells['corrected'].append(inc)
                    else:
                        cells['broken'].append(inc)
                row_c = {'model': model, 'stratum': stratum, 'method': method}
                for cname, incs in cells.items():
                    mean, lo, hi = boot_ci(incs)
                    row_c[f'{cname}_n'] = len(incs)
                    row_c[f'{cname}_dconf'] = mean
                    row_c[f'{cname}_lo'] = lo
                    row_c[f'{cname}_hi'] = hi
                row_c['same_wrong_answer_share'] = both_wrong_same / both_wrong if both_wrong else float('nan')
                keys = list(cells)
                row_c['cells_similar'] = 'yes'
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = cells[keys[i]], cells[keys[j]]
                        if not a or not b:
                            continue
                        boots = [np.asarray(a)[rng.randint(0, len(a), len(a))].mean() -
                                 np.asarray(b)[rng.randint(0, len(b), len(b))].mean()
                                 for _ in range(B)]
                        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
                        if lo > 0 or hi < 0:
                            row_c['cells_similar'] = f'no({keys[i]}vs{keys[j]})'
                tables['fourcell'].append(row_c)

                # ---------------- experiment B: sharpening controls --------------
                if firstdist_eval and firstdist_dev:
                    dev_files = sorted({f for f, m in dv if m == 'direct'})
                    dev_d = [dv[(f, 'direct')] for f in dev_files]
                    dev_m = [dv.get((f, method)) for f in dev_files]
                    dev_unc = [i for i, (d, r) in enumerate(zip(dev_d, dev_m))
                               if r is not None and r.get('answer_maxp') is not None
                               and norm_text(r['text']) == norm_text(d['text'])
                               and r['tokens'][0] == d['tokens'][0]]
                    if len(dev_unc) >= 20:
                        target = np.mean([dev_m[i]['answer_maxp'] -
                                          firstdist_dev[dev_files[i]][dev_m[i]['tokens'][0]]
                                          for i in dev_unc])
                        best_T, best_gap = float('nan'), float('inf')
                        for T in T_GRID:
                            ctrl_dev = np.mean([
                                _temp_conf(firstdist_dev[dev_files[i]],
                                           dev_d[i]['tokens'][0], T) -
                                firstdist_dev[dev_files[i]][dev_d[i]['tokens'][0]]
                                for i in dev_unc])
                            gap = abs(ctrl_dev - target)
                            if gap < best_gap:
                                best_T, best_gap = float(T), gap
                        # eval-half judgment
                        idx_un = [i for i in range(len(m_rows)) if unchanged[i]]
                        d_m_eval = np.mean(dconf_all[unchanged])
                        d_c1 = np.array([_trunc_conf(firstdist_eval[f],
                                                     kept[i]['tokens'][0]) -
                                         firstdist_eval[f][kept[i]['tokens'][0]]
                                         for i, f in zip(idx_un, [m_rows[i]['file'] for i in idx_un])])
                        d_c2 = np.array([_temp_conf(firstdist_eval[f],
                                                    kept[i]['tokens'][0], best_T) -
                                         firstdist_eval[f][kept[i]['tokens'][0]]
                                         for i, f in zip(idx_un, [m_rows[i]['file'] for i in idx_un])])
                        rho1 = 1 - abs(d_c1.mean() - d_m_eval) / abs(d_m_eval) if d_m_eval else float('nan')
                        rho2 = 1 - abs(d_c2.mean() - d_m_eval) / abs(d_m_eval) if d_m_eval else float('nan')
                        _, g1lo, g1hi = boot_ci(d_c1 - dconf_all[unchanged])
                        _, g2lo, g2hi = boot_ci(d_c2 - dconf_all[unchanged])
                        # control ECE over the whole stratum (answers = direct's)
                        cf1 = [_trunc_conf(firstdist_eval[f], r['tokens'][0])
                               for r, f in zip(d_rows, sfiles) if f in firstdist_eval]
                        cf2 = [_temp_conf(firstdist_eval[f], r['tokens'][0], best_T)
                               for r, f in zip(d_rows, sfiles) if f in firstdist_eval]
                        yl = [r['outcome'] == 'correct'
                              for r, f in zip(d_rows, sfiles) if f in firstdist_eval]
                        verdict = ('sharpening' if (rho2 >= 0.8 and d_m_eval > 0 and d_c2.mean() > 0)
                                   else ('structure' if (rho2 < 0.5 or d_c2.mean() <= 0 < d_m_eval)
                                         else 'partial'))
                        tables['sharpening'].append({
                            'model': model, 'stratum': stratum, 'method': method,
                            'T_fit': best_T, 'dev_target': float(target),
                            'dconf_method': float(d_m_eval),
                            'dconf_ctrl_trunc': float(d_c1.mean()), 'rho_trunc': rho1,
                            'gap_trunc_lo': g1lo, 'gap_trunc_hi': g1hi,
                            'dconf_ctrl_temp': float(d_c2.mean()), 'rho_temp': rho2,
                            'gap_temp_lo': g2lo, 'gap_temp_hi': g2hi,
                            'ece_ctrl_trunc': ece(cf1, yl), 'ece_ctrl_temp': ece(cf2, yl),
                            'ece_method': ece_m, 'ece_direct': ece_d,
                            'verdict': verdict})

                        # ---------------- experiment F: offset correction ----------
                        offset = float(target)
                        cf_corr = np.clip(c_m - offset, 0, 1)
                        ece_after = ece(cf_corr, y_m)
                        tables['correction'].append({
                            'model': model, 'stratum': stratum, 'method': method,
                            'offset_dev': offset,
                            'ece_direct': ece_d, 'ece_before': ece_m,
                            'ece_after': ece_after,
                            'acc_before': float(y_m.mean()), 'acc_after': float(y_m.mean()),
                            'high_recovered': bool(ece_after <= ece_d + 0.01) if stratum == 'high_acc' else ''})

                # ---------------- experiment D: name-level confidence ------------
                name_conf = lambda r: (float(np.prod(r['step_probs']))
                                       if r.get('step_probs') else float('nan'))
                m_wrong = np.array([r['outcome'] in ('wrong', 'wrong_unparsed')
                                    for r in m_rows])
                if m_wrong.any():
                    inc_first = c_m[m_wrong] - np.array([d['answer_maxp'] for d, w in zip(kept, m_wrong) if w])
                    inc_name = np.array([name_conf(r) for r, w in zip(m_rows, m_wrong) if w]) - \
                        np.array([name_conf(d) for d, w in zip(kept, m_wrong) if w])
                    fi = boot_ci(inc_first)
                    ni = boot_ci(inc_name)
                    tables['dname'].append({
                        'model': model, 'stratum': stratum, 'method': method,
                        'n_wrong': int(m_wrong.sum()),
                        'nameconf_direct': float(np.nanmean([name_conf(d) for d, w in zip(kept, m_wrong) if w])),
                        'nameconf_method': float(np.nanmean([name_conf(r) for r, w in zip(m_rows, m_wrong) if w])),
                        'd_first_token': fi[0], 'd_first_lo': fi[1], 'd_first_hi': fi[2],
                        'd_name_level': ni[0], 'd_name_lo': ni[1], 'd_name_hi': ni[2]})

            # stratum-level summary for DiD (per method, done above per stratum;
            # DiD assembled after both strata)
        summary[model]['files'] = len(files)

    # ---------------- experiment E: difference in differences ----------------
    core = {(r['model'], r['stratum'], r['method']): r for r in tables['core']}
    for model in models:
        for method in METHODS:
            if (model, 'low_acc', method) not in core or (model, 'high_acc', method) not in core:
                continue
            lo, hi = core[(model, 'low_acc', method)], core[(model, 'high_acc', method)]
            did = lo['dece'] - hi['dece']
            # bootstrap DiD: resample each stratum's files independently
            did_boots = []
            ev = load(model, 'eval')
            fl_lo = [f for f, m in ev if m == method and ev[(f, method)]['stratum'] == 'low_acc']
            fl_hi = [f for f, m in ev if m == method and ev[(f, method)]['stratum'] == 'high_acc']
            dr_lo = [(ev[(f, 'direct')], ev[(f, method)]) for f in fl_lo
                     if (f, 'direct') in ev and ev[(f, method)].get('answer_maxp') is not None]
            dr_hi = [(ev[(f, 'direct')], ev[(f, method)]) for f in fl_hi
                     if (f, 'direct') in ev and ev[(f, method)].get('answer_maxp') is not None]
            for _ in range(B):
                i1 = rng.randint(0, len(dr_lo), len(dr_lo))
                i2 = rng.randint(0, len(dr_hi), len(dr_hi))
                d1 = paired_ece_diff([dr_lo[i][0] for i in i1], [dr_lo[i][1] for i in i1])
                d2 = paired_ece_diff([dr_hi[i][0] for i in i2], [dr_hi[i][1] for i in i2])
                did_boots.append(d1 - d2)
            tables['did'].append({
                'model': model, 'method': method,
                'dece_low': lo['dece'], 'dece_high': hi['dece'], 'did': did,
                'did_lo': float(np.percentile(did_boots, 2.5)),
                'did_hi': float(np.percentile(did_boots, 97.5)),
                'did_excludes0': bool(np.percentile(did_boots, 2.5) > 0 or
                                      np.percentile(did_boots, 97.5) < 0),
                'rescue_low': lo['rescue_rate'], 'rescue_high': hi['rescue_rate'],
                'dconf_low': lo['dconf_unchanged'], 'dconf_high': hi['dconf_unchanged']})

    # ---------------- experiment G: dose-response (q4b) ------------------------
    dz = load('q4b', 'dose')
    if dz:
        ev = load('q4b', 'eval')
        files = sorted({f for f, m in ev if m == 'direct'})
        for stratum in ('low_acc', 'high_acc'):
            for tag, alpha in DOSE_ALPHAS.items():
                rows = [(ev[(f, 'direct')], dz[(f, tag)]) for f in files
                        if (f, tag) in dz and (f, 'direct') in ev
                        and dz[(f, tag)]['stratum'] == stratum
                        and dz[(f, tag)].get('answer_maxp') is not None]
                if not rows:
                    continue
                y_d = [r[0]['outcome'] == 'correct' for r in rows]
                y_m = [r[1]['outcome'] == 'correct' for r in rows]
                c_d = [r[0]['answer_maxp'] for r in rows]
                c_m = [r[1]['answer_maxp'] for r in rows]
                tables['dose'].append({
                    'model': 'q4b', 'stratum': stratum, 'alpha': alpha,
                    'n': len(rows), 'acc_direct': float(np.mean(y_d)),
                    'acc_method': float(np.mean(y_m)),
                    'dconf_all': float(np.mean(np.array(c_m) - np.array(c_d))),
                    'ece_direct': ece(c_d, y_d), 'ece_method': ece(c_m, y_m),
                    'dece': ece(c_m, y_m) - ece(c_d, y_d)})
        # class-level crossing per alpha (50 strata classes)
        for tag, alpha in DOSE_ALPHAS.items():
            per_class = {}
            for (f, m), r in dz.items():
                if m != tag or r.get('answer_maxp') is None:
                    continue
                d = ev.get((f, 'direct'))
                if d is None:
                    continue
                per_class.setdefault(r['class'], []).append((d, r))
            xs, ys = [], []
            for cls, pairs in per_class.items():
                if len(pairs) < 8:
                    continue
                acc = np.mean([d['outcome'] == 'correct' for d, _ in pairs])
                dece = ece([r['answer_maxp'] for _, r in pairs],
                           [r['outcome'] == 'correct' for _, r in pairs]) - \
                    ece([d['answer_maxp'] for d, _ in pairs],
                        [d['outcome'] == 'correct' for d, _ in pairs])
                xs.append(acc)
                ys.append(dece)
            if len(xs) >= 10:
                b1, b0 = np.polyfit(xs, ys, 1)
                if b1 < 0:
                    cross = -b0 / b1
                    classes = sorted(per_class)
                    boots = []
                    for _ in range(B):
                        sel = rng.randint(0, len(xs), len(xs))
                        if len(set(np.array(xs)[sel])) < 3:
                            continue
                        bb1, bb0 = np.polyfit(np.array(xs)[sel], np.array(ys)[sel], 1)
                        if bb1 < 0:
                            boots.append(-bb0 / bb1)
                    ci = (float(np.percentile(boots, 2.5)),
                          float(np.percentile(boots, 97.5))) if boots else (float('nan'),) * 2
                else:
                    cross, ci = float('nan'), (float('nan'), float('nan'))
                tables['dose'].append({'model': 'q4b_class_crossing', 'stratum': 'both',
                                       'alpha': alpha, 'n': len(xs),
                                       'crossing': cross, 'cross_lo': ci[0], 'cross_hi': ci[1]})

    # ---------------- variant-implementation supplement ------------------------
    for model in models:
        lp = ROOT / LEGACY[model]
        if not lp.exists():
            continue
        legacy = {}
        for line in open(lp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task') == 'naming' and r.get('method') in LEGACY_MAP:
                legacy.setdefault(r['method'], {})[r['file']] = r
        ev = load(model, 'eval')
        for var, faithful in LEGACY_MAP.items():
            for stratum in ('low_acc', 'high_acc'):
                old = {f: r for f, r in legacy.get(var, {}).items()
                       if r.get('stratum', '') in ('deficient', 'known', 'low_acc', 'high_acc')}
                # stratum names in legacy files are deficient/known
                rows = []
                for (f, m), r in ev.items():
                    if m != faithful or r.get('answer_maxp') is None:
                        continue
                    d = ev.get((f, 'direct'))
                    o = old.get(f)
                    od = legacy.get('direct', {}).get(f)
                    if d is None or o is None or od is None:
                        continue
                    if r['stratum'] != stratum or o.get('stratum') != (
                            'deficient' if stratum == 'low_acc' else 'known'):
                        continue
                    rows.append((d, r, od, o))
                if len(rows) < 100:
                    continue
                dece_old = ece([o['answer_maxp'] for _, _, _, o in rows],
                               [o['outcome'] == 'correct' for _, _, _, o in rows]) - \
                    ece([od['answer_maxp'] for _, _, od, _ in rows],
                        [od['outcome'] == 'correct' for _, _, od, _ in rows])
                d_new = [r for _, r, _, _ in rows]
                d_old_direct = [od for _, _, od, _ in rows]
                dece_new = ece([r['answer_maxp'] for _, r, _, _ in rows],
                               [r['outcome'] == 'correct' for _, r, _, _ in rows]) - \
                    ece([d['answer_maxp'] for d, _, _, _ in rows],
                        [d['outcome'] == 'correct' for d, _, _, _ in rows])
                tables['variant'].append({
                    'model': model, 'stratum': stratum,
                    'variant_impl': var, 'faithful_impl': faithful,
                    'n': len(rows),
                    'acc_variant': float(np.mean([o['outcome'] == 'correct' for _, _, _, o in rows])),
                    'acc_faithful': float(np.mean([r['outcome'] == 'correct' for _, r, _, _ in rows])),
                    'dece_variant': dece_old, 'dece_faithful': dece_new})

    # ---------------- write ----------------
    outdir = ROOT / 'outputs' / 'tables'
    outdir.mkdir(exist_ok=True)
    import csv
    for name, rows in tables.items():
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(outdir / f'stage6_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'stage6_{name}.csv: {len(rows)} rows')
    summary['tables'] = {k: len(v) for k, v in tables.items() if v}
    json.dump(summary, open(outdir / 'stage6_summary.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print('stage6_summary.json written')


def y_d_kept(kept):
    return np.array([d['outcome'] == 'correct' for d in kept])


def _trunc_conf(vec, tok, beta=TRUNC_BETA):
    v = np.asarray(vec, dtype=np.float64)
    keep = v >= beta * v.max()
    return v[tok] / v[keep].sum()


def _temp_conf(vec, tok, T):
    v = np.asarray(vec, dtype=np.float64)
    w = v ** (1.0 / T)
    return w[tok] / w.sum()


if __name__ == '__main__':
    main(models=sys.argv[1].split(',') if len(sys.argv) > 1 else None)
