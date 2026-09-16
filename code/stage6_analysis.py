"""Stage-6 analysis: builds all experiment tables from the faithful runs.

Reads outputs/raw/{model}_s6_{eval,dev,dose}_naming.jsonl (+ firstdist npz) and
produces outputs/tables/stage6_*.csv + outputs/tables/stage6_summary.json.

Pre-registered judgments (docs/stage6/PREREGISTER_STAGE6.md section 4):
  H-A  paired confidence increment on unchanged-answer samples, bootstrap CI
  H-B  sharpening controls (truncation / temperature) reproduction ratio rho
  C    four-cell decomposition, pairwise increment-difference CIs
  D    name-level confidence (product of step probs) on wrong answers
  E    difference-in-differences of ECE (low_acc minus high_acc) + rescue rates
  F    one-shot constant-offset correction estimated on the dev subset
  G    VCD dose-response on q4b (alpha in {0.5, 1.0, 2.0})
Bootstrap: paired resampling of files, B=2000, seed 20260915.

Strata convention (established stages 3-5): q4b uses the style3 protocol; a
round-A abstention marks every config's outcome as 'abstain' and DROPS its
confidence (accuracy denominators keep the sample; confidence/ECE/increment
computations exclude it). The runner stores the already-overridden outcome plus
the 'abstained' flag; the legacy-consistency check compares RAW text-level
outcomes (recomputed via score_naming on both sides).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import scoring  # noqa: E402

MODELS = ['q4b', 'q9b', 'llava16', 'internvl4b']
ROUND_A = {'q4b'}          # models using the style3 round-A protocol
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
CLASSES = sorted({json.loads(l)['class'] for l in
                  open(ROOT / 'data' / 'samples_manifest.jsonl')})


def load(model, part):
    """(file, method) -> record; applies the round-A convention (see header)."""
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
        if model in ROUND_A and r.get('abstained'):
            r['outcome'] = 'abstain'
            r['answer_maxp'] = None
        recs[(r['file'], r['method'])] = r
    return recs


def raw_outcome(model, r):
    """Text-level naming outcome without the round-A override (legacy storage)."""
    if model in ROUND_A and r.get('abstained'):
        return scoring.score_naming(r['text'], r['class'], CLASSES)
    return r['outcome']


def ece(confs, corrects):
    return scoring.calibration_error(confs, corrects, 10)


def norm_text(t):
    return scoring._norm(t or '')


def boot_mean(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    if n == 0:
        return (float('nan'),) * 3
    boots = [vals[rng.randint(0, n, n)].mean() for _ in range(B)]
    return float(vals.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5))


def ece_delta(d_rows, m_rows):
    """dECE (method - direct) on rows where both have confidences."""
    pairs = [(d, m) for d, m in zip(d_rows, m_rows)
             if d.get('answer_maxp') is not None and m.get('answer_maxp') is not None]
    if not pairs:
        return float('nan')
    cd = [d['answer_maxp'] for d, _ in pairs]
    yd = [d['outcome'] == 'correct' for d, _ in pairs]
    cm = [m['answer_maxp'] for _, m in pairs]
    ym = [m['outcome'] == 'correct' for _, m in pairs]
    return ece(cm, ym) - ece(cd, yd)


def main(models=None):
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
        fde_p = ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_eval.npz'
        fdd_p = ROOT / 'outputs' / 'raw' / f'{model}_s6_firstdist_dev.npz'
        firstdist_eval = np.load(fde_p) if fde_p.exists() else {}
        firstdist_dev = np.load(fdd_p) if fdd_p.exists() else {}

        # -------- legacy direct consistency (raw text-level outcomes) --------
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
        conf_pairs = []
        for f in files:
            if f not in legacy_out:
                continue
            n_cmp += 1
            r = ev[(f, 'direct')]
            n_mis += raw_outcome(model, r) != legacy_out[f]
            if r.get('answer_maxp') is not None and f in firstdist_eval:
                v = np.asarray(firstdist_eval[f], dtype=np.float64)
                tok = r['tokens'][0]
                conf_pairs.append(abs(float(v[tok]) - r['answer_maxp']))
        mismatch_rate = n_mis / n_cmp if n_cmp else float('nan')
        tables['consistency'].append({
            'model': model, 'n_compared': n_cmp, 'n_mismatch': n_mis,
            'mismatch_rate': round(mismatch_rate, 5),
            'gate_pass': bool(mismatch_rate <= 0.005),
            'npz_vs_recordd_conf_maxdiff': float(np.max(conf_pairs)) if conf_pairs else float('nan')})
        summary.setdefault(model, {})['direct_consistency'] = {
            'n': n_cmp, 'mismatch': n_mis, 'rate': mismatch_rate}
        summary[model]['files'] = len(files)

        for stratum in ('low_acc', 'high_acc'):
            sfiles = [f for f in files if ev[(f, 'direct')]['stratum'] == stratum]
            d_rows = [ev[(f, 'direct')] for f in sfiles]
            ok_d = [r.get('answer_maxp') is not None for r in d_rows]
            ece_d = ece([r['answer_maxp'] for r in d_rows if r['answer_maxp'] is not None],
                        [r['outcome'] == 'correct' for r in d_rows if r['answer_maxp'] is not None])
            summary.setdefault(model, {}).setdefault(stratum, {})['direct'] = {
                'n': len(sfiles),
                'acc': float(np.mean([r['outcome'] == 'correct' for r in d_rows])),
                'mean_conf': float(np.mean([r['answer_maxp'] for r in d_rows
                                            if r['answer_maxp'] is not None])),
                'ece': ece_d}

            for method in METHODS:
                m_rows, kept = [], []
                for f, dr in zip(sfiles, d_rows):
                    r = ev.get((f, method))
                    if r is None:
                        continue
                    m_rows.append(r)
                    kept.append(dr)
                if not m_rows:
                    continue
                ok = [d.get('answer_maxp') is not None and
                      m.get('answer_maxp') is not None for d, m in zip(kept, m_rows)]
                y_d = np.array([d['outcome'] == 'correct' for d in kept])
                y_m = np.array([m['outcome'] == 'correct' for m in m_rows])
                ece_m = ece([m['answer_maxp'] for m in m_rows if m['answer_maxp'] is not None],
                            [m['outcome'] == 'correct' for m in m_rows if m['answer_maxp'] is not None])
                unchanged = np.array([
                    o and norm_text(m['text']) == norm_text(d['text']) and
                    m['tokens'][0] == d['tokens'][0]
                    for o, d, m in zip(ok, kept, m_rows)])
                dconf_all = np.array([
                    m['answer_maxp'] - d['answer_maxp'] if o else np.nan
                    for o, d, m in zip(ok, kept, m_rows)])
                u = unchanged & ~np.isnan(dconf_all)
                m_ci = boot_mean(dconf_all[u])
                dece_point = ece_m - ece_d
                dece_boots = []
                n = len(m_rows)
                for _ in range(B):
                    idx = rng.randint(0, n, n)
                    dece_boots.append(ece_delta([kept[i] for i in idx],
                                                [m_rows[i] for i in idx]))
                dece_ci = (float(np.percentile(dece_boots, 2.5)),
                           float(np.percentile(dece_boots, 97.5)))
                not_correct_d = ~y_d
                rescue = float(y_m[not_correct_d].mean()) if not_correct_d.any() else float('nan')
                brk = float((~y_m)[y_d].mean()) if y_d.any() else float('nan')

                tables['core'].append({
                    'model': model, 'stratum': stratum, 'method': method,
                    'n': len(m_rows), 'n_with_conf': int(np.sum(ok)),
                    'acc_direct': float(y_d.mean()), 'acc_method': float(y_m.mean()),
                    'abstain_direct': float(np.mean([d['outcome'] == 'abstain' for d in kept])),
                    'abstain_method': float(np.mean([m['outcome'] == 'abstain' for m in m_rows])),
                    'conf_direct': float(np.mean([d['answer_maxp'] for d, o in zip(kept, ok) if o])),
                    'conf_method': float(np.mean([m['answer_maxp'] for m, o in zip(m_rows, ok) if o])),
                    'dconf_all': float(np.nanmean(dconf_all)),
                    'dconf_unchanged': m_ci[0], 'dconf_un_lo': m_ci[1], 'dconf_un_hi': m_ci[2],
                    'n_unchanged': int(np.sum(u)),
                    'ece_direct': ece_d, 'ece_method': ece_m,
                    'dece': dece_point, 'dece_lo': dece_ci[0], 'dece_hi': dece_ci[1],
                    'rescue_rate': rescue, 'break_rate': brk})

                # ---------------- C: four-cell decomposition -------------------
                cells = {'always_right': [], 'always_wrong': [],
                         'corrected': [], 'broken': []}
                cell_n = {k: 0 for k in cells}
                both_wrong = both_wrong_same = 0
                for d, m, o in zip(kept, m_rows, ok):
                    dc, mc = d['outcome'] == 'correct', m['outcome'] == 'correct'
                    if dc and mc:
                        k = 'always_right'
                    elif (not dc) and (not mc):
                        k = 'always_wrong'
                        both_wrong += 1
                        both_wrong_same += norm_text(m['text']) == norm_text(d['text'])
                    elif (not dc) and mc:
                        k = 'corrected'
                    else:
                        k = 'broken'
                    cell_n[k] += 1
                    if o:
                        cells[k].append(m['answer_maxp'] - d['answer_maxp'])
                row_c = {'model': model, 'stratum': stratum, 'method': method,
                         'same_wrong_answer_share':
                             both_wrong_same / both_wrong if both_wrong else float('nan')}
                for cname, incs in cells.items():
                    mean, lo, hi = boot_mean(incs)
                    row_c[f'{cname}_n'] = cell_n[cname]
                    row_c[f'{cname}_n_conf'] = len(incs)
                    row_c[f'{cname}_dconf'] = mean
                    row_c[f'{cname}_lo'] = lo
                    row_c[f'{cname}_hi'] = hi
                keys = list(cells)
                row_c['cells_similar'] = 'yes'
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = cells[keys[i]], cells[keys[j]]
                        if not a or not b:
                            continue
                        aa, bb = np.asarray(a), np.asarray(b)
                        boots = [aa[rng.randint(0, len(a), len(a))].mean() -
                                 bb[rng.randint(0, len(b), len(b))].mean()
                                 for _ in range(B)]
                        lo_, hi_ = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
                        if lo_ > 0 or hi_ < 0:
                            row_c['cells_similar'] = f'no({keys[i]}vs{keys[j]})'
                tables['fourcell'].append(row_c)

                # ---------------- B: sharpening controls + F: offset ----------
                if len(firstdist_eval) and len(firstdist_dev):
                    dev_files = sorted({f for f, m in dv if m == 'direct'})
                    dev_unc = []
                    for f in dev_files:
                        d, r = dv[(f, 'direct')], dv.get((f, method))
                        if (r is None or r.get('answer_maxp') is None or
                                d.get('answer_maxp') is None or
                                f not in firstdist_dev):
                            continue
                        if norm_text(r['text']) == norm_text(d['text']) and \
                                r['tokens'][0] == d['tokens'][0]:
                            dev_unc.append(f)
                    if len(dev_unc) >= 20:
                        target = float(np.mean([
                            dv[(f, method)]['answer_maxp'] -
                            np.float64(firstdist_dev[f][dv[(f, method)]['tokens'][0]])
                            for f in dev_unc]))
                        best_T, best_gap = float('nan'), float('inf')
                        for T in T_GRID:
                            ctrl = np.mean([
                                _temp_conf(firstdist_dev[f], dv[(f, 'direct')]['tokens'][0], T) -
                                np.float64(firstdist_dev[f][dv[(f, 'direct')]['tokens'][0]])
                                for f in dev_unc])
                            gap = abs(ctrl - target)
                            if gap < best_gap:
                                best_T, best_gap = float(T), gap
                        idx_un = [i for i in range(len(m_rows)) if u[i]
                                  and m_rows[i]['file'] in firstdist_eval]
                        fs_un = [m_rows[i]['file'] for i in idx_un]
                        d_m_eval = float(np.nanmean(dconf_all[idx_un]))
                        base_un = np.array([dconf_all[i] for i in idx_un])
                        d_c1 = np.array([
                            _trunc_conf(firstdist_eval[f], kept[i]['tokens'][0]) -
                            np.float64(firstdist_eval[f][kept[i]['tokens'][0]])
                            for i, f in zip(idx_un, fs_un)])
                        d_c2 = np.array([
                            _temp_conf(firstdist_eval[f], kept[i]['tokens'][0], best_T) -
                            np.float64(firstdist_eval[f][kept[i]['tokens'][0]])
                            for i, f in zip(idx_un, fs_un)])
                        rho1 = 1 - abs(d_c1.mean() - d_m_eval) / abs(d_m_eval) if d_m_eval else float('nan')
                        rho2 = 1 - abs(d_c2.mean() - d_m_eval) / abs(d_m_eval) if d_m_eval else float('nan')
                        _, g1lo, g1hi = boot_mean(d_c1 - base_un)
                        _, g2lo, g2hi = boot_mean(d_c2 - base_un)
                        ev_ok = [(r, f) for r, f in zip(d_rows, sfiles)
                                 if f in firstdist_eval and r.get('answer_maxp') is not None]
                        cf1 = [_trunc_conf(firstdist_eval[f], r['tokens'][0]) for r, f in ev_ok]
                        cf2 = [_temp_conf(firstdist_eval[f], r['tokens'][0], best_T) for r, f in ev_ok]
                        yl = [r['outcome'] == 'correct' for r, _ in ev_ok]
                        verdict = ('sharpening' if (rho2 >= 0.8 and d_m_eval > 0 and d_c2.mean() > 0)
                                   else ('structure' if (rho2 < 0.5 or (d_m_eval > 0 and d_c2.mean() <= 0))
                                         else 'partial'))
                        tables['sharpening'].append({
                            'model': model, 'stratum': stratum, 'method': method,
                            'T_fit': best_T, 'n_dev_unchanged': len(dev_unc),
                            'dev_target': target,
                            'dconf_method': d_m_eval,
                            'dconf_ctrl_trunc': float(d_c1.mean()), 'rho_trunc': rho1,
                            'gap_trunc_lo': g1lo, 'gap_trunc_hi': g1hi,
                            'dconf_ctrl_temp': float(d_c2.mean()), 'rho_temp': rho2,
                            'gap_temp_lo': g2lo, 'gap_temp_hi': g2hi,
                            'ece_ctrl_trunc': ece(cf1, yl), 'ece_ctrl_temp': ece(cf2, yl),
                            'ece_method': ece_m, 'ece_direct': ece_d,
                            'verdict': verdict})

                        offset = target
                        conf_rows = [(m['answer_maxp'], m['outcome'] == 'correct')
                                     for m in m_rows if m['answer_maxp'] is not None]
                        cf_corr = np.clip(np.array([c for c, _ in conf_rows]) - offset, 0, 1)
                        ece_after = ece(cf_corr, [y for _, y in conf_rows])
                        tables['correction'].append({
                            'model': model, 'stratum': stratum, 'method': method,
                            'offset_dev': offset,
                            'ece_direct': ece_d, 'ece_before': ece_m,
                            'ece_after': ece_after,
                            'acc_before': float(y_m.mean()), 'acc_after': float(y_m.mean()),
                            'high_recovered': bool(ece_after <= ece_d + 0.01)
                            if stratum == 'high_acc' else ''})

                # ---------------- D: name-level confidence ---------------------
                def name_conf(r):
                    return float(np.prod(r['step_probs'])) if r.get('step_probs') else float('nan')

                wrong = np.array([m['outcome'] in ('wrong', 'wrong_unparsed')
                                  for m in m_rows])
                w_ok = wrong & np.array(ok)
                if w_ok.any():
                    inc_first = np.array([m['answer_maxp'] for m, w in zip(m_rows, w_ok) if w]) - \
                        np.array([d['answer_maxp'] for d, w in zip(kept, w_ok) if w])
                    inc_name = np.array([name_conf(m) for m, w in zip(m_rows, w_ok) if w]) - \
                        np.array([name_conf(d) for d, w in zip(kept, w_ok) if w])
                    fi = boot_mean(inc_first)
                    ni = boot_mean(inc_name)
                    tables['dname'].append({
                        'model': model, 'stratum': stratum, 'method': method,
                        'n_wrong': int(w_ok.sum()),
                        'nameconf_direct': float(np.nanmean(
                            [name_conf(d) for d, w in zip(kept, w_ok) if w])),
                        'nameconf_method': float(np.nanmean(
                            [name_conf(m) for m, w in zip(m_rows, w_ok) if w])),
                        'd_first_token': fi[0], 'd_first_lo': fi[1], 'd_first_hi': fi[2],
                        'd_name_level': ni[0], 'd_name_lo': ni[1], 'd_name_hi': ni[2]})

    # ---------------- E: difference in differences -----------------------------
    core = {(r['model'], r['stratum'], r['method']): r for r in tables['core']}
    for model in models:
        for method in METHODS:
            if (model, 'low_acc', method) not in core or \
                    (model, 'high_acc', method) not in core:
                continue
            lo, hi = core[(model, 'low_acc', method)], core[(model, 'high_acc', method)]
            ev = load(model, 'eval')
            rows = {'low_acc': [], 'high_acc': []}
            for (f, m), r in ev.items():
                if m != method or r.get('answer_maxp') is None:
                    continue
                d = ev.get((f, 'direct'))
                if d is None or d.get('answer_maxp') is None:
                    continue
                rows[r['stratum']].append((d, r))
            did_boots = []
            for _ in range(B):
                b = []
                for st in ('low_acc', 'high_acc'):
                    rs = rows[st]
                    idx = rng.randint(0, len(rs), len(rs))
                    b.append(ece_delta([rs[i][0] for i in idx], [rs[i][1] for i in idx]))
                did_boots.append(b[0] - b[1])
            tables['did'].append({
                'model': model, 'method': method,
                'dece_low': lo['dece'], 'dece_high': hi['dece'],
                'did': lo['dece'] - hi['dece'],
                'did_lo': float(np.percentile(did_boots, 2.5)),
                'did_hi': float(np.percentile(did_boots, 97.5)),
                'did_excludes0': bool(np.percentile(did_boots, 2.5) > 0 or
                                      np.percentile(did_boots, 97.5) < 0),
                'rescue_low': lo['rescue_rate'], 'rescue_high': hi['rescue_rate'],
                'dconf_low': lo['dconf_unchanged'], 'dconf_high': hi['dconf_unchanged']})

    # ---------------- G: dose-response (q4b) ------------------------------------
    dz = load('q4b', 'dose')
    if dz:
        ev = load('q4b', 'eval')
        # alpha=1.0 reuses the experiment-A VCD records (same eval half)
        for (f, m), r in list(ev.items()):
            if m == 'vcd':
                dz[(f, 'vcd')] = r
        for stratum in ('low_acc', 'high_acc'):
            for tag, alpha in DOSE_ALPHAS.items():
                pairs = [(ev[(f, 'direct')], dz[(f, tag)]) for f in
                         sorted({ff for (ff, m) in dz if m == tag})
                         if (f, 'direct') in ev
                         and dz[(f, tag)]['stratum'] == stratum
                         and dz[(f, tag)].get('answer_maxp') is not None
                         and ev[(f, 'direct')].get('answer_maxp') is not None]
                if not pairs:
                    continue
                y_d = [d['outcome'] == 'correct' for d, _ in pairs]
                y_m = [r['outcome'] == 'correct' for _, r in pairs]
                c_d = [d['answer_maxp'] for d, _ in pairs]
                c_m = [r['answer_maxp'] for _, r in pairs]
                tables['dose'].append({
                    'model': 'q4b', 'stratum': stratum, 'alpha': alpha,
                    'n': len(pairs), 'acc_direct': float(np.mean(y_d)),
                    'acc_method': float(np.mean(y_m)),
                    'dconf_all': float(np.mean(np.array(c_m) - np.array(c_d))),
                    'ece_direct': ece(c_d, y_d), 'ece_method': ece(c_m, y_m),
                    'dece': ece(c_m, y_m) - ece(c_d, y_d)})
        per_alpha = {}
        for (f, m), r in dz.items():
            if r.get('answer_maxp') is None:
                continue
            d = ev.get((f, 'direct'))
            if d is None or d.get('answer_maxp') is None:
                continue
            per_alpha.setdefault(m, []).append((d, r))
        for tag, alpha in DOSE_ALPHAS.items():
            by_class = {}
            for d, r in per_alpha.get(tag, []):
                by_class.setdefault(r['class'], []).append((d, r))
            xs, ys = [], []
            for cls, prs in by_class.items():
                if len(prs) < 8:
                    continue
                xs.append(np.mean([d['outcome'] == 'correct' for d, _ in prs]))
                ys.append(ece_delta([d for d, _ in prs], [r for _, r in prs]))
            if len(xs) >= 10:
                b1, b0 = np.polyfit(xs, ys, 1)
                if b1 < 0:
                    cross = -b0 / b1
                    boots = []
                    xa, ya = np.array(xs), np.array(ys)
                    for _ in range(B):
                        sel = rng.randint(0, len(xa), len(xa))
                        if len(set(xa[sel])) < 3:
                            continue
                        bb1, bb0 = np.polyfit(xa[sel], ya[sel], 1)
                        if bb1 < 0:
                            boots.append(-bb0 / bb1)
                    ci = (float(np.percentile(boots, 2.5)),
                          float(np.percentile(boots, 97.5))) if boots else (float('nan'),) * 2
                else:
                    cross, ci = float('nan'), (float('nan'), float('nan'))
                tables['dose'].append({'model': 'q4b_class_crossing', 'stratum': 'both',
                                       'alpha': alpha, 'n': len(xs),
                                       'crossing': cross, 'cross_lo': ci[0], 'cross_hi': ci[1]})

    # ---------------- variant supplement ---------------------------------------
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
            if r.get('task') == 'naming':
                legacy.setdefault(r.get('method'), {})[r['file']] = r
        ev = load(model, 'eval')
        for var, faithful in LEGACY_MAP.items():
            for stratum in ('low_acc', 'high_acc'):
                old_st = 'deficient' if stratum == 'low_acc' else 'known'
                rows = []
                for (f, m), r in ev.items():
                    if m != faithful:
                        continue
                    d = ev.get((f, 'direct'))
                    o = legacy.get(var, {}).get(f)
                    od = legacy.get('direct', {}).get(f)
                    if d is None or o is None or od is None:
                        continue
                    if r['stratum'] == stratum and o.get('stratum') == old_st:
                        rows.append((d, r, od, o))
                if len(rows) < 100:
                    continue
                if model in ROUND_A:
                    rows = [(d, r, od, o) for (d, r, od, o) in rows
                            if not (r.get('abstained') or d.get('abstained') or
                                    o.get('abstained') or od.get('abstained'))]
                if len(rows) < 100:
                    continue
                dece_old = ece([o['answer_maxp'] for _, _, _, o in rows],
                               [o['outcome'] == 'correct' for _, _, _, o in rows]) - \
                    ece([od['answer_maxp'] for _, _, od, _ in rows],
                        [od['outcome'] == 'correct' for _, _, od, _ in rows])
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

    # ---------------- write -----------------------------------------------------
    outdir = ROOT / 'outputs' / 'tables'
    outdir.mkdir(exist_ok=True)
    import csv
    for name, rows2 in tables.items():
        if not rows2:
            continue
        keys = sorted({k for r in rows2 for k in r})
        with open(outdir / f'stage6_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows2)
        print(f'stage6_{name}.csv: {len(rows2)} rows')
    summary['tables'] = {k: len(v) for k, v in tables.items() if v}
    json.dump(summary, open(outdir / 'stage6_summary.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print('stage6_summary.json written')


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
