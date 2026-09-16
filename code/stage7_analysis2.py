"""Stage-7 part 2 analyses: fidelity close-out tables.

  1. seed robustness (VCD noise salts s2/s3 vs the stage-6 base draw)
  2. DoLa official-subset candidate layers vs the all-layers caliber
  3. dogs direction prediction check (faithful reruns, all cells predicted
     dECE >= 0; legacy round-A abstention applied for q4b)
  4. SID table on llava16 (faithful reimplementation)
  5. M3ID gate-inactivity ratios (already written per model; merged here)
Outputs: outputs/tables/stage7_{seedrobust, dolasubset, dogspredict, sid}.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from stage6_analysis import load, ece, norm_text, ece_delta  # noqa: E402

B, SEED = 2000, 20260916
rng = np.random.RandomState(SEED)
LEGACY_DOGS = {'q4b': 'outputs/raw/q4b_dogs_main.jsonl',
               'q9b': 'outputs/raw/q9b_dogs_main.jsonl'}


def load7(model, part):
    recs = {}
    p = ROOT / 'outputs' / 'raw' / f'{model}_s7_{part}_naming.jsonl'
    if not p.exists():
        return recs
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'error' in r:
            continue
        if model == 'q4b' and r.get('abstained'):
            r['outcome'] = 'abstain'
            r['answer_maxp'] = None
        recs[(r['file'], r['method'])] = r
    return recs


def cell_stats(d_rows, m_rows):
    ok = [d.get('answer_maxp') is not None and m.get('answer_maxp') is not None
          for d, m in zip(d_rows, m_rows)]
    unchanged = np.array([
        o and norm_text(m['text']) == norm_text(d['text']) and
        m['tokens'][0] == d['tokens'][0]
        for o, d, m in zip(ok, d_rows, m_rows)])
    incs = np.array([m['answer_maxp'] - d['answer_maxp']
                     if u else np.nan
                     for u, d, m in zip(unchanged, d_rows, m_rows)])
    y_d = np.array([d['outcome'] == 'correct' for d in d_rows])
    y_m = np.array([m['outcome'] == 'correct' for m in m_rows])
    cd = [d['answer_maxp'] for d, o in zip(d_rows, ok) if o]
    yd = [d['outcome'] == 'correct' for d, o in zip(d_rows, ok) if o]
    cm = [m['answer_maxp'] for m, o in zip(m_rows, ok) if o]
    ym = [m['outcome'] == 'correct' for m, o in zip(m_rows, ok) if o]
    return {'n': len(m_rows), 'n_unchanged': int(np.nansum(~np.isnan(incs))),
            'dconf': float(np.nanmean(incs)),
            'acc_d': float(y_d.mean()), 'acc_m': float(y_m.mean()),
            'ece_d': ece(cd, yd), 'ece_m': ece(cm, ym),
            'dece': ece(cm, ym) - ece(cd, yd)}


def paired(mrecs, drecs, stratum, method):
    files = sorted({f for f, m in mrecs if m == method
                    and mrecs[(f, method)]['stratum'] == stratum
                    and f in {ff for ff, mm in drecs if mm == 'direct'}})

    def get(f):
        d = drecs.get((f, 'direct'))
        m = mrecs.get((f, method))
        if d is None or m is None:
            return None
        return d, m
    pairs = [p for p in (get(f) for f in files) if p is not None]
    if not pairs:
        return None
    return cell_stats([d for d, _ in pairs], [m for _, m in pairs])


def main():
    ev6 = {m: load(m, 'eval') for m in ('q4b', 'q9b', 'llava16', 'internvl4b')}
    rows_seed, rows_dola, rows_dogs, rows_sid = [], [], [], []

    # ---------- 1. seed robustness ------------------------------------------
    for model in ('q4b', 'q9b'):
        s7 = load7(model, 'seed')
        for tag in ('vcd', 'vcd_s2', 'vcd_s3'):
            src = ev6[model] if tag == 'vcd' else s7
            for st in ('low_acc', 'high_acc'):
                r = paired(src, ev6[model], st, tag)
                if r:
                    rows_seed.append({'model': model, 'stratum': st, 'draw': tag, **r})
    for model in ('q4b', 'q9b'):
        for st in ('low_acc', 'high_acc'):
            vals = [r['dconf'] for r in rows_seed
                    if r['model'] == model and r['stratum'] == st]
            des = [r['dece'] for r in rows_seed
                   if r['model'] == model and r['stratum'] == st]
            base = vals[0]
            spread = max(vals) - min(vals)
            for r in rows_seed:
                if r['model'] == model and r['stratum'] == st:
                    r['spread_dconf'] = spread
                    r['spread_over_base'] = spread / abs(base) if base else float('nan')
                    r['dece_range'] = f'[{min(des):+.3f},{max(des):+.3f}]'
                    r['exceeds_20pct'] = bool(spread > 0.2 * abs(base)) if base else None

    # ---------- 2. DoLa official-subset caliber ------------------------------
    for model in ('q4b', 'q9b'):
        s7 = load7(model, 'dola_subset')
        for st in ('low_acc', 'high_acc'):
            r_sub = paired(s7, ev6[model], st, 'dola_subset')
            r_full = paired(ev6[model], ev6[model], st, 'dola')
            if r_sub:
                sel = [l for (f, m), rec in s7.items() if m == 'dola_subset'
                       and rec['stratum'] == st for l in rec.get('layers_selected', [])]
                from collections import Counter
                top = Counter(sel).most_common(5)
                rows_dola.append({'model': model, 'stratum': st,
                                  'layers_top': ';'.join(f'{k}:{v}' for k, v in top),
                                  **{f'subset_{k}': v for k, v in r_sub.items()},
                                  'full_dconf': r_full['dconf'] if r_full else float('nan'),
                                  'full_dece': r_full['dece'] if r_full else float('nan')})
        # subset DiD
        sub = {r['stratum']: r for r in rows_dola
               if r['model'] == model and 'subset_dece' in r}
        if 'low_acc' in sub and 'high_acc' in sub:
            did = sub['low_acc']['subset_dece'] - sub['high_acc']['subset_dece']
            for r in rows_dola:
                if r['model'] == model:
                    r['subset_did'] = did

    # ---------- 3. dogs direction prediction ---------------------------------
    hits = misses = 0
    for model in ('q4b', 'q9b'):
        s7 = load7(model, 'dogs')
        # legacy direct + round-A (inputs unchanged)
        drecs, ra = {}, {}
        for line in open(ROOT / LEGACY_DOGS[model]):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task') != 'naming':
                continue
            if r.get('method') == 'direct':
                drecs[(r['file'], 'direct')] = r
            elif r.get('method') == 'roundA':
                ra[r['file']] = bool(r.get('abstained'))
        if model == 'q4b':  # apply the round-A convention to reused direct rows
            for k, r in drecs.items():
                if ra.get(k[0]):
                    r['outcome'] = 'abstain'
                    r['answer_maxp'] = None
        for st in ('low_acc', 'high_acc'):
            for method in ('vcd', 'm3id', 'dola', 'deco'):
                r = paired(s7, drecs, st, method)
                if r is None:
                    continue
                hit = r['dece'] >= 0
                hits += hit
                misses += not hit
                rows_dogs.append({'model': model, 'stratum': st, 'method': method,
                                  'predicted': 'dECE >= 0 (harmed side)',
                                  'hit': bool(hit), **r})
    # ---------- 4. SID on llava16 ---------------------------------------------
    s7 = load7('llava16', 'sid')
    for st in ('low_acc', 'high_acc'):
        r = paired(s7, ev6['llava16'], st, 'sid')
        if r:
            rows_sid.append({'model': 'llava16', 'stratum': st,
                             'impl': 'faithful reimplementation from official code',
                             **r})
    if len(rows_sid) == 2:
        did = rows_sid[0]['dece'] - rows_sid[1]['dece']
        for r in rows_sid:
            r['did'] = did

    outdir = ROOT / 'outputs' / 'tables'
    for name, rows in [('seedrobust', rows_seed), ('dolasubset', rows_dola),
                       ('dogspredict', rows_dogs), ('sid', rows_sid)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(outdir / f'stage7_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'stage7_{name}.csv: {len(rows)} rows')
    print(f'dogs prediction: hits={hits} misses={misses}')


if __name__ == '__main__':
    main()
