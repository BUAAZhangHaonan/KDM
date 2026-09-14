"""Stage-5 base-rate relation analysis (pre-registered in PREREGISTER_STAGE5.md).

口径一 (class-set populations): per class points -> Pearson r(ΔECE, gap),
r(ΔECE, acc) with class-cluster bootstrap (B=2000); OLS crossing point with
bootstrap CI; stratum aggregates + 8 equal-count display bins.
口径二 (mixed populations): alpha-grid mixtures of known/deficient eval
samples, 200 draws each, sizes fixed at 600 per stratum component.

Outputs: outputs/tables/baserate.csv, outputs/tables/crossing.csv,
outputs/tables/stage5_summary.json

Usage: ./venv/bin/python code/stage5_baserate.py
"""
import os, sys, json, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'mpl'))
import scoring

B = 2000
METHODS = ['vcd', 'mib', 'lcd']
N_BINS = 8
MIX_SIZE = 600
N_MIX_DRAWS = 200
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
LABEL = {'deficient': 'low_acc', 'known': 'high_acc', 'middle': 'middle'}

PAIRS = [
    ('q4b', 'food101', 'outputs/raw/q4b_main_naming.jsonl', 'outputs/raw/q4b_s5middle_naming.jsonl'),
    ('q9b', 'food101', 'outputs/raw/q9b_main_naming.jsonl', 'outputs/raw/q9b_s5middle_naming.jsonl'),
    ('q4b', 'dogs', 'outputs/raw/q4b_dogs_main.jsonl', 'outputs/raw/q4b_s5middle_dogs.jsonl'),
    ('q9b', 'dogs', 'outputs/raw/q9b_dogs_main.jsonl', 'outputs/raw/q9b_s5middle_dogs.jsonl'),
]


def ece(confs, corrects):
    return scoring.calibration_error(list(confs), list(corrects), 10)


def load_pair(main_path, mid_path):
    """-> by_class[class][method] = {file: record}, stratum_of[class]"""
    by_class = defaultdict(lambda: defaultdict(dict))
    stratum_of = {}
    for path in (main_path, mid_path):
        p = ROOT / path
        if not p.exists():
            return None
        for line in open(p):
            d = json.loads(line)
            if d.get('task') != 'naming' or d['method'] not in ('direct',) + tuple(METHODS):
                continue
            if d.get('answer_maxp') is None or 'outcome' not in d:
                continue
            st = LABEL[d['stratum']]
            stratum_of[d['class']] = st
            by_class[d['class']][d['method']][d['file']] = d
    return by_class, stratum_of


def class_point(by_class_class, m):
    """paired arrays for direct and method m -> dict or None if unpaired"""
    dmap, mmap = by_class_class['direct'], by_class_class[m]
    files = sorted(set(dmap) & set(mmap))
    if len(files) < 10:
        return None
    cd = np.array([dmap[f]['answer_maxp'] for f in files])
    kd = np.array([1.0 * (dmap[f]['outcome'] == 'correct') for f in files])
    cm = np.array([mmap[f]['answer_maxp'] for f in files])
    km = np.array([1.0 * (mmap[f]['outcome'] == 'correct') for f in files])
    return {'n': len(files),
            'acc': float(kd.mean()), 'gap': float(cd.mean() - kd.mean()),
            'ece_d': ece(cd, kd), 'ece_m': ece(cm, km),
            'd_ece': ece(cm, km) - ece(cd, kd)}


def pooled(records):
    cd = np.array([r['answer_maxp'] for r in records['direct']])
    kd = np.array([1.0 * (r['outcome'] == 'correct') for r in records['direct']])
    return float(cd.mean()), float(kd.mean()), ece(cd, kd)


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def analyse(model, domain, by_class, stratum_of, base_rows, cross_rows, summary):
    # -------- 口径一: class points --------
    pts = {}   # method -> list of (class, acc, gap, d_ece)
    shared = None
    for c in sorted(by_class):
        ok = all(m in by_class[c] for m in ['direct'] + METHODS)
        if not ok:
            continue
        cs = set()
        for m in ['direct'] + METHODS:
            r = class_point(by_class[c], m)
            if r is None:
                cs.add(m)
        # class kept if every method has a valid paired point; x values shared
        if cs:
            continue
        shared = shared or set()
        for m in METHODS:
            r = class_point(by_class[c], m)
            pts.setdefault(m, []).append((c, r['acc'], r['gap'], r['d_ece'], r['n'],
                                          r['ece_d'], r['ece_m']))
        shared.add(c)
    n_cls = len(shared)
    summary['n_classes'][f'{model}@{domain}'] = n_cls

    for m in METHODS:
        for (c, acc, gap, de, n, ed, em) in pts[m]:
            base_rows.append({'population': 'class', 'model': model, 'domain': domain,
                              'group': c, 'method': m, 'n': n,
                              'acc_direct': round(acc, 4), 'gap_direct': round(gap, 4),
                              'ece_direct': round(ed, 4), 'ece_method': round(em, 4),
                              'd_ece': round(de, 4)})

    # -------- correlations + crossing (cluster bootstrap over classes) --------
    for m in METHODS:
        arr = pts[m]
        for reg, pred_sign in (('acc', -1), ('gap', +1)):
            x = np.array([a[1] if reg == 'acc' else a[2] for a in arr])
            y = np.array([a[3] for a in arr])
            r0 = pearson(x, y)
            rng = np.random.default_rng(0)
            rs, x0s = [], []
            for _ in range(B):
                idx = rng.integers(0, len(x), len(x))
                rb = pearson(x[idx], y[idx])
                if rb == rb:
                    rs.append(rb)
                b_, a_ = np.polyfit(x[idx], y[idx], 1)
                if (pred_sign < 0 and b_ < 0) or (pred_sign > 0 and b_ > 0):
                    x0s.append(-a_ / b_)
            r_lo, r_hi = (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (float('nan'),) * 2
            b0, a0 = np.polyfit(x, y, 1)
            supported = (r0 == r0) and (pred_sign * r0 > 0) and (pred_sign * r_lo > 0)
            row = {'model': model, 'domain': domain, 'method': m, 'regressor': reg,
                   'n_classes': n_cls, 'r': round(r0, 4),
                   'r_lo': round(float(r_lo), 4), 'r_hi': round(float(r_hi), 4),
                   'r2': round(r0 * r0, 4) if r0 == r0 else '',
                   'slope': round(float(b0), 4), 'intercept': round(float(a0), 4),
                   'x0': round(float(-a0 / b0), 4) if (pred_sign < 0 and b0 < 0) or (pred_sign > 0 and b0 > 0) else '',
                   'x0_lo': round(float(np.percentile(x0s, 2.5)), 4) if len(x0s) > 50 else '',
                   'x0_hi': round(float(np.percentile(x0s, 97.5)), 4) if len(x0s) > 50 else '',
                   'boot_valid_frac': round(len(x0s) / B, 3),
                   'direction_supported': supported}
            cross_rows.append(row)
            summary['direction'][f'{model}@{domain}|{m}|{reg}'] = bool(supported)

    # -------- stratum aggregates --------
    for st in ('low_acc', 'middle', 'high_acc'):
        cls = [c for c in shared if stratum_of[c] == st]
        if not cls:
            continue
        recs = defaultdict(dict)
        for c in cls:
            for f in by_class[c]['direct']:
                for m in ['direct'] + METHODS:
                    if f in by_class[c][m]:
                        recs[m][f] = by_class[c][m][f]
        conf, acc, ece_d = pooled(recs)
        for m in METHODS:
            cm = np.array([r['answer_maxp'] for r in recs[m].values()])
            km = np.array([1.0 * (r['outcome'] == 'correct') for r in recs[m].values()])
            em_ = ece(cm, km)
            base_rows.append({'population': 'stratum', 'model': model, 'domain': domain,
                              'group': st, 'method': m, 'n': len(recs['direct']),
                              'acc_direct': round(acc, 4), 'gap_direct': round(conf - acc, 4),
                              'ece_direct': round(ece_d, 4), 'ece_method': round(em_, 4),
                              'd_ece': round(em_ - ece_d, 4)})

    # -------- display bins (8 equal-count by class acc) --------
    order = sorted(pts[METHODS[0]], key=lambda a: a[1])
    for i in range(N_BINS):
        chunk = order[i * len(order) // N_BINS:(i + 1) * len(order) // N_BINS]
        if not chunk:
            continue
        cls = [a[0] for a in chunk]
        recs = defaultdict(dict)
        for c in cls:
            for f in by_class[c]['direct']:
                for m in ['direct'] + METHODS:
                    if f in by_class[c][m]:
                        recs[m][f] = by_class[c][m][f]
        conf, acc, ece_d = pooled(recs)
        for m in METHODS:
            cm = np.array([r['answer_maxp'] for r in recs[m].values()])
            km = np.array([1.0 * (r['outcome'] == 'correct') for r in recs[m].values()])
            em_ = ece(cm, km)
            base_rows.append({'population': 'bin', 'model': model, 'domain': domain,
                              'group': f'bin{i+1}', 'method': m, 'n': len(recs['direct']),
                              'acc_direct': round(acc, 4), 'gap_direct': round(conf - acc, 4),
                              'ece_direct': round(ece_d, 4), 'ece_method': round(em_, 4),
                              'd_ece': round(em_ - ece_d, 4)})

    # -------- 口径二: mixtures --------
    flat = {m: {} for m in ['direct'] + METHODS}
    stratum_of_file = {}
    for c in shared:
        for m in ['direct'] + METHODS:
            flat[m].update(by_class[c][m])
        for f in by_class[c]['direct']:
            stratum_of_file[f] = stratum_of[c]
    usable = [f for f in stratum_of_file
              if all(f in flat[m] for m in ['direct'] + METHODS)]
    pools = {'low_acc': sorted(f for f in usable if stratum_of_file[f] == 'low_acc'),
             'high_acc': sorted(f for f in usable if stratum_of_file[f] == 'high_acc')}
    n_lo, n_hi = len(pools['low_acc']), len(pools['high_acc'])
    size = min(MIX_SIZE, n_lo, n_hi)
    rng = np.random.default_rng(7)
    for alpha in ALPHAS:
        k = round(alpha * size)
        accs, gaps, des = [], [], {m: [] for m in METHODS}
        for _ in range(N_MIX_DRAWS):
            files = list(rng.choice(pools['low_acc'], size - k, replace=False) if size - k > 0 else []) \
                  + list(rng.choice(pools['high_acc'], k, replace=False) if k > 0 else [])
            cd = np.array([flat['direct'][f]['answer_maxp'] for f in files])
            kd = np.array([1.0 * (flat['direct'][f]['outcome'] == 'correct') for f in files])
            accs.append(kd.mean()); gaps.append(cd.mean() - kd.mean())
            for m in METHODS:
                cm = np.array([flat[m][f]['answer_maxp'] for f in files])
                km = np.array([1.0 * (flat[m][f]['outcome'] == 'correct') for f in files])
                des[m].append(ece(cm, km) - ece(cd, kd))
        for m in METHODS:
            base_rows.append({'population': 'mixture', 'model': model, 'domain': domain,
                              'group': f'alpha={alpha}', 'method': m, 'n': size,
                              'acc_direct': round(float(np.mean(accs)), 4),
                              'gap_direct': round(float(np.mean(gaps)), 4),
                              'ece_direct': '', 'ece_method': '',
                              'd_ece': round(float(np.mean(des[m])), 4)})


def main():
    base_rows, cross_rows = [], []
    summary = {'n_classes': {}, 'direction': {}}
    used = []
    for model, domain, main_path, mid_path in PAIRS:
        loaded = load_pair(main_path, mid_path)
        if loaded is None:
            print(f'skip {model}@{domain} (missing middle records)', flush=True)
            continue
        by_class, stratum_of = loaded
        analyse(model, domain, by_class, stratum_of, base_rows, cross_rows, summary)
        used.append(f'{model}@{domain}')
        print(f'[{model}@{domain}] analysed', flush=True)

    fields = ['population', 'model', 'domain', 'group', 'method', 'n', 'acc_direct',
              'gap_direct', 'ece_direct', 'ece_method', 'd_ece']
    with open(ROOT / 'outputs' / 'tables' / 'baserate.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(base_rows)

    cfields = ['model', 'domain', 'method', 'regressor', 'n_classes', 'r', 'r_lo', 'r_hi',
               'r2', 'slope', 'intercept', 'x0', 'x0_lo', 'x0_hi', 'boot_valid_frac',
               'direction_supported']
    with open(ROOT / 'outputs' / 'tables' / 'crossing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        w.writerows(cross_rows)

    summary['pairs'] = used
    json.dump(summary, open(ROOT / 'outputs' / 'tables' / 'stage5_summary.json', 'w'),
              indent=1, ensure_ascii=False)
    print('wrote baserate.csv (%d rows), crossing.csv (%d rows)' % (len(base_rows), len(cross_rows)))


if __name__ == '__main__':
    main()
