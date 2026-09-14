"""Stage-5 re-analysis: main-line quantities for non-Qwen families (no new inference).

From existing per-sample naming records, per (model, stratum, method):
  acc/gap of direct (pre-intervention state), ECE of direct and method,
  dECE with paired sample bootstrap CI, paired wrong-answer confidence change;
plus difference-in-differences rows (low_acc minus high_acc).

Strata renaming (stage 5): deficient -> low_acc, known -> high_acc.  The low
stratum is "low naming accuracy" and is NOT equivalent to "model does not know
the name" (stage 4 showed name accessibility must be stated at a granularity).

Usage: ./venv/bin/python code/stage5_nonqwen.py
"""
import os, sys, json, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'mpl'))
import scoring

B = 2000
METHODS = ['vcd', 'mib', 'lcd']
STRATUM_LABEL = {'deficient': 'low_acc', 'known': 'high_acc'}
FAMILY = {'q4b': 'qwen35', 'q9b': 'qwen35', 'q3vl4b': 'qwen_vl',
          'llava16': 'llava', 'internvl4b': 'internvl'}
SOURCES = {
    'q4b': 'outputs/raw/q4b_main_naming.jsonl',
    'q9b': 'outputs/raw/q9b_main_naming.jsonl',
    'q3vl4b': 'outputs/raw/q3vl4b_food101_main.jsonl',
    'llava16': 'outputs/raw/llava16_food101_main.jsonl',
    'internvl4b': 'outputs/raw/internvl4b_food101_main.jsonl',
}


def ece(confs, corrects):
    return scoring.calibration_error(confs, corrects, 10)


def load_pair(path):
    """by_sid[stratum][file][method] -> record (naming, 4 methods only)."""
    by_sid = defaultdict(lambda: defaultdict(dict))
    for line in open(ROOT / path):
        d = json.loads(line)
        if d.get('task') != 'naming' or d['method'] not in ('direct',) + tuple(METHODS):
            continue
        if d.get('answer_maxp') is None:
            continue
        by_sid[d['stratum']][d['file']][d['method']] = d
    return by_sid


def d_ece_of(sub_m, sub_d):
    cm = np.array([r['answer_maxp'] for r in sub_m])
    km = np.array([1.0 * (r['outcome'] == 'correct') for r in sub_m])
    cd = np.array([r['answer_maxp'] for r in sub_d])
    kd = np.array([1.0 * (r['outcome'] == 'correct') for r in sub_d])
    return ece(cm, km) - ece(cd, kd)


def wrong_pairs(sids, by_file, m):
    out = []
    for s in sids:
        d_r, m_r = by_file[s]['direct'], by_file[s][m]
        if d_r['outcome'] != 'correct' and m_r['outcome'] != 'correct':
            out.append(m_r['answer_maxp'] - d_r['answer_maxp'])
    return out


def main():
    rows = []
    for model, path in SOURCES.items():
        by_sid = load_pair(path)
        did_store = {}
        for st_old in ('deficient', 'known'):
            st = STRATUM_LABEL[st_old]
            by_file = by_sid[st_old]
            sids = sorted(by_file)
            sub_d_all = [by_file[s]['direct'] for s in sids]
            for m in METHODS:
                sub_m_all = [by_file[s][m] for s in sids]
                n = len(sids)
                acc_d = np.mean([1.0 * (r['outcome'] == 'correct') for r in sub_d_all])
                acc_m = np.mean([1.0 * (r['outcome'] == 'correct') for r in sub_m_all])
                gap_d = np.mean([r['answer_maxp'] for r in sub_d_all]) - acc_d
                ece_d = ece(np.array([r['answer_maxp'] for r in sub_d_all]),
                            np.array([1.0 * (r['outcome'] == 'correct') for r in sub_d_all]))
                ece_m = ece(np.array([r['answer_maxp'] for r in sub_m_all]),
                            np.array([1.0 * (r['outcome'] == 'correct') for r in sub_m_all]))
                rng = np.random.default_rng(0)
                vals = [d_ece_of([by_file[sids[i]][m] for i in idx],
                                 [by_file[sids[i]]['direct'] for i in idx])
                        for idx in (rng.integers(0, n, n) for _ in range(B))]
                lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
                wp = wrong_pairs(sids, by_file, m)
                wc = float(np.mean(wp)) if wp else float('nan')
                if wp:
                    arr = np.array(wp)
                    vals2 = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(B)]
                    wlo, whi = np.percentile(vals2, 2.5), np.percentile(vals2, 97.5)
                else:
                    wlo = whi = float('nan')
                rows.append({'model': model, 'family': FAMILY[model], 'stratum': st,
                             'method': m, 'n': n,
                             'acc_direct': round(acc_d, 4), 'acc_method': round(acc_m, 4),
                             'gap_direct': round(gap_d, 4),
                             'ece_direct': round(ece_d, 4), 'ece_method': round(ece_m, 4),
                             'd_ece': round(ece_m - ece_d, 4),
                             'd_ece_lo': round(float(lo), 4), 'd_ece_hi': round(float(hi), 4),
                             'd_wrong_conf': round(wc, 4) if wp else '',
                             'd_wrong_conf_lo': round(float(wlo), 4) if wp else '',
                             'd_wrong_conf_hi': round(float(whi), 4) if wp else '',
                             'n_wrong_pairs': len(wp)})
                did_store.setdefault(m, {})[st] = (n, sids, by_file)
        # DiD: resample each stratum independently, take dECE difference
        for m in METHODS:
            n_lo, sids_lo, bf_lo = did_store[m]['low_acc']
            n_hi, sids_hi, bf_hi = did_store[m]['high_acc']
            rng = np.random.default_rng(1)
            diffs = []
            for _ in range(B):
                idx_lo = rng.integers(0, n_lo, n_lo)
                idx_hi = rng.integers(0, n_hi, n_hi)
                e_lo = d_ece_of([bf_lo[sids_lo[i]][m] for i in idx_lo],
                                [bf_lo[sids_lo[i]]['direct'] for i in idx_lo])
                e_hi = d_ece_of([bf_hi[sids_hi[i]][m] for i in idx_hi],
                                [bf_hi[sids_hi[i]]['direct'] for i in idx_hi])
                diffs.append(e_lo - e_hi)
            wp_lo = wrong_pairs(sids_lo, bf_lo, m)
            wp_hi = wrong_pairs(sids_hi, bf_hi, m)
            dw = (np.mean(wp_lo) - np.mean(wp_hi)) if (wp_lo and wp_hi) else float('nan')
            lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            rows.append({'model': model, 'family': FAMILY[model], 'stratum': 'did',
                         'method': m, 'n': min(n_lo, n_hi),
                         'did_ece': round(float(np.mean(diffs)), 4),
                         'did_ece_lo': round(float(lo), 4), 'did_ece_hi': round(float(hi), 4),
                         'did_wrong_conf': round(float(dw), 4) if dw == dw else ''})
        print(f'[{model}] done', flush=True)

    out = ROOT / 'outputs' / 'tables' / 'nonqwen_core.csv'
    fields = ['model', 'family', 'stratum', 'method', 'n', 'acc_direct', 'acc_method',
              'gap_direct', 'ece_direct', 'ece_method', 'd_ece', 'd_ece_lo', 'd_ece_hi',
              'd_wrong_conf', 'd_wrong_conf_lo', 'd_wrong_conf_hi', 'n_wrong_pairs',
              'did_ece', 'did_ece_lo', 'did_ece_hi', 'did_wrong_conf']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote', out, len(rows), 'rows')


if __name__ == '__main__':
    main()
