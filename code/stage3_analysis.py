"""Stage 3 analysis: four-cell decomposition and judgments 1-4, over all
(model, domain) pairs with completed data.

Pair naming:
  legacy  : q4b/q9b on food101 -> {m}_main_naming.jsonl + {m}_stage3_closedset.jsonl
  std     : extended models    -> {key}_{domain}_main.jsonl + {key}_{domain}_closedset.jsonl
Verdicts per pair use the SAME pre-registered thresholds (PREREGISTER_STAGE3.md
section 7); the primary verdicts (the ones the assertions stand on) are the two
legacy pairs; extended pairs are replication evidence, reported side by side.
"""
import os, sys, json, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np

DELTA = 0.05
AUC_LIMIT = 0.70
J1_SHARE, J1_CI_LO, J1_MIN_CLASSES = 0.20, 0.10, 7
METHODS3 = ['vcd', 'mib', 'lcd']
PRIMARY = ['q4b', 'q9b']          # legacy pairs carry the assertions


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, c - h)
    return (p, lo, min(1.0, c + h + (0.0 if k == n else 0.0)))


def prop_diff(k1, n1, k2, n2):
    p1, p2 = k1 / max(1, n1), k2 / max(1, n2)
    se = math.sqrt(p1 * (1 - p1) / max(1, n1) + p2 * (1 - p2) / max(1, n2))
    return p1 - p2, (p1 - p2) - 1.96 * se, (p1 - p2) + 1.96 * se


def auc_mw(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    order = np.concatenate([pos, neg])
    ranks = np.empty(len(order))
    srt = np.argsort(order, kind='mergesort')
    sorted_v = order[srt]
    r = np.arange(1, len(order) + 1, dtype=float)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[srt] = r
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def tiny_logistic_auc(X, y, iters=300, lr=0.5):
    X, y = np.asarray(X, float), np.asarray(y, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y))
        b -= lr * (p - y).mean()
    s = X @ w + b
    return auc_mw(s[y == 1], s[y == 0])


def ece(confs, corrects, n_bins=10):
    c, k = np.asarray(confs, float), np.asarray(corrects, float)
    if len(c) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (c > edges[i]) & (c <= edges[i + 1]) if i else (c >= 0) & (c <= edges[1])
        if m.sum():
            tot += m.mean() * abs(k[m].mean() - c[m].mean())
    return float(tot)


def discover_pairs():
    pairs = []
    for m in PRIMARY:
        if (ROOT / 'outputs' / 'raw' / f'{m}_main_naming.jsonl').exists() and \
           (ROOT / 'outputs' / 'raw' / f'{m}_stage3_closedset.jsonl').exists():
            pairs.append({'label': m, 'domain': 'food101', 'scheme': 'legacy',
                          'naming': f'{m}_main_naming.jsonl',
                          'closed': f'{m}_stage3_closedset.jsonl'})
    raw = ROOT / 'outputs' / 'raw'
    for p in sorted(raw.glob('*_main.jsonl')):
        tag = p.name[:-len('_main.jsonl')]            # e.g. q3vl4b_food101
        if '_' not in tag:
            continue
        key, domain = tag.rsplit('_', 1)
        if (raw / f'{tag}_closedset.jsonl').exists():
            pairs.append({'label': key, 'domain': domain, 'scheme': 'std',
                          'naming': f'{tag}_main.jsonl',
                          'closed': f'{tag}_closedset.jsonl'})
    return pairs


def load_pair(pair):
    S = {}
    for line in open(ROOT / 'outputs' / 'raw' / pair['naming']):
        r = json.loads(line)
        if r.get('task') != 'naming' or r.get('unavailable') or r.get('error'):
            continue
        d = S.setdefault(r['file'], {'stratum': r['stratum'], 'class': r['class'],
                                     'open': {}, 'signals': None, 'closed': {}})
        if r['method'] == 'roundA':
            d['roundA_abstain'] = bool(r.get('abstained'))
        else:
            d['open'][r['method']] = (r['outcome'], r.get('answer_maxp'))
        if r['method'] == 'direct' and r.get('signals'):
            d['signals'] = r['signals']
    for line in open(ROOT / 'outputs' / 'raw' / pair['closed']):
        r = json.loads(line)
        d = S.setdefault(r['file'], {'stratum': r['stratum'], 'class': r['class'],
                                     'open': {}, 'signals': None, 'closed': {}})
        d['closed'][r['variant']] = (bool(r['correct']), r.get('parsed'), r.get('text'))
    for f, d in S.items():
        if d.get('roundA_abstain'):
            d['open'] = {m: ('abstain', None) for m in d['open']}
    return S


def cause_of(d, variant='full101'):
    if 'direct' not in d['open']:
        return None
    out = d['open']['direct'][0]
    cl = d['closed'].get(variant)
    if cl is None:
        return None
    if out == 'abstain':
        return 'abstain_' + ('closed_ok' if cl[0] else 'closed_bad')
    if out == 'correct':
        return 'known_object' if cl[0] else 'open_ok_closed_bad'
    return 'retrieval_failure' if cl[0] else 'recognition_failure'


def analyse_pair(pair, nb, synset_classes):
    label, dom = pair['label'], pair['domain']
    pid = f'{label}@{dom}'
    S = load_pair(pair)
    out = {'n_samples': len(S), 'j1': {}, 'j2': {}, 'j3': {}, 'j4': {}, 'sanity': {}, 'caveat': {}}
    rows_closed, rows_twoway, rows_bycause, rows_equiv, rows_sig = [], [], [], [], []

    by_class = defaultdict(list)
    for f, d in S.items():
        if 'full101' in d['closed']:
            by_class[(d['stratum'], d['class'])].append(d)
    for (st, c), ds in sorted(by_class.items()):
        n = len(ds)
        op = [d['open'].get('direct', (None,))[0] == 'correct' for d in ds]
        f101 = [d['closed']['full101'][0] for d in ds]
        n10 = [d['closed']['near10'][0] for d in ds] if all('near10' in d['closed'] for d in ds) else []
        rows_closed.append({'pair': pid, 'stratum': st, 'class': c, 'n': n,
                            'open_acc': round(float(np.mean(op)), 4),
                            'closed_full_acc': round(float(np.mean(f101)), 4),
                            'closed_near10_acc': round(float(np.mean(n10)), 4) if n10 else ''})
    for st in ['known', 'deficient']:
        ds = [d for (s, c), dd in by_class.items() if s == st for d in dd]
        if ds:
            rows_closed.append({'pair': pid, 'stratum': st, 'class': '(overall)', 'n': len(ds),
                                'open_acc': round(float(np.mean([d['open'].get('direct', (None,))[0] == 'correct' for d in ds])), 4),
                                'closed_full_acc': round(float(np.mean([d['closed']['full101'][0] for d in ds])), 4),
                                'closed_near10_acc': round(float(np.mean([d['closed']['near10'][0] for d in ds])), 4)})
            if st == 'known':
                out['sanity']['known_full'] = rows_closed[-1]['closed_full_acc']
                out['sanity']['known_near10'] = rows_closed[-1]['closed_near10_acc']
            else:
                out['sanity']['deficient_full'] = rows_closed[-1]['closed_full_acc']
    unpar = [d['closed']['full101'][1] is None for d in S.values() if 'full101' in d['closed']]
    out['sanity']['unparsed_rate_full'] = round(float(np.mean(unpar)), 4) if unpar else None

    for st in ['known', 'deficient']:
        cells = defaultdict(int)
        for f, d in S.items():
            if d['stratum'] != st:
                continue
            cz = cause_of(d)
            if cz:
                cells[cz] += 1
        n_err = cells['retrieval_failure'] + cells['recognition_failure']
        k_, n_ = cells['retrieval_failure'], n_err
        p_, lo_, hi_ = wilson(k_, n_)
        classes_cov = len({d['class'] for f, d in S.items()
                           if d['stratum'] == st and cause_of(d) == 'retrieval_failure'})
        row = {'pair': pid, 'stratum': st, 'n_total': sum(cells.values())}
        row.update({c: cells[c] for c in ['known_object', 'open_ok_closed_bad',
                                          'retrieval_failure', 'recognition_failure',
                                          'abstain_closed_ok', 'abstain_closed_bad']})
        row['retrieval_share_of_errors'] = round(p_, 4) if n_ else ''
        row['share_lo'], row['share_hi'] = (round(lo_, 4), round(hi_, 4)) if n_ else ('', '')
        row['classes_with_R'] = classes_cov
        rows_twoway.append(row)
        if st == 'deficient':
            out['j1'] = {'share': round(p_, 4) if n_ else None,
                         'ci': [round(lo_, 4), round(hi_, 4)] if n_ else None,
                         'classes_with_R': classes_cov, 'n_R': k_,
                         'n_K': cells['recognition_failure'],
                         'pass': bool(p_ >= J1_SHARE and lo_ > J1_CI_LO and classes_cov >= J1_MIN_CLASSES)}

    R = [f for f, d in S.items() if d['stratum'] == 'deficient' and cause_of(d) == 'retrieval_failure']
    K = [f for f, d in S.items() if d['stratum'] == 'deficient' and cause_of(d) == 'recognition_failure']
    for m in METHODS3:
        if not R and not K:
            break
        kR = sum(S[f]['open'][m][0] == 'correct' for f in R if m in S[f]['open'])
        kK = sum(S[f]['open'][m][0] == 'correct' for f in K if m in S[f]['open'])
        nR, nK = len(R), len(K)
        aR, loR, hiR = wilson(kR, nR)
        aK, loK, hiK = wilson(kK, nK)
        d_, dlo, dhi = prop_diff(kR, nR, kK, nK)
        rows_bycause += [{'pair': pid, 'method': m, 'cause': 'retrieval', 'n': nR,
                          'acc': round(aR, 4), 'acc_lo': round(loR, 4), 'acc_hi': round(hiR, 4)},
                         {'pair': pid, 'method': m, 'cause': 'recognition', 'n': nK,
                          'acc': round(aK, 4), 'acc_lo': round(loK, 4), 'acc_hi': round(hiK, 4)},
                         {'pair': pid, 'method': m, 'cause': 'diff_R-K', 'n': nR + nK,
                          'acc': round(d_, 4), 'acc_lo': round(dlo, 4), 'acc_hi': round(dhi, 4),
                          'ci_excl_zero': bool(dlo > 0 or dhi < 0)}]
        out['j2'][m] = {'acc_R': round(aR, 4), 'ci_R': [round(loR, 4), round(hiR, 4)],
                        'acc_K': round(aK, 4), 'ci_K': [round(loK, 4), round(hiK, 4)],
                        'diff': round(d_, 4), 'ci_diff': [round(dlo, 4), round(dhi, 4)],
                        'sig_R_pos': bool(loR > 0), 'sig_K_zero': bool(loK <= 0 <= hiK),
                        'diff_excl_zero': bool(dlo > 0 or dhi < 0)}
    out['j2']['satisfied_methods'] = [m for m in METHODS3
                                      if m in out['j2'] and out['j2'][m]['sig_R_pos']
                                      and out['j2'][m]['sig_K_zero'] and out['j2'][m]['diff_excl_zero']]

    rng = np.random.default_rng(42)
    for m in ['vcd', 'lcd']:
        def bothwrong(files):
            return [S[f]['open'][m][1] - S[f]['open']['direct'][1] for f in files
                    if m in S[f]['open'] and S[f]['open'][m][0] in ('wrong', 'wrong_unparsed')
                    and S[f]['open']['direct'][0] in ('wrong', 'wrong_unparsed')
                    and S[f]['open'][m][1] is not None and S[f]['open']['direct'][1] is not None]
        dR, dK = bothwrong(R), bothwrong(K)
        if dR and dK:
            mR, mK = float(np.mean(dR)), float(np.mean(dK))
            boots = [float(np.mean(rng.choice(dR, len(dR))) - np.mean(rng.choice(dK, len(dK))))
                     for _ in range(1000)]
            lo90, hi90 = np.percentile(boots, [5, 95])
            eq_c = bool(-DELTA < lo90 and hi90 < DELTA)
        else:
            mR = mK = lo90 = hi90 = float('nan'); eq_c = None
        rows_equiv.append({'pair': pid, 'method': m, 'metric': 'd_wrong_conf',
                           'mean_R': round(mR, 4), 'mean_K': round(mK, 4),
                           'n_R': len(dR), 'n_K': len(dK), 'D': round(mR - mK, 4),
                           'lo90': round(float(lo90), 4), 'hi90': round(float(hi90), 4),
                           'equivalent(d=0.05)': eq_c})

        def ece_of(files, method):
            cs = [S[f]['open'][method][1] for f in files
                  if method in S[f]['open'] and S[f]['open'][method][1] is not None]
            ks = [1.0 * (S[f]['open'][method][0] == 'correct') for f in files
                  if method in S[f]['open'] and S[f]['open'][method][1] is not None]
            return ece(cs, ks)

        def dE(iR, iK):
            return (ece_of([R[i] for i in iR], m) - ece_of([R[i] for i in iR], 'direct')) - \
                   (ece_of([K[i] for i in iK], m) - ece_of([K[i] for i in iK], 'direct'))
        if R and K:
            eR = ece_of(R, m) - ece_of(R, 'direct')
            eK = ece_of(K, m) - ece_of(K, 'direct')
            boots = [dE(rng.integers(0, len(R), len(R)), rng.integers(0, len(K), len(K)))
                     for _ in range(500)]
            elo, ehi = np.percentile(boots, [5, 95])
            eq_e = bool(-DELTA < elo and ehi < DELTA)
        else:
            eR = eK = elo = ehi = float('nan'); eq_e = None
        rows_equiv.append({'pair': pid, 'method': m, 'metric': 'd_ece',
                           'mean_R': round(eR, 4), 'mean_K': round(eK, 4),
                           'n_R': len(R), 'n_K': len(K), 'D': round(eR - eK, 4),
                           'lo90': round(float(elo), 4), 'hi90': round(float(ehi), 4),
                           'equivalent(d=0.05)': eq_e})
        out['j3'][m] = {'conf': {'D': round(mR - mK, 4), 'ci90': [round(float(lo90), 4), round(float(hi90), 4)], 'equiv': eq_c},
                        'ece': {'D': round(eR - eK, 4), 'ci90': [round(float(elo), 4), round(float(ehi), 4)], 'equiv': eq_e}}
    out['j3']['all_equiv'] = all(out['j3'][m][mt]['equiv'] is True
                                 for m in ['vcd', 'lcd'] for mt in ['conf', 'ece'])

    pos_e = [S[f]['signals']['entropy'] for f in R if S[f]['signals']]
    pos_j = [S[f]['signals']['jsd'] for f in R if S[f]['signals']]
    neg_e = [S[f]['signals']['entropy'] for f in K if S[f]['signals']]
    neg_j = [S[f]['signals']['jsd'] for f in K if S[f]['signals']]
    if pos_e and neg_e:
        a_e, a_j = auc_mw(pos_e, neg_e), auc_mw(pos_j, neg_j)
        a_c = tiny_logistic_auc(list(zip(pos_e, pos_j)) + list(zip(neg_e, neg_j)),
                                [1] * len(pos_e) + [0] * len(neg_e))
        for nm, a in [('entropy', a_e), ('jsd_blank', a_j), ('logistic_combo', a_c)]:
            rows_sig.append({'pair': pid, 'signal': nm, 'auc': round(a, 4),
                             'n_R': len(pos_e), 'n_K': len(neg_e)})
        out['j4'] = {'entropy': round(a_e, 4), 'jsd_blank': round(a_j, 4),
                     'logistic_combo': round(a_c, 4),
                     'all_below_limit': bool(max(a_e, a_j, a_c) < AUC_LIMIT)}
    else:
        out['j4'] = {'all_below_limit': None}

    R10 = [f for f, d in S.items() if d['stratum'] == 'deficient'
           and cause_of(d, 'near10') == 'retrieval_failure']
    K10 = [f for f, d in S.items() if d['stratum'] == 'deficient'
           and cause_of(d, 'near10') == 'recognition_failure']
    p10 = len(R10) / max(1, len(R10) + len(K10))
    Rs = [f for f in R + K if S[f]['class'] in synset_classes]
    R10s = [f for f in Rs if cause_of(S[f], 'near10') == 'retrieval_failure']
    p10s = len(R10s) / max(1, len(Rs))
    agree = float(np.mean([cause_of(S[f]) == cause_of(S[f], 'near10') for f in R + K])) if R + K else None
    out['caveat'] = {'R_share_near10_all': round(p10, 4),
                     'R_share_near10_synset_only': round(p10s, 4),
                     'cross_variant_agreement': round(agree, 4) if agree is not None else None}
    return out, rows_closed, rows_twoway, rows_bycause, rows_equiv, rows_sig


def main():
    pairs = discover_pairs()
    summary = {'generated': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
               'preregister': 'PREREGISTER_STAGE3.md', 'pairs': [f"{p['label']}@{p['domain']}" for p in pairs],
               'per_pair': {}, 'primary_verdicts': {}}
    all_rows = {'closedset': [], 'twoway': [], 'bycause': [], 'equivalence': [], 'signal_split': []}
    for pair in pairs:
        dom = pair['domain']
        nbfile = {'food101': 'neighbours_food101.json', 'dogs': 'neighbours_stanford_dogs.json'}[dom]
        nb = json.load(open(ROOT / 'data' / nbfile))
        synset_classes = {c for c, ns in nb.items() if ns != sorted(ns)}
        res, rc, rt, rb, re_, rs = analyse_pair(pair, nb, synset_classes)
        pid = f"{pair['label']}@{pair['domain']}"
        summary['per_pair'][pid] = res
        all_rows['closedset'] += rc
        all_rows['twoway'] += rt
        all_rows['bycause'] += rb
        all_rows['equivalence'] += re_
        all_rows['signal_split'] += rs
        print(f"== {pid}: J1 {'PASS' if res['j1'].get('pass') else 'FAIL'} "
              f"share={res['j1'].get('share')} | J2 satisfied={res['j2'].get('satisfied_methods')} "
              f"| J3 all_equiv={res['j3'].get('all_equiv')} | J4 all_below={res['j4'].get('all_below_limit')}",
              flush=True)

    prim = [f'{m}@food101' for m in PRIMARY if f'{m}@food101' in summary['per_pair']]
    summary['primary_verdicts']['j1'] = 'PASS' if all(summary['per_pair'][p]['j1']['pass'] for p in prim) else 'FAIL'
    summary['primary_verdicts']['j2'] = ('PASS' if all(summary['per_pair'][p]['j2']['satisfied_methods'] for p in prim)
                                         else ('PARTIAL' if any(summary['per_pair'][p]['j2']['satisfied_methods'] for p in prim) else 'FAIL'))
    summary['primary_verdicts']['j3'] = ('PASS(equivalent)' if all(summary['per_pair'][p]['j3']['all_equiv'] for p in prim)
                                         else 'FAIL(not equivalent)')
    summary['primary_verdicts']['j4'] = ('PASS(cannot split)' if all(summary['per_pair'][p]['j4']['all_below_limit'] for p in prim)
                                         else 'FAIL(some signal can split)')

    for name, rows in all_rows.items():
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(ROOT / 'outputs' / 'tables' / f'{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    json.dump(summary, open(ROOT / 'outputs' / 'tables' / 'stage3_summary.json', 'w'),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary['primary_verdicts'], indent=1))


if __name__ == '__main__':
    main()
