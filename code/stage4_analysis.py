"""Stage 4 analysis: LL-probe decomposition, judgments 1-4 (PREREGISTER_STAGE4).

Exploratory pairs  : q4b/q9b/q3vl4b @food101, q4b/q9b @dogs  (recomputation)
Confirmatory pairs : llava16/internvl4b @food101 (fresh probe + stage-3 naming)

Outputs: outputs/tables/{probe_validity,twoway_v4,bycause_v4,robustness_probe}.csv
         outputs/tables/stage4_summary.json
"""
import os, sys, json, math, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
from stage3_analysis import (wilson, prop_diff, auc_mw, tiny_logistic_auc, ece,
                             METHODS3, DELTA, AUC_LIMIT, J1_SHARE, J1_CI_LO, J1_MIN_CLASSES)
from stage3_analysis import load_pair as load_naming_pair
from stage3_analysis import discover_pairs

CONFIRMATORY = {'llava16@food101', 'internvl4b@food101'}
NICE = {'q4b': 'Qwen3.5-4B', 'q9b': 'Qwen3.5-9B', 'q3vl4b': 'Qwen3-VL-4B',
        'llava16': 'LLaVA-v1.6-7B', 'internvl4b': 'InternVL3.5-4B', 'glm46v': 'GLM-4.6V-Flash'}


def load_probe(model, domain):
    """file -> probe record (image/blank rank etc)."""
    out = {}
    p = ROOT / 'outputs' / 'raw' / f'{model}_stage4_probe_{domain}.jsonl'
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        out[r['file']] = r
    return out


def load_direct_texts(pair):
    """file -> direct-condition generated text (granularity diagnosis)."""
    out = {}
    for line in open(ROOT / 'outputs' / 'raw' / pair['naming']):
        r = json.loads(line)
        if r.get('task') == 'naming' and r.get('method') == 'direct' \
                and not r.get('error') and not r.get('unavailable'):
            out[r['file']] = (r.get('text') or '')
    return out


def cause_v4(d, probe):
    """Four-cell row using the LL probe as the knowledge classifier."""
    if 'direct' not in d['open']:
        return None
    o = d['open']['direct'][0]
    if o == 'abstain':
        return 'abstain_' + ('probe_ok' if probe['rank_image'] == 0 else 'probe_bad')
    if o == 'correct':
        return 'known_object' if probe['rank_image'] == 0 else 'open_ok_probe_bad'
    return 'retrieval_failure' if probe['rank_image'] == 0 else 'recognition_failure'


def analyse(pair, role):
    pid = f"{pair['label']}@{pair['domain']}"
    S = load_naming_pair(pair)
    P = load_probe(pair['label'], pair['domain'])
    # keep only files with both naming (direct) and probe records
    files = [f for f in S if 'direct' in S[f]['open'] and f in P]
    res = {'role': role, 'n': len(files), 'j1': {}, 'j2': {}, 'j3': {}, 'j4': {},
           'validity': {}, 'rows': {'twoway': [], 'bycause': []}}
    if not files:
        return res, None

    # ---------- probe validity ----------
    kn = [f for f in files if S[f]['stratum'] == 'known']
    acc_kn = float(np.mean([P[f]['rank_image'] == 0 for f in kn])) if kn else None
    # granularity diagnosis (post-hoc, diagnostic only -- NOT a gate): probe
    # accuracy conditioned on the model having produced the FULL class name
    # open-set; separates "name-level knowledge absent" from "generic naming"
    txts = load_direct_texts(pair)
    import re as _re
    def _norm_txt(t):
        return _re.sub(r'[^a-z]', '', (t or '').lower())
    full_name = lambda f: _norm_txt(txts.get(f, '')) == _norm_txt(S[f]['class'].replace('_', ''))
    kn_full = [f for f in kn if full_name(f)]
    acc_kn_full = float(np.mean([P[f]['rank_image'] == 0 for f in kn_full])) if kn_full else None
    kn_ok_nf = [f for f in kn if not full_name(f) and S[f]['open']['direct'][0] == 'correct']
    acc_ok_nf = float(np.mean([P[f]['rank_image'] == 0 for f in kn_ok_nf])) if kn_ok_nf else None
    rd = [P[f]['rank_blank'] - P[f]['rank_image'] for f in files]
    m = float(np.mean(rd))
    se = float(np.std(rd, ddof=1) / math.sqrt(len(rd))) if len(rd) > 1 else float('inf')
    gate1 = bool(acc_kn is not None and acc_kn >= 0.90)
    gate2 = bool(m - 1.96 * se > 0)
    res['validity'] = {'known_probe_acc': round(acc_kn, 4) if acc_kn is not None else None,
                       'n_known': len(kn),
                       'known_probe_acc_given_fullname': round(acc_kn_full, 4) if acc_kn_full is not None else None,
                       'n_known_fullname': len(kn_full),
                       'known_probe_acc_given_correct_notfull': round(acc_ok_nf, 4) if acc_ok_nf is not None else None,
                       'n_known_correct_notfull': len(kn_ok_nf),
                       'rank_diff_mean': round(m, 3),
                       'rank_diff_ci': [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)],
                       'gate1_known_acc': gate1, 'gate2_blank_control': gate2,
                       'probe_valid': bool(gate1 and gate2)}

    # ---------- twoway ----------
    k_per_class = 30 if pair['domain'] == 'dogs' else 25
    for st in ['known', 'deficient']:
        cells = defaultdict(int)
        for f in files:
            if S[f]['stratum'] != st:
                continue
            cz = cause_v4(S[f], P[f])
            if cz:
                cells[cz] += 1
        row = {'pair': pid, 'stratum': st, 'role': role, 'n_total': sum(cells.values())}
        row.update({c: cells[c] for c in
                    ['known_object', 'open_ok_probe_bad', 'retrieval_failure',
                     'recognition_failure', 'abstain_probe_ok', 'abstain_probe_bad']})
        n_err = cells['retrieval_failure'] + cells['recognition_failure']
        p_, lo, hi = wilson(cells['retrieval_failure'], n_err)
        row['retrieval_share'] = round(p_, 4) if n_err else ''
        row['share_lo'], row['share_hi'] = (round(lo, 4), round(hi, 4)) if n_err else ('', '')
        row['classes_with_R'] = len({S[f]['class'] for f in files
                                     if S[f]['stratum'] == st
                                     and cause_v4(S[f], P[f]) == 'retrieval_failure'})
        res['rows']['twoway'].append(row)
        if st == 'deficient':
            res['j1'] = {'share': round(p_, 4), 'ci': [round(lo, 4), round(hi, 4)],
                         'classes_with_R': row['classes_with_R'],
                         'n_R': cells['retrieval_failure'],
                         'n_K': cells['recognition_failure'],
                         'pass': bool(p_ >= J1_SHARE and lo > J1_CI_LO
                                      and row['classes_with_R'] >= math.ceil(0.28 * k_per_class))}

    # ---------- bycause ----------
    R = [f for f in files if S[f]['stratum'] == 'deficient'
         and cause_v4(S[f], P[f]) == 'retrieval_failure']
    K = [f for f in files if S[f]['stratum'] == 'deficient'
         and cause_v4(S[f], P[f]) == 'recognition_failure']
    for meth in METHODS3:
        kR = sum(S[f]['open'][meth][0] == 'correct' for f in R)
        kK = sum(S[f]['open'][meth][0] == 'correct' for f in K)
        aR, loR, hiR = wilson(kR, len(R))
        aK, loK, hiK = wilson(kK, len(K))
        d_, dlo, dhi = prop_diff(kR, len(R), kK, len(K))
        rows = [{'pair': pid, 'role': role, 'method': meth, 'cause': 'retrieval',
                 'n': len(R), 'acc': round(aR, 4), 'acc_lo': round(loR, 4), 'acc_hi': round(hiR, 4)},
                {'pair': pid, 'role': role, 'method': meth, 'cause': 'recognition',
                 'n': len(K), 'acc': round(aK, 4), 'acc_lo': round(loK, 4), 'acc_hi': round(hiK, 4)},
                {'pair': pid, 'role': role, 'method': meth, 'cause': 'diff_R-K',
                 'n': len(R) + len(K), 'acc': round(d_, 4), 'acc_lo': round(dlo, 4),
                 'acc_hi': round(dhi, 4), 'ci_excl_zero': bool(dlo > 0 or dhi < 0)}]
        res['rows']['bycause'] += rows
        res['j2'][meth] = {'acc_R': round(aR, 4), 'ci_R': [round(loR, 4), round(hiR, 4)],
                           'acc_K': round(aK, 4), 'ci_K': [round(loK, 4), round(hiK, 4)],
                           'diff': round(d_, 4), 'ci_diff': [round(dlo, 4), round(dhi, 4)],
                           'cond1_R_pos': bool(loR > 0),
                           'cond2_diff_excl0': bool(dlo > 0 or dhi < 0),
                           'cond3_K_lt_half': bool(aK < aR / 2)}
    res['j2']['methods_ok'] = [m for m in METHODS3
                               if all(res['j2'][m][c] for c in
                                      ['cond1_R_pos', 'cond2_diff_excl0', 'cond3_K_lt_half'])]

    # ---------- J3: D = dECE(R) - dECE(K) ----------
    rng = np.random.default_rng(42)

    def ece_of(fs, method):
        cs = [S[f]['open'][method][1] for f in fs
              if method in S[f]['open'] and S[f]['open'][method][1] is not None]
        ks = [1.0 * (S[f]['open'][method][0] == 'correct') for f in fs
              if method in S[f]['open'] and S[f]['open'][method][1] is not None]
        return ece(cs, ks)

    for meth in ['vcd', 'lcd']:
        def dE(iR, iK):
            return (ece_of([R[i] for i in iR], meth) - ece_of([R[i] for i in iR], 'direct')) - \
                   (ece_of([K[i] for i in iK], meth) - ece_of([K[i] for i in iK], 'direct'))
        eR = ece_of(R, meth) - ece_of(R, 'direct')
        eK = ece_of(K, meth) - ece_of(K, 'direct')
        boots = [dE(rng.integers(0, len(R), len(R)), rng.integers(0, len(K), len(K)))
                 for _ in range(500)]
        lo, hi = np.percentile(boots, [5, 95])
        res['j3'][meth] = {'D': round(eR - eK, 4), 'ci90': [round(float(lo), 4), round(float(hi), 4)],
                           'hold': bool(eR - eK < 0 and (lo > 0 or hi < 0) and lo < 0)}
        res['rows']['bycause'].append({'pair': pid, 'role': role, 'method': meth,
                                       'cause': 'D_dECE_R_minus_K', 'n': len(R) + len(K),
                                       'acc': round(eR - eK, 4), 'acc_lo': round(float(lo), 4),
                                       'acc_hi': round(float(hi), 4),
                                       'ci_excl_zero': bool(lo > 0 or hi < 0)})

    # ---------- J4 ----------
    pos_e = [S[f]['signals']['entropy'] for f in R if S[f]['signals']]
    pos_j = [S[f]['signals']['jsd'] for f in R if S[f]['signals']]
    neg_e = [S[f]['signals']['entropy'] for f in K if S[f]['signals']]
    neg_j = [S[f]['signals']['jsd'] for f in K if S[f]['signals']]
    if pos_e and neg_e:
        a_e = auc_mw(pos_e, neg_e)
        a_j = auc_mw(pos_j, neg_j)
        a_c = tiny_logistic_auc(list(zip(pos_e, pos_j)) + list(zip(neg_e, neg_j)),
                                [1] * len(pos_e) + [0] * len(neg_e))
        res['j4'] = {'entropy': round(a_e, 4), 'jsd_blank': round(a_j, 4),
                     'logistic_combo': round(a_c, 4),
                     'all_below': bool(max(a_e, a_j, a_c) < AUC_LIMIT)}
    return res, res


def main():
    pairs = discover_pairs()
    summary = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
               'preregister': 'PREREGISTER_STAGE4.md', 'per_pair': {}, 'judgments': {}}
    all_twoway, all_bycause, all_validity, all_rob = [], [], [], []

    for pair in pairs:
        pid = f"{pair['label']}@{pair['domain']}"
        role = 'confirmatory' if pid in CONFIRMATORY else 'exploratory'
        if pid not in CONFIRMATORY and not pair['domain'] == 'food101' and pair['label'] in ('q4b', 'q9b'):
            role = 'exploratory'  # dogs recomputation
        if pair['label'] in ('llava16', 'internvl4b') and pair['domain'] != 'food101':
            continue
        probe_file = ROOT / 'outputs' / 'raw' / f"{pair['label']}_stage4_probe_{pair['domain']}.jsonl"
        if not probe_file.exists():
            continue
        res, _ = analyse(pair, role)
        summary['per_pair'][pid] = res
        all_validity.append(dict(res['validity'], pair=pid, role=role, n=res['n']))
        all_twoway += res['rows']['twoway']
        all_bycause += res['rows']['bycause']
        print(f"== {pid} [{role}] probe_valid={res['validity'].get('probe_valid')} "
              f"known_acc={res['validity'].get('known_probe_acc')} "
              f"J1={'PASS' if res['j1'].get('pass') else 'FAIL'} "
              f"J2ok={res['j2'].get('methods_ok')}", flush=True)

    # robustness: R-share under three probe versions
    for pair in pairs:
        pid = f"{pair['label']}@{pair['domain']}"
        if pid not in summary['per_pair']:
            continue
        import csv as _csv
        s3 = ROOT / 'outputs' / 'tables' / 'twoway.csv'
        if s3.exists():
            for r in _csv.DictReader(open(s3)):
                if r['pair'] == pid and r['stratum'] == 'deficient':
                    all_rob.append({'pair': pid, 'probe': 'full_number_choice',
                                    'R_share': r['retrieval_share_of_errors'],
                                    'lo': r['share_lo'], 'hi': r['share_hi']})
        v4row = next((r for r in all_twoway if r['pair'] == pid and r['stratum'] == 'deficient'), None)
        if v4row:
            all_rob.append({'pair': pid, 'probe': 'll_probe_v4',
                            'R_share': v4row['retrieval_share'], 'lo': v4row['share_lo'],
                            'hi': v4row['share_hi']})
        # near10 share from stage-3 summary
        s3s = ROOT / 'outputs' / 'tables' / 'stage3_summary.json'
        if s3s.exists():
            s3sum = json.load(open(s3s))
            cav = s3sum.get('per_pair', {}).get(pid, {}).get('caveat', {})
            if cav.get('R_share_near10_all') is not None:
                all_rob.append({'pair': pid, 'probe': 'near10_choice',
                                'R_share': cav['R_share_near10_all'], 'lo': '', 'hi': ''})

    # overall judgments per preregister
    valid_pairs = [p for p, r in summary['per_pair'].items() if r['validity'].get('probe_valid')]
    conf_valid = [p for p in valid_pairs if p in CONFIRMATORY]
    summary['judgments']['j1_pairs_pass'] = [p for p in valid_pairs if summary['per_pair'][p]['j1'].get('pass')]
    summary['judgments']['j2'] = {
        'pairs_with_method': [p for p in valid_pairs if summary['per_pair'][p]['j2'].get('methods_ok')],
        'reversals': [p for p in valid_pairs for m in METHODS3
                      if summary['per_pair'][p]['j2'][m]['ci_diff'][1] < 0]}
    summary['judgments']['j3_confirmatory'] = {
        p: summary['per_pair'][p]['j3'] for p in conf_valid}
    summary['judgments']['j4_pairs_fail'] = [p for p in valid_pairs
                                             if not summary['per_pair'][p]['j4'].get('all_below')]
    summary['judgments']['valid_pairs'] = valid_pairs
    summary['judgments']['confirmatory_valid'] = conf_valid

    def dump(name, rows):
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(ROOT / 'outputs' / 'tables' / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    dump('probe_validity.csv', all_validity)
    dump('twoway_v4.csv', all_twoway)
    dump('bycause_v4.csv', all_bycause)
    dump('robustness_probe.csv', all_rob)
    json.dump(summary, open(ROOT / 'outputs' / 'tables' / 'stage4_summary.json', 'w'),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary['judgments'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
