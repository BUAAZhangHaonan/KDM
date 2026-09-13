"""v2 analysis: judgments 1-3 with wrong-answer confidence as the core outcome.

Reads outputs/raw/{model}_{tag}_{naming,existence}.jsonl + phase-1 files,
recomputes outcomes with final scoring, and writes:
  outputs/tables/main.csv, effects.csv, judgment2_auc.csv, remedy.csv
  outputs/tables/analysis_summary.json
"""
import os, sys, json, math, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import scoring

METHODS = ['direct', 'vcd', 'mib', 'lcd']
STRATA = ['known', 'deficient']


def load_all(model, tag='main'):
    out = {}
    for name in [f'{model}_{tag}_naming.jsonl', f'{model}_{tag}_existence.jsonl']:
        p = ROOT / 'outputs' / 'raw' / name
        out[name.split('_', 1)[1].replace('.jsonl', '')] = (
            [json.loads(l) for l in open(p)] if p.exists() else [])
    return out


def paired_ci(x, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float('nan'),) * 3 + (n,)
    from scipy import stats as sps
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    tc = sps.t.ppf(1 - alpha / 2, n - 1)
    return m, m - tc * se, m + tc * se, n


def boot_ci(f, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    v0 = f(np.arange(n))
    return v0, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_score(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), len(pos), len(neg)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    u = ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg)), len(pos), len(neg)


def tiny_logistic(X, y, iters=300, lr=0.1):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        w -= lr * (Z.T @ (1 / (1 + np.exp(-Z @ w)) - y) / len(y))
    return lambda V: 1 / (1 + np.exp(-(np.c_[np.ones(len(V)), (np.asarray(V, float) - mu) / sd] @ w)))


def ece(confs, corrects, n_bins=10):
    return scoring.calibration_error(confs, corrects, n_bins)


# ---------------------------------------------------------------- main table
def build_main(recs, model, task, classes):
    rows = []
    for m in METHODS:
        sel = [r for r in recs if r['method'] == m]
        for st in STRATA:
            tr = [r for r in sel if r.get('stratum') == st]
            if not tr:
                continue
            mm = scoring.metrics(tr)
            wconf, nw = scoring.wrong_answer_confidence(tr)
            e = ece([r['answer_maxp'] for r in tr if r.get('answer_maxp') is not None],
                    [1.0 * (r['outcome'] == 'correct') for r in tr if r.get('answer_maxp') is not None])
            row = {'model': model, 'task': task, 'method': m, 'stratum': st, 'n': mm['n'],
                   'accuracy': round(mm['accuracy'], 4), 'abstain_rate': round(mm['abstain_rate'], 4),
                   'certain_error_rate': round(mm['certain_error_rate'], 4),
                   'wrong_answer_conf': round(wconf, 4) if wconf == wconf else '',
                   'n_wrong': nw, 'calibration_ece': round(e, 4) if e == e else '',
                   'mean_forwards': round(float(np.mean([r['n_forwards'] for r in tr])), 1),
                   'mean_wall_s': round(float(np.mean([r['wall_s'] for r in tr])), 3)}
            rows.append(row)
        if task == 'existence':
            for qk in ['pos', 'neg']:
                tr = [r for r in sel if r.get('question_kind') == qk]
                if tr:
                    rows.append({'model': model, 'task': task, 'method': m,
                                 'stratum': f'{qk}_pooled', 'n': len(tr),
                                 'accuracy': round(float(np.mean([r['outcome'] == 'correct' for r in tr])), 4),
                                 'abstain_rate': 0.0,
                                 'certain_error_rate': round(float(np.mean([r['outcome'] != 'correct' for r in tr])), 4),
                                 'wrong_answer_conf': '', 'n_wrong': '',
                                 'calibration_ece': '',
                                 'hallucination_on_neg': round(float(np.mean([r['outcome'] == 'wrong' for r in tr])), 4) if qk == 'neg' else ''})
    return rows


# ---------------------------------------------------------------- effects
def build_effects(recs, model, task):
    by_sid = defaultdict(dict)
    for r in recs:
        if r['method'] in METHODS:
            by_sid[r['key'].rsplit(':', 1)[0]][r['method']] = r
    rows = []
    for m in METHODS[1:]:
        for st in STRATA:
            sids = [s for s, d in by_sid.items()
                    if m in d and 'direct' in d and d[m].get('stratum') == st]
            if not sids:
                continue
            ind = lambda r: 1.0 if r['outcome'] in ('wrong', 'wrong_unparsed') else 0.0
            d_err = [ind(by_sid[s][m]) - ind(by_sid[s]['direct']) for s in sids]
            d_acc = [(by_sid[s][m]['outcome'] == 'correct') - (by_sid[s]['direct']['outcome'] == 'correct') for s in sids]
            d_abs = [(by_sid[s][m]['outcome'] == 'abstain') - (by_sid[s]['direct']['outcome'] == 'abstain') for s in sids]
            # paired wrong-confidence on wrong-in-both subset
            both = [s for s in sids if ind(by_sid[s][m]) == 1 and ind(by_sid[s]['direct']) == 1
                    and by_sid[s][m].get('answer_maxp') is not None
                    and by_sid[s]['direct'].get('answer_maxp') is not None]
            d_conf = [by_sid[s][m]['answer_maxp'] - by_sid[s]['direct']['answer_maxp'] for s in both]
            me, le, he, n = paired_ci(d_err)
            ma, la, ha, _ = paired_ci(d_acc)
            mab, lab, hab, _ = paired_ci(d_abs)
            mc, lc, hc, nc = paired_ci(d_conf) if d_conf else (float('nan'),) * 3 + (0,)
            # ECE delta via paired bootstrap
            def ece_pair(idx):
                sub_m = [by_sid[sids[i]][m] for i in idx]
                sub_d = [by_sid[sids[i]]['direct'] for i in idx]
                cm = [r['answer_maxp'] for r in sub_m if r.get('answer_maxp') is not None]
                km = [1.0 * (r['outcome'] == 'correct') for r in sub_m if r.get('answer_maxp') is not None]
                cd = [r['answer_maxp'] for r in sub_d if r.get('answer_maxp') is not None]
                kd = [1.0 * (r['outcome'] == 'correct') for r in sub_d if r.get('answer_maxp') is not None]
                return ece(cm, km) - ece(cd, kd)
            e0, elo, ehi = boot_ci(ece_pair, len(sids))
            rows.append({'model': model, 'task': task, 'method': m, 'stratum': st, 'n': n,
                         'd_certain_error': round(me, 4), 'err_lo': round(le, 4), 'err_hi': round(he, 4),
                         'd_accuracy': round(ma, 4), 'acc_lo': round(la, 4), 'acc_hi': round(ha, 4),
                         'd_abstain': round(mab, 4),
                         'd_wrong_conf': round(mc, 4) if mc == mc else '', 'conf_lo': round(lc, 4) if lc == lc else '',
                         'conf_hi': round(hc, 4) if hc == hc else '', 'n_conf_pairs': nc,
                         'd_ece': round(float(e0), 4) if e0 == e0 else '', 'ece_lo': round(float(elo), 4) if elo == elo else '',
                         'ece_hi': round(float(ehi), 4) if ehi == ehi else '',
                         'err_ci_excl_zero': not (le <= 0 <= he),
                         'conf_ci_excl_zero': (lc == lc and lc > 0),
                         'ece_ci_excl_zero': (elo == elo and elo > 0)})
    # diff-in-diff for confidence + error (deficient - known)
    for m in METHODS[1:]:
        def vec(st, kind):
            out = []
            for s, d in by_sid.items():
                if m in d and 'direct' in d and d[m].get('stratum') == st:
                    if kind == 'err':
                        ind = lambda r: 1.0 if r['outcome'] in ('wrong', 'wrong_unparsed') else 0.0
                        out.append(ind(d[m]) - ind(d['direct']))
                    else:
                        ind = lambda r: 1.0 if r['outcome'] in ('wrong', 'wrong_unparsed') else 0.0
                        if ind(d[m]) == 1 and ind(d['direct']) == 1 and d[m].get('answer_maxp') is not None:
                            out.append(d[m]['answer_maxp'] - d['direct']['answer_maxp'])
            return out
        for kind in ['err', 'conf']:
            vl, vh = vec('deficient', kind), vec('known', kind)
            if vl and vh:
                m1, lo1, hi1, _ = paired_ci(vl)
                m2, lo2, hi2, _ = paired_ci(vh)
                dd = m1 - m2
                se = math.sqrt(((hi1 - lo1) / 3.92) ** 2 + ((hi2 - lo2) / 3.92) ** 2)
                rows.append({'model': model, 'task': task, 'method': m, 'stratum': f'did_{kind}',
                             'n': len(vl),
                             'd_certain_error' if kind == 'err' else 'd_wrong_conf': round(dd, 4),
                             'err_lo' if kind == 'err' else 'conf_lo': round(dd - 1.96 * se, 4),
                             'err_hi' if kind == 'err' else 'conf_hi': round(dd + 1.96 * se, 4),
                             'err_ci_excl_zero' if kind == 'err' else 'conf_ci_excl_zero':
                                 not (dd - 1.96 * se <= 0 <= dd + 1.96 * se)})
    return rows


# ---------------------------------------------------------------- judgment 2
def gather_signals(recs):
    by_sid = defaultdict(dict)
    for r in recs:
        if r['method'] == 'direct':
            d = by_sid[r['key'].rsplit(':', 1)[0]]
            d['stratum'] = r['stratum']
            d['direct'] = r['outcome']
            if r.get('signals'):
                d.update(entropy=r['signals']['entropy'], jsd=r['signals']['jsd'])
        elif r['method'] in METHODS[1:]:
            by_sid[r['key'].rsplit(':', 1)[0]][r['method']] = r['outcome']
    return [v for v in by_sid.values() if 'entropy' in v and 'direct' in v]


def hh(s, m):
    d, mm = s.get('direct'), s.get(m)
    if d is None or mm is None:
        return None, None
    harmed = int(d in ('correct', 'abstain') and mm in ('wrong', 'wrong_unparsed'))
    helped = int(d in ('wrong', 'wrong_unparsed') and mm in ('correct', 'abstain'))
    return harmed, helped


def build_j2(recs, model):
    sig = gather_signals(recs)
    rows = []
    def emit(sel_methods, label, st_filter):
        ss = [s for s in sig if st_filter == 'all' or s['stratum'] == st_filter]
        pairs = []
        for s in ss:
            hs, gs = [], []
            for m in sel_methods:
                h, g = hh(s, m)
                if h is not None:
                    hs.append(h); gs.append(g)
            if hs and (any(hs) or any(gs)) and not (any(hs) and any(gs)):
                pairs.append((s, 1 if any(hs) else 0))
        if len({p[1] for p in pairs}) < 2:
            for nm in ['entropy', 'jsd', 'logistic_combo']:
                rows.append({'model': model, 'method': label, 'stratum': st_filter, 'signal': nm,
                             'auc': float('nan'), 'n_harmed': sum(p[1] for p in pairs), 'n_helped': len(pairs) - sum(p[1] for p in pairs)})
            return
        labels = [p[1] for p in pairs]
        for nm in ['entropy', 'jsd']:
            a, nh, ng = auc_score([p[0][nm] for p in pairs], labels)
            rows.append({'model': model, 'method': label, 'stratum': st_filter, 'signal': nm,
                         'auc': round(a, 4), 'n_harmed': nh, 'n_helped': ng})
        X = np.array([[p[0]['entropy'], p[0]['jsd']] for p in pairs])
        clf = tiny_logistic(X, np.array(labels))
        a, _, _ = auc_score(clf(X), labels)
        rows.append({'model': model, 'method': label, 'stratum': st_filter, 'signal': 'logistic_combo',
                     'auc': round(a, 4), 'n_harmed': int(sum(labels)), 'n_helped': len(labels) - int(sum(labels))})
    for m in METHODS[1:]:
        for st in STRATA + ['all']:
            emit([m], m, st)
    for st in STRATA + ['all']:
        emit(METHODS[1:], 'pooled', st)
    return rows


# ---------------------------------------------------------------- judgment 3
def apply_rule(s, m_star, tau_h, tau_j):
    if s['entropy'] > tau_h:
        return 'abstain'
    if s['jsd'] > tau_j:
        return m_star
    return 'direct'


def build_j3(recs, model):
    sig = gather_signals(recs)
    known = [s for s in sig if s['stratum'] == 'known']
    defic = [s for s in sig if s['stratum'] == 'deficient']
    gains = {}
    for m in METHODS[1:]:
        pairs = [(s.get(m), s.get('direct')) for s in known]
        pairs = [(a, b) for a, b in pairs if a and b]
        gains[m] = (np.mean([a == 'correct' for a, b in pairs]) -
                    np.mean([b == 'correct' for a, b in pairs])) if pairs else -1
    m_star = max(gains, key=gains.get)

    direct_acc = {st: np.mean([s['direct'] == 'correct' for s in ss])
                  for st, ss in [('known', known), ('deficient', defic)]}
    direct_cer = {st: np.mean([s['direct'] in ('wrong', 'wrong_unparsed') for s in ss])
                  for st, ss in [('known', known), ('deficient', defic)]}

    def evaluate(ss, th, tj):
        outs = []
        for s in ss:
            act = apply_rule(s, m_star, th, tj)
            outs.append('abstain' if act == 'abstain' else s.get(act, s['direct']))
        mm = scoring.metrics([{'outcome': o} for o in outs])
        return mm

    qs = [i / 10 for i in range(11)]
    best = None
    for qh in qs:
        for qj in qs:
            th = float(np.quantile([s['entropy'] for s in known], qh))
            tj = float(np.quantile([s['jsd'] for s in known], qj))
            mk = evaluate(known, th, tj)
            if mk['accuracy'] < direct_acc['known'] - 0.02 - 1e-9:
                continue
            obj = mk['accuracy'] - mk['certain_error_rate']
            if best is None or obj > best[0]:
                best = (obj, th, tj, qh, qj)
    if best is None:
        best = (float('-inf'), float('inf'), float('inf'), 1.0, 1.0)
    _, th, tj, qh, qj = best
    th_med = float(np.median([s['entropy'] for s in sig]))
    tj_med = float(np.median([s['jsd'] for s in sig]))

    rows = []
    for rule, a, b in [('remedy_known_tuned', th, tj), ('remedy_median', th_med, tj_med)]:
        for st, ss in [('known', known), ('deficient', defic)]:
            mm = evaluate(ss, a, b)
            rows.append({'model': model, 'rule': rule, 'm_star': m_star,
                         'tau_entropy': round(a, 4), 'tau_jsd': round(b, 4), 'stratum': st,
                         'n': mm['n'], 'accuracy': round(mm['accuracy'], 4),
                         'abstain_rate': round(mm['abstain_rate'], 4),
                         'certain_error_rate': round(mm['certain_error_rate'], 4),
                         'd_cer_vs_direct': round(mm['certain_error_rate'] - direct_cer[st], 4),
                         'd_acc_vs_direct': round(mm['accuracy'] - direct_acc[st], 4)})
    for st, ss in [('known', known), ('deficient', defic)]:
        outs = [('correct' if s['direct'] == 'correct'
                 else ('correct' if any(s.get(m) == 'correct' for m in METHODS[1:]) else 'abstain'))
                for s in ss]
        mm = scoring.metrics([{'outcome': o} for o in outs])
        rows.append({'model': model, 'rule': 'ideal', 'm_star': '', 'tau_entropy': '', 'tau_jsd': '',
                     'stratum': st, 'n': mm['n'], 'accuracy': round(mm['accuracy'], 4),
                     'abstain_rate': round(mm['abstain_rate'], 4),
                     'certain_error_rate': round(mm['certain_error_rate'], 4),
                     'd_cer_vs_direct': round(mm['certain_error_rate'] - direct_cer[st], 4),
                     'd_acc_vs_direct': round(mm['accuracy'] - direct_acc[st], 4)})
    info = {'m_star': m_star, 'm_star_known_gains': {k: round(float(v), 4) for k, v in gains.items()},
            'tau_entropy': th, 'tau_jsd': tj, 'q_entropy': qh, 'q_jsd': qj}
    return rows, info


def find_examples(recs, model, k=6):
    by_sid = {}
    for r in recs:
        if r['method'] in METHODS:
            by_sid.setdefault(r['key'].rsplit(':', 1)[0], {})[r['method']] = r
    ex = []
    for sid, d in by_sid.items():
        if 'direct' not in d:
            continue
        dr = d['direct']
        for m in METHODS[1:]:
            if m not in d:
                continue
            mr = d[m]
            both_wrong = (dr['outcome'] in ('wrong', 'wrong_unparsed')
                          and mr['outcome'] in ('wrong', 'wrong_unparsed'))
            conf_up = (both_wrong and dr.get('answer_maxp') is not None
                       and mr.get('answer_maxp') is not None
                       and mr['answer_maxp'] - dr['answer_maxp'] > 0.15)
            abst2err = (dr['outcome'] == 'abstain' and mr['outcome'] in ('wrong', 'wrong_unparsed'))
            correct2err = (dr['outcome'] == 'correct' and mr['outcome'] in ('wrong', 'wrong_unparsed'))
            if conf_up or abst2err or correct2err:
                ex.append({'model': model, 'file': dr['file'], 'class': dr['class'],
                           'stratum': dr['stratum'], 'method': m,
                           'direct_text': dr['text'].strip(), 'direct_outcome': dr['outcome'],
                           'direct_conf': dr.get('answer_maxp'),
                           'method_text': mr['text'].strip(), 'method_outcome': mr['outcome'],
                           'method_conf': mr.get('answer_maxp'), 'kind': ('conf_up' if conf_up else 'flip')})
                break
        if len(ex) >= k * 4:
            break
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='q4b,q9b')
    ap.add_argument('--tag', default='main')
    args = ap.parse_args()
    models = args.models.split(',')
    tabs = ROOT / 'outputs' / 'tables'
    tabs.mkdir(parents=True, exist_ok=True)
    summary = {'models': models, 'generated': time.strftime('%Y-%m-%d %H:%M:%S')}

    all_main, all_eff, all_j2, all_j3, all_ex = [], [], [], [], []
    for model in models:
        data = load_all(model, args.tag)
        classes = sorted({r['class'] for r in data['naming']})
        for r in data['naming']:
            r['outcome'] = scoring.score_naming(r['text'], r['class'], classes)
        for r in data['existence']:
            r['outcome'] = scoring.score_existence(r['text'], r['gold_present'])
        summary.setdefault('counts', {})[model] = {k: len(v) for k, v in data.items()}
        all_main += build_main(data['naming'], model, 'naming', classes)
        all_main += build_main(data['existence'], model, 'existence', classes)
        all_eff += build_effects(data['naming'], model, 'naming')
        all_eff += build_effects(data['existence'], model, 'existence')
        all_j2 += build_j2(data['naming'], model)
        j3rows, j3info = build_j3(data['naming'], model)
        all_j3 += j3rows
        summary.setdefault('j3_info', {})[model] = j3info
        all_ex += find_examples(data['naming'], model)

    def dump(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for kk in r:
                if kk not in keys:
                    keys.append(kk)
        with open(tabs / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    import csv
    dump('main.csv', all_main)
    dump('effects.csv', all_eff)
    dump('judgment2_auc.csv', all_j2)
    dump('remedy.csv', all_j3)
    summary['examples'] = all_ex[:24]
    json.dump(summary, open(tabs / 'analysis_summary.json', 'w'), indent=1, ensure_ascii=False)
    print(f'main:{len(all_main)} effects:{len(all_eff)} j2:{len(all_j2)} j3:{len(all_j3)} examples:{len(all_ex)}')


if __name__ == '__main__':
    main()
