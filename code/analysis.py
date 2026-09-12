"""Analysis: judgments 1-3, main/effects/remedy tables, figures.

Reads outputs/raw/{model}_{tag}_{task}.jsonl, recomputes outcomes with the FINAL
scoring rules (records carry raw text + gold metadata), and produces:
  outputs/tables/main.csv, effects.csv, remedy.csv
  outputs/figures/figure1.pdf, figure2.pdf, figure3.pdf
  outputs/analysis_summary.json (all numbers used by CONCLUSIONS.md)
"""
import os, sys, json, math, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import scoring

METHODS = ['direct', 'vcd', 'mib', 'lcd']
TIERS = ['high', 'low']


# ---------------------------------------------------------------- loading
def load_recs(model, tag='main'):
    out = {}
    for task in ['naming', 'existence']:
        f = ROOT / 'outputs' / 'raw' / f'{model}_{tag}_{task}.jsonl'
        recs = []
        if f.exists():
            for line in open(f):
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        out[task] = recs
    return out


def rescore(recs, task):
    for r in recs:
        if task == 'naming':
            syns = SYNONYMS.get(r['gold'])
            r['outcome'] = scoring.score_naming(r['text'], r['gold'], syns, r.get('synset'))
        else:
            r['outcome'] = scoring.score_existence(r['text'], r['gold_present'])
    return recs


SYNONYMS = {}


def load_synonyms():
    for split in ['lvis_v1_val.json', 'lvis_v1_train.json']:
        p = ROOT / 'data' / split
        if not p.exists():
            continue
        d = json.load(open(p))
        for c in d['categories']:
            SYNONYMS.setdefault(c['name'], c['synonyms'])


# ---------------------------------------------------------------- statistics
def paired_ci(x, alpha=0.05):
    """x: per-sample 0/1 differences. Returns (mean, lo, hi, n) via normal-approx
    paired t interval."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float('nan'),) * 3 + (n,)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    from scipy import stats as sps
    tc = sps.t.ppf(1 - alpha / 2, n - 1)
    return m, m - tc * se, m + tc * se, n


def auc_score(scores, labels):
    """Mann-Whitney AUC; labels 1 = positive class (harmed)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), len(pos), len(neg)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # mid-ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg)), len(pos), len(neg)


def tiny_logistic(X, y, iters=300, lr=0.1):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.c_[np.ones(len(Z)), Z]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y)
        w -= lr * g
    return lambda V: 1 / (1 + np.exp(-(np.c_[np.ones(len(V)), (np.asarray(V, float) - mu) / sd] @ w)))


def ci_diff_of_diff(d1, d2):
    """CI for d1 - d2 given per-sample difference vectors."""
    m1, lo1, hi1, n1 = paired_ci(d1)
    m2, lo2, hi2, n2 = paired_ci(d2)
    se1 = (hi1 - lo1) / (2 * 1.96)
    se2 = (hi2 - lo2) / (2 * 1.96)
    dd = m1 - m2
    se = math.sqrt(se1 ** 2 + se2 ** 2)
    return dd, dd - 1.96 * se, dd + 1.96 * se


# ---------------------------------------------------------------- tables
def build_main(recs_by_task, model):
    rows = []
    for task, recs in recs_by_task.items():
        for m in METHODS:
            sel = [r for r in recs if r['method'] == m]
            for tier in TIERS:
                tr = [r for r in sel if r['tier'] == tier]
                if not tr:
                    continue
                mm = scoring.metrics(tr)
                row = {'model': model, 'task': task, 'method': m, 'tier': tier,
                       'n': mm['n'], 'accuracy': round(mm['accuracy'], 4),
                       'abstain_rate': round(mm['abstain_rate'], 4),
                       'certain_error_rate': round(mm['certain_error_rate'], 4)}
                if task == 'naming':
                    confs = [r['first_maxp'] for r in tr if r['method'] == m]
                    corr = [1.0 * (r['outcome'] == 'correct') for r in tr]
                    row['calibration_ece'] = round(scoring.calibration_error(confs, corr), 4)
                    row['mean_forwards'] = round(np.mean([r['n_forwards'] for r in tr]), 1)
                    row['mean_wall_s'] = round(np.mean([r['wall_s'] for r in tr]), 3)
                rows.append(row)
            # existence: hallucination on negatives, per tier and pooled
            if task == 'existence':
                for tier in TIERS + ['both']:
                    tr = [r for r in sel if r.get('question_kind') == 'neg' and
                          (tier == 'both' or r['tier'] == tier)]
                    if tr:
                        hal = np.mean([r['outcome'] != 'correct' for r in tr])
                        acc = np.mean([r['outcome'] == 'correct' for r in tr])
                        rows.append({'model': model, 'task': 'existence', 'method': m,
                                     'tier': f'neg_{tier}', 'n': len(tr),
                                     'accuracy': round(float(acc), 4), 'abstain_rate': 0.0,
                                     'certain_error_rate': round(float(hal), 4),
                                     'hallucination_on_neg': round(float(hal), 4)})
    return rows


def build_effects(recs_by_task, model):
    rows = []
    for task, recs in recs_by_task.items():
        by_sid = defaultdict(dict)
        for r in recs:
            by_sid[r['sample_id']][r['method']] = r
        for m in METHODS[1:]:
            for tier in TIERS:
                sids = [s for s, d in by_sid.items()
                        if m in d and 'direct' in d and d[m]['tier'] == tier]
                if not sids:
                    continue
                def ind(r):
                    return 1.0 if r['outcome'] in ('wrong', 'wrong_unparsed') else 0.0
                d_err = [ind(by_sid[s][m]) - ind(by_sid[s]['direct']) for s in sids]
                d_acc = [(1.0 if by_sid[s][m]['outcome'] == 'correct' else 0.0) -
                         (1.0 if by_sid[s]['direct']['outcome'] == 'correct' else 0.0) for s in sids]
                d_abs = [(1.0 if by_sid[s][m]['outcome'] == 'abstain' else 0.0) -
                         (1.0 if by_sid[s]['direct']['outcome'] == 'abstain' else 0.0) for s in sids]
                me, le, he, n = paired_ci(d_err)
                ma, la, ha, _ = paired_ci(d_acc)
                mab, lab, hab, _ = paired_ci(d_abs)
                rows.append({'model': model, 'task': task, 'method': m, 'tier': tier, 'n': n,
                             'd_certain_error': round(me, 4), 'err_lo': round(le, 4), 'err_hi': round(he, 4),
                             'd_accuracy': round(ma, 4), 'acc_lo': round(la, 4), 'acc_hi': round(ha, 4),
                             'd_abstain': round(mab, 4), 'abs_lo': round(lab, 4), 'abs_hi': round(hab, 4),
                             'err_ci_excl_zero': not (le <= 0 <= he)})
    # judgment-1 diff-in-diff (low - high) per method/task
    for task in recs_by_task:
        recs = recs_by_task[task]
        by_sid = defaultdict(dict)
        for r in recs:
            by_sid[r['sample_id']][r['method']] = r
        for m in METHODS[1:]:
            def diff_vec(tier):
                out = []
                for s, d in by_sid.items():
                    if m in d and 'direct' in d and d[m]['tier'] == tier:
                        ind = lambda r: 1.0 if r['outcome'] in ('wrong', 'wrong_unparsed') else 0.0
                        out.append(ind(d[m]) - ind(d['direct']))
                return out
            lo_v, hi_v = diff_vec('low'), diff_vec('high')
            if lo_v and hi_v:
                dd, dlo, dhi = ci_diff_of_diff(lo_v, hi_v)
                rows.append({'model': model, 'task': task, 'method': m, 'tier': 'low_minus_high',
                             'n': len(lo_v), 'd_certain_error': round(dd, 4),
                             'err_lo': round(dlo, 4), 'err_hi': round(dhi, 4),
                             'err_ci_excl_zero': not (dlo <= 0 <= dhi)})
    return rows


# ---------------------------------------------------------------- signals
def gather_signals(recs_naming):
    """join direct + per-method outcomes with direct-record signals"""
    by_sid = {}
    for r in recs_naming:
        if r['method'] == 'direct' and r.get('signals'):
            by_sid.setdefault(r['sample_id'], {}).update(
                tier=r['tier'], entropy=r['signals']['entropy'], jsd=r['signals']['jsd'],
                maxp=r['signals'].get('maxp'))
        elif r['method'] != 'direct':
            by_sid.setdefault(r['sample_id'], {})[r['method']] = r['outcome']
    return [v for v in by_sid.values() if 'entropy' in v and 'direct' in v]


def harmed_helped(sig_row, m):
    d, mm = sig_row.get('direct'), sig_row.get(m)
    if d is None or mm is None:
        return None, None
    harmed = int((d in ('correct', 'abstain')) and mm in ('wrong', 'wrong_unparsed'))
    helped = int(d in ('wrong', 'wrong_unparsed') and mm in ('correct', 'abstain'))
    return harmed, helped


def build_judgment2(recs_naming, model):
    rows = []
    sig = gather_signals(recs_naming)
    for m in METHODS[1:]:
        for tier in TIERS + ['all']:
            ss = [s for s in sig if tier == 'all' or s['tier'] == tier]
            h = [harmed_helped(s, m)[0] for s in ss]
            g = [harmed_helped(s, m)[1] for s in ss]
            mask = [(hh is not None and gg is not None and (hh or gg)) for hh, gg in zip(h, g)]
            sel = [s for s, k in zip(ss, mask) if k]
            labels = [harmed_helped(s, m)[0] for s in sel]
            if len(set(labels)) < 2:
                for name in ['entropy', 'jsd']:
                    rows.append({'model': model, 'method': m, 'tier': tier, 'signal': name,
                                 'auc': float('nan'), 'n_harmed': int(np.sum(labels)), 'n_helped': len(labels) - int(np.sum(labels))})
                continue
            for name in ['entropy', 'jsd']:
                a, nh, ng = auc_score([s[name] for s in sel], labels)
                rows.append({'model': model, 'method': m, 'tier': tier, 'signal': name,
                             'auc': round(a, 4), 'n_harmed': nh, 'n_helped': ng})
            X = np.array([[s['entropy'], s['jsd']] for s in sel])
            clf = tiny_logistic(X, np.array(labels))
            a, _, _ = auc_score(clf(X), labels)
            rows.append({'model': model, 'method': m, 'tier': tier, 'signal': 'logistic_combo',
                         'auc': round(a, 4), 'n_harmed': int(np.sum(labels)),
                         'n_helped': len(labels) - int(np.sum(labels))})
    return rows


# ---------------------------------------------------------------- judgment 3
def apply_rule(sig_row, m_star, tau_h, tau_j):
    if sig_row['entropy'] > tau_h:
        return 'abstain'
    if sig_row['jsd'] > tau_j:
        return m_star
    return 'direct'


def build_judgment3(recs_naming, model):
    """Thresholds tuned ONLY on high tier (grid over deciles), objective:
    accuracy - certain_error. Also a median-threshold variant. Plus ideal rule."""
    sig_all = gather_signals(recs_naming)
    high = [s for s in sig_all if s['tier'] == 'high']
    low = [s for s in sig_all if s['tier'] == 'low']

    # choose m* on HIGH tier: best accuracy gain vs direct
    gains = {}
    for m in METHODS[1:]:
        pairs = [(s.get(m), s.get('direct')) for s in high]
        pairs = [(a, b) for a, b in pairs if a and b]
        gains[m] = (np.mean([a == 'correct' for a, b in pairs]) -
                    np.mean([b == 'correct' for a, b in pairs])) if pairs else -1
    m_star = max(gains, key=gains.get)

    ent_h = sorted(s['entropy'] for s in high)
    jsd_h = sorted(s['jsd'] for s in high)
    qs = [i / 10 for i in range(0, 11)]

    def eval_on(ss, tau_h, tau_j):
        out_rows = []
        for s in ss:
            act = apply_rule(s, m_star, tau_h, tau_j)
            oc = 'abstain' if act == 'abstain' else s.get(act, s['direct'])
            out_rows.append(oc)
        met = scoring.metrics([{'outcome': o} for o in out_rows])
        return met

    best = None
    for qh in qs:
        for qj in qs:
            tau_h = np.quantile([s['entropy'] for s in high], qh)
            tau_j = np.quantile([s['jsd'] for s in high], qj)
            met = eval_on(high, tau_h, tau_j)
            obj = met['accuracy'] - met['certain_error_rate']
            if best is None or obj > best[0]:
                best = (obj, tau_h, tau_j, qh, qj)
    _, tau_h, tau_j, qh, qj = best

    # median variant (all samples, no gold used)
    tau_h_med = float(np.median([s['entropy'] for s in sig_all]))
    tau_j_med = float(np.median([s['jsd'] for s in sig_all]))

    direct_met = {tier: scoring.metrics([{'outcome': s['direct']} for s in ss])
                  for tier, ss in [('high', high), ('low', low)]}

    rows = []
    for rule_name, th, tj in [('remedy_high_tuned', tau_h, tau_j), ('remedy_median', tau_h_med, tau_j_med)]:
        for tier, ss in [('high', high), ('low', low)]:
            met = eval_on(ss, th, tj)
            dmet = direct_met[tier]
            rows.append({'model': model, 'rule': rule_name, 'm_star': m_star,
                         'tau_entropy': round(th, 4), 'tau_jsd': round(tj, 4), 'tier': tier,
                         'n': met['n'], 'accuracy': round(met['accuracy'], 4),
                         'abstain_rate': round(met['abstain_rate'], 4),
                         'certain_error_rate': round(met['certain_error_rate'], 4),
                         'd_err_vs_direct': round(met['certain_error_rate'] - dmet['certain_error_rate'], 4),
                         'd_acc_vs_direct': round(met['accuracy'] - dmet['accuracy'], 4)})
    # ideal rule
    for tier, ss in [('high', high), ('low', low)]:
        outs = []
        for s in ss:
            if s['direct'] == 'correct':
                outs.append('correct')
            elif any(s.get(m) == 'correct' for m in METHODS[1:]):
                outs.append('correct')
            else:
                outs.append('abstain')
        met = scoring.metrics([{'outcome': o} for o in outs])
        dmet = direct_met[tier]
        rows.append({'model': model, 'rule': 'ideal', 'm_star': '', 'tau_entropy': '', 'tau_jsd': '',
                     'tier': tier, 'n': met['n'], 'accuracy': round(met['accuracy'], 4),
                     'abstain_rate': round(met['abstain_rate'], 4),
                     'certain_error_rate': round(met['certain_error_rate'], 4),
                     'd_err_vs_direct': round(met['certain_error_rate'] - dmet['certain_error_rate'], 4),
                     'd_acc_vs_direct': round(met['accuracy'] - dmet['accuracy'], 4)})
    info = {'m_star': m_star, 'm_star_high_gains': {k: round(v, 4) for k, v in gains.items()},
            'tau_entropy': tau_h, 'tau_jsd': tau_j, 'q_entropy': qh, 'q_jsd': qj}
    return rows, info


# ---------------------------------------------------------------- examples
def find_examples(recs_naming, model, k=5):
    by_sid = defaultdict(dict)
    for r in recs_naming:
        by_sid[r['sample_id']][r['method']] = r
    ex = []
    for sid, d in by_sid.items():
        if 'direct' not in d:
            continue
        dr = d['direct']
        for m in METHODS[1:]:
            if m not in d:
                continue
            mr = d[m]
            if dr['outcome'] in ('correct', 'abstain') and mr['outcome'] in ('wrong', 'wrong_unparsed'):
                ex.append({'model': model, 'image_id': dr['image_id'], 'sample_id': sid,
                           'tier': dr['tier'], 'gold': dr['gold'], 'method': m,
                           'direct_out': dr['text'].strip(), 'direct_outcome': dr['outcome'],
                           'method_out': mr['text'].strip(),
                           'image_path': str(ROOT / 'data' / 'images' / f"{dr['image_id']}.jpg"),
                           'prompt': dr['prompt']})
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

    load_synonyms()
    import csv
    tabs = ROOT / 'outputs' / 'tables'
    tabs.mkdir(parents=True, exist_ok=True)
    summary = {'models': models, 'generated': time.strftime('%Y-%m-%d %H:%M:%S')}

    all_main, all_effects, all_j2, all_j3, all_ex = [], [], [], [], []
    for model in models:
        recs = load_recs(model, args.tag)
        recs = {t: rescore(r, t) for t, r in recs.items()}
        summary.setdefault('counts', {})[model] = {t: len(r) for t, r in recs.items()}
        all_main += build_main(recs, model)
        all_effects += build_effects(recs, model)
        all_j2 += build_judgment2(recs['naming'], model)
        j3rows, j3info = build_judgment3(recs['naming'], model)
        all_j3 += j3rows
        summary.setdefault('judgment3_info', {})[model] = j3info
        all_ex += find_examples(recs['naming'], model)

    def dump(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(tabs / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    dump('main.csv', all_main)
    dump('effects.csv', all_effects)
    dump('judgment2_auc.csv', all_j2)
    dump('remedy.csv', all_j3)
    summary['examples'] = all_ex[:20]
    json.dump(summary, open(tabs / 'analysis_summary.json', 'w'), indent=1, ensure_ascii=False)
    print(f'main rows: {len(all_main)}, effects: {len(all_effects)}, j2: {len(all_j2)}, j3: {len(all_j3)}')
    print('tables written')


if __name__ == '__main__':
    main()
