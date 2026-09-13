"""Phase-1 analysis: strata validation on the eval half + abstention pre-test
decision. Writes strata_validation.csv and primary_outcome_decision_{model}.json.

Validation PASS (per model) iff:
  (a) class-level Welch t 95% CI for mean_acc(known) - mean_acc(deficient)
      excludes 0, AND
  (b) mean eval-half accuracy of the deficient stratum <= 0.25 (near floor).
"""
import os, sys, json, math, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))

ABSTAIN_FLOOR = 0.10
DEFICIENT_ACC_MAX = 0.25


def welch_ci(a, b, alpha=0.05):
    from scipy import stats as sps
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = sps.t.ppf(1 - alpha / 2, dof)
    return d, d - tc * se, d + tc * se


def main():
    models = sys.argv[1].split(',') if len(sys.argv) > 1 else ['q4b', 'q9b']
    rows = []
    for model in models:
        strata = json.load(open(ROOT / 'data' / f'strata_{model}.json'))
        # group-half accuracies (for the record)
        gacc = strata['group_acc']

        erecs = [json.loads(l) for l in open(ROOT / 'outputs' / 'raw' / f'{model}_phase1_eval.jsonl')]
        cls_acc = defaultdict(list)
        img = defaultdict(lambda: [0, 0])
        for r in erecs:
            cls_acc[r['class']].append(r['outcome'] == 'correct')
            img[r['stratum']][0] += (r['outcome'] == 'correct')
            img[r['stratum']][1] += 1
        eacc = {c: np.mean(v) for c, v in cls_acc.items()}
        known = [eacc[c] for c in strata['known'] if c in eacc]
        defic = [eacc[c] for c in strata['deficient'] if c in eacc]
        g_known = [gacc[c] for c in strata['known']]
        g_defic = [gacc[c] for c in strata['deficient']]
        d, lo, hi = welch_ci(known, defic)
        img_known = img['known'][0] / img['known'][1]
        img_defic = img['deficient'][0] / img['deficient'][1]
        passed = bool((lo > 0) and (np.mean(defic) <= DEFICIENT_ACC_MAX))

        # abstention pre-test
        arecs = [json.loads(l) for l in open(ROOT / 'outputs' / 'raw' / f'{model}_phase1_abstain.jsonl')]
        abst = defaultdict(lambda: [0, 0])
        for r in arecs:
            abst[r['style']][0] += bool(r['abstained'])
            abst[r['style']][1] += 1
        abst_rates = {s: v[0] / v[1] for s, v in abst.items() if v[1]}
        best_style = max(abst_rates, key=abst_rates.get) if abst_rates else None
        usable = bool(best_style and abst_rates.get(best_style, 0) >= ABSTAIN_FLOOR)
        json.dump({'model': model, 'abstain_usable': usable,
                   'chosen_style': best_style if usable else 'style1',
                   'abstain_rates': abst_rates,
                   'primary_outcome': 'abstention+confidence' if usable else 'confidence_and_calibration'},
                  open(ROOT / 'data' / f'primary_outcome_decision_{model}.json', 'w'), indent=1)

        rows.append({'model': model,
                     'n_known_classes': len(strata['known']),
                     'n_deficient_classes': len(strata['deficient']),
                     'group_acc_known': round(float(np.mean(g_known)), 4),
                     'group_acc_deficient': round(float(np.mean(g_defic)), 4),
                     'eval_acc_known': round(float(np.mean(known)), 4),
                     'eval_acc_deficient': round(float(np.mean(defic)), 4),
                     'eval_diff': round(float(d), 4),
                     'diff_lo': round(float(lo), 4), 'diff_hi': round(float(hi), 4),
                     'img_acc_known': round(img_known, 4),
                     'img_acc_deficient': round(img_defic, 4),
                     'PASS': passed,
                     'abstain_style1': round(abst_rates.get('style1', float('nan')), 4),
                     'abstain_style2': round(abst_rates.get('style2', float('nan')), 4),
                     'abstain_style3': round(abst_rates.get('style3', float('nan')), 4),
                     'abstain_usable': usable,
                     'chosen_style': best_style if usable else '(dropped->style1)'})
        print(json.dumps(rows[-1], indent=1))

    import csv
    keys = list(rows[0].keys())
    with open(ROOT / 'outputs' / 'tables' / 'strata_validation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('strata_validation.csv written')


if __name__ == '__main__':
    main()
