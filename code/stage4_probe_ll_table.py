"""Stage 4 deliverable: probe_ll.csv — per-class and overall LL-probe results
(image + blank conditions, rank stats) per pair."""
import json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path('/home/g203-4028/projects/knowledge-deficit-mitigation')
pairs = [('q4b','food101'), ('q9b','food101'), ('q3vl4b','food101'),
         ('llava16','food101'), ('internvl4b','food101')]
rows = []
for model, dom in pairs:
    P = {}
    for l in open(ROOT/f'outputs/raw/{model}_stage4_probe_{dom}.jsonl'):
        r = json.loads(l); P[r['file']] = r
    sp = ROOT/f'data/strata_{model}.json' if model in ('q4b','q9b') else ROOT/f'data/strata_{model}_{dom}.json'
    strata = json.load(open(sp))
    st_of = {c: 'deficient' for c in strata['deficient']}
    st_of.update({c: 'known' for c in strata['known']})
    by = defaultdict(list)
    for f, r in P.items():
        by[(st_of[r['class']], r['class'])].append(r)
    for (st, c), rs in sorted(by.items()):
        rows.append({'pair': f'{model}@{dom}', 'stratum': st, 'class': c, 'n': len(rs),
                     'probe_acc_image': round(float(np.mean([r['correct_image'] for r in rs])), 4),
                     'probe_acc_blank': round(float(np.mean([r['correct_blank'] for r in rs])), 4),
                     'gold_rank_image_mean': round(float(np.mean([r['rank_image'] for r in rs])), 2),
                     'rank_diff_mean': round(float(np.mean([r['rank_blank'] - r['rank_image'] for r in rs])), 2)})
    for st in ['known', 'deficient']:
        rs = [r for f, r in P.items() if st_of[r['class']] == st]
        rows.append({'pair': f'{model}@{dom}', 'stratum': st, 'class': '(overall)', 'n': len(rs),
                     'probe_acc_image': round(float(np.mean([r['correct_image'] for r in rs])), 4),
                     'probe_acc_blank': round(float(np.mean([r['correct_blank'] for r in rs])), 4),
                     'gold_rank_image_mean': round(float(np.mean([r['rank_image'] for r in rs])), 2),
                     'rank_diff_mean': round(float(np.mean([r['rank_blank'] - r['rank_image'] for r in rs])), 2)})
with open(ROOT/'outputs/tables/probe_ll.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('probe_ll.csv written:', len(rows), 'rows')
