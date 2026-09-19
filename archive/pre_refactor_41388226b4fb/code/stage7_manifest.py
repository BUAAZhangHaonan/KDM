"""Append stage-7 information to run_manifest.json."""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / 'run_manifest.json'))

durations = {}
for model in ('q4b', 'q9b'):
    p = ROOT / 'logs' / f'stage7_{model}.log'
    if p.exists():
        for line in open(p):
            m = re.search(r'\[(\S+) (\S+)\] DONE in ([0-9.]+) min', line)
            if m:
                durations.setdefault(model, {})[m.group(2)] = float(m.group(3))
p = ROOT / 'logs' / 'stage7_sid_llava16.log'
if p.exists():
    for line in open(p):
        m = re.search(r'\[llava16 sid\] DONE in ([0-9.]+) min', line)
        if m:
            durations.setdefault('llava16', {})['sid'] = float(m.group(1))

man['stage7'] = {
    'date': '2026-09-16',
    'preregister': 'docs/stage7/PREREGISTER_STAGE7.md (commit 30d9528, before any inference '
                   'or judgment computation)',
    'purpose': 'final round: wording-deciding supplementary analyses, fidelity '
               'close-out (SID bounded reproduction, robustness, dogs faithful '
               'rerun), evidence map + mainline; no new measurements or methods',
    'branch_decision': 'strong wording branch (cell-wise sharpening consistency: '
                       '11 explainable vs 12 residual-structure cells of 23 judged)',
    'extra_env': {
        'venv_sid': 'project-local conda-prefix env (python 3.11) with the '
                    'official SID repo transformers 4.29.2 installed; used ONLY '
                    'to generate official selection-logic reference outputs for '
                    'the unit check (stage7_sid_reference.py, sha256 recorded)'},
    'runs': {
        'q4b': {'gpu': 2, 'jobs': ['seed (2 salts)', 'dola_subset', 'dogs'],
                'done_min_cumulative': durations.get('q4b', {}).get('dogs')},
        'q9b': {'gpu': 3, 'jobs': ['seed (2 salts)', 'dola_subset', 'dogs'],
                'done_min_cumulative': durations.get('q9b', {}).get('dogs')},
        'llava16_sid': {'gpu': 4, 'done_min': durations.get('llava16', {}).get('sid')},
        'm3idgate_replay': {'gpu': '4/5', 'note': 'conditional-branch replay over '
                            'stored tokens, ~6 min per model'},
        'internvl4b_sid': {'status': 'runtime-incompatible: remote attention '
                           'hardwired to packed flash-varlen; unit check passed, '
                           'keep-mask not injectable'},
    },
    'key_results': {
        'cellwise_consistency': '11 explainable / 12 residual (23 judged, '
                                '9 insufficient-n) -> strong branch',
        'distlevel': 'Spearman median 0.352, W1 median 0.110',
        'positive_control_MIB': 'q4b low acc 0.179->0.335 dECE -0.061 (sign flip '
                                'demonstrated); q9b 0.247->0.358 dECE +0.041',
        'dual_caliber': 'matchable 28: 20/7/1; all 32: 20/7/5',
        'drift': 'mean -0.001 sd 0.027 range [-0.060,+0.038]',
        'sid_llava16': 'dconf +0.390/+0.308, dECE +0.361/-0.157, DiD +0.518; '
                       '12/12 unit checks vs official reference',
        'seed_robustness': 'dconf spread <= 0.021 (<= 11.8% of effect), '
                           'directions stable across 3 draws',
        'dola_subset': 'DiD 0.256/0.266 vs all-layer 0.255/0.297 (unchanged)',
        'm3id_gate_closed': 'q4b 97.2%/85.9%, q9b 98.0%/87.4% (high/low strata)',
        'dogs_prediction': '16/16 cells hit (dECE >= 0), 0 misses',
    },
    'stage7_outputs': sorted(str(p.relative_to(ROOT)) for p in
                             (ROOT / 'outputs').rglob('*stage7*')),
    'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
print('run_manifest.json updated with stage7 block; durations:', durations)
