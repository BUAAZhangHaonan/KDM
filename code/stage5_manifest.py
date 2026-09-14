"""Append stage-5 information to run_manifest.json."""
import json, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / 'run_manifest.json'))

def log_done_minutes(log):
    p = ROOT / 'logs' / log
    if not p.exists():
        return None
    m = re.search(r'DONE in ([0-9.]+) min', p.read_text())
    return float(m.group(1)) if m else None

man['stage5'] = {
    'date': '2026-09-15',
    'preregister': 'PREREGISTER_STAGE5.md',
    'goal': 'unify conclusions into one main line; test the pre-registered '
            'base-rate relation (dECE predictable from pre-intervention '
            'accuracy / calibration gap, with a crossing point)',
    'new_inference': {
        'food101_middle': {
            'classes': 51, 'samples_per_class': 24, 'configs': ['direct', 'vcd', 'mib', 'lcd'],
            'runner': 'code/stage5_middle.py (stage-2 engine path, settings unchanged)',
            'duration_min': {'q4b': log_done_minutes('stage5_middle_q4b.log'),
                             'q9b': log_done_minutes('stage5_middle_q9b.log')},
            'records': ['outputs/raw/q4b_s5middle_naming.jsonl',
                        'outputs/raw/q9b_s5middle_naming.jsonl']},
        'dogs_middle': {
            'classes': 60, 'samples_per_class': 24, 'configs': ['direct', 'vcd', 'mib', 'lcd'],
            'runner': 'code/stage5_middle_dogs.py (stage-3 engine path, settings unchanged)',
            'duration_min': {'q4b': log_done_minutes('stage5_middle_q4b_dogs.log'),
                             'q9b': log_done_minutes('stage5_middle_q9b_dogs.log')},
            'records': ['outputs/raw/q4b_s5middle_dogs.jsonl',
                        'outputs/raw/q9b_s5middle_dogs.jsonl']},
    },
    'reanalysis': ['code/stage5_nonqwen.py (non-Qwen main-line quantities from existing '
                   'records; no probe dependency)',
                   'code/stage5_baserate.py (base-rate relation, both population '
                   'constructions, class-cluster bootstrap)'],
    'strata_renaming': 'deficient -> low_acc (低准确率组), known -> high_acc (高准确率组); '
                       'the low stratum mixes name-accessible-but-not-emitted samples '
                       'and is NOT equivalent to "model does not know the name"',
    'stage5_outputs': sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'outputs').rglob('*')
                             if 'stage5' in p.name or p.name in
                             ('baserate.csv', 'crossing.csv', 'nonqwen_core.csv',
                              'related_work.csv', 'fig9.pdf')),
    'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
print('run_manifest.json updated with stage5 block')
