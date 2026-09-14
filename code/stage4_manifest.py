"""Append stage-4 information to run_manifest.json."""
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / 'run_manifest.json'))

man['stage4'] = {
    'date': '2026-09-14',
    'preregister': 'PREREGISTER_STAGE4.md',
    'probe': {
        'definition': 'name-restricted sum log-likelihood ranking under the open-set '
                      'naming prompt; teacher-forced scoring against a shared prompt '
                      'KV cache (per-candidate deepcopy), exact lp0 pruning; '
                      'blank-image control on the same samples',
        'validity_gate': 'known-stratum probe accuracy >= 0.90 AND blank-control rank '
                         'diff CI > 0 (uniform across models)',
    },
    'models_probed': {
        'q4b': {'known_acc': 0.7667, 'gate': 'FAIL', 'fullname_conditional': 1.0},
        'q9b': {'known_acc': 0.7033, 'gate': 'FAIL', 'fullname_conditional': 0.9897},
        'q3vl4b': {'known_acc': 0.6067, 'gate': 'FAIL', 'fullname_conditional': 1.0},
        'llava16': {'known_acc': 0.52, 'gate': 'FAIL', 'fullname_conditional': 0.9533},
        'internvl4b': {'known_acc': 0.4167, 'gate': 'FAIL', 'fullname_conditional': 0.6701},
        'glm46v': 'probe + naming subset only (see STAGE4.md section 7)',
    },
    'outcome': 'primary probe failed the pre-registered validity gate on ALL models; '
               'confirmatory set (llava16, internvl4b) invalid -> no confirmatory '
               'judgments executed; conditional granularity diagnosis: probe accuracy '
               '~1.0 among full-name answerers; blank control passes everywhere',
    'dropped': {'dogs_probes': 'cut after the probe failed its gate on all models; '
                               'does not change any verdict (task sec-7 priority)'},
    'stage4_outputs': sorted(str(p.relative_to(ROOT)) for p in
                             (ROOT / 'outputs').rglob('*stage4*')),
    'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
print('run_manifest.json updated with stage4 block')
