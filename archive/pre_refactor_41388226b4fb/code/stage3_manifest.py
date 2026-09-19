"""Append stage-3 information to run_manifest.json."""
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / 'run_manifest.json'))

man['stage3'] = {
    'date': '2026-09-13/14',
    'preregister': 'docs/stage3/PREREGISTER_STAGE3.md',
    'closedset': {
        'task': '101-option (food101) / 120-option (dogs) number-choice, '
                'per-sample deterministic candidate shuffle seed f"{file}|{variant}"',
        'variants': ['full101 (all classes)', 'near10 (gold + 9 WordNet Wu-Palmer neighbours)'],
        'scoring': 'first integer in reply == gold position; unparsed counts as wrong',
    },
    'models_extended': {
        'q3vl4b': {'path': '/home/g203-4028/Models/Qwen3-VL-4B-Instruct',
                   'notes': 'Conv3d->matmul patch (verified 0.0078 bf16), max-side 640px input cap'},
        'llava16': {'path': '/home/g203-4028/Models/llava-v1.6-mistral-7b-hf', 'notes': 'native anyres'},
        'internvl4b': {'path': '/home/g203-4028/Models/InternVL3_5-4B',
                       'notes': 'remote code; manual ChatML template, single 448 crop, '
                                'all_tied_weights_keys patch, timm installed'},
        'excluded': {
            'glm46v': 'loads and decodes but 16.5 s/decode steady-state (measured); '
                      'full pipeline ~2 days/GPU, excluded per task section 9',
            'MiniCPM-V-2_6': 'weights incomplete (48K)',
            'Ministral-3/8B': 'missing dirs; text-only',
            'Qwen3.5-35B-A3B': '67G bf16 exceeds single 3090',
        },
    },
    'domains': {
        'food101': {'source': 'ethz/food101 (stage-2 manifest reused)'},
        'dogs': {'source': 'Donghyun99/Stanford-Dogs (HF parquet)', 'n_classes': 120,
                 'per_class_min': 148, 'n_group': 2880, 'n_eval': 2880, 'seed': 42},
    },
    'stage3_outputs': sorted(str(p.relative_to(ROOT)) for p in
                             (ROOT / 'outputs').rglob('*stage3*')),
    'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
print('run_manifest.json updated with stage3 block')
