"""Append stage-6 information to run_manifest.json."""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / 'run_manifest.json'))

GPU_MAP = {'q4b': 2, 'q9b': 3, 'llava16': 4, 'internvl4b': 5}
durations = {}
for model in GPU_MAP:
    p = ROOT / 'logs' / f'stage6_{model}.log'
    if p.exists():
        for line in open(p):
            m = re.search(r'DONE (.+?) in ([0-9.]+) min', line)
            if m:
                durations[model] = {'parts': m.group(1), 'min': float(m.group(2))}

man['stage6'] = {
    'date': '2026-09-15',
    'preregister': 'docs/stage6/PREREGISTER_STAGE6.md',
    'purpose': 'faithful reimplementation of official decoding methods + '
               'sharpening controls + constant-offset correction test; '
               'no new measurement schemes, no new methods proposed',
    'methods': {
        'vcd': 'official DAMO-NLP-SG/VCD@d6568ff: sigmoid-schedule diffusion noise '
               'on processor pixel_values (t=500), (1+a)z_clean - a z_noise, '
               'beta=0.1 plausibility mask; greedy (deviation: official samples)',
        'm3id': 'paper arXiv:2403.14003 Alg.1/Eq.(4) (no official code): '
                'l* = l_c + gate(max p_c<0.3) * (1-g)/g * (l_c - l_u), '
                'g=exp(-0.02(t+t0)), t0=question-token-count, text-only prior branch',
        'dola': 'official voidism/DoLa@805230e dola_greedy_decode: candidates '
                'hidden[0..L-1] via lm_head WITHOUT final norm, per-step argmax '
                'JSD(final,candidate), relative_top=0.1 on final, '
                'z*=masked_final - log_softmax(prem) with -1e3 clamp',
        'deco': 'official zjunlp/DeCo@c1a9129 deco_greedy_search: candidate layers '
                '[ceil(0.625L),floor(0.875L)] via lm_head AFTER final norm, '
                'final top-k=20 truncated at cumulative top-p=0.9, '
                'z*=z_final+0.6*maxprob*z_sel, candidates-only mask',
        'sid': 'BLOCKED: official fork requires transformers 4.29.2 (project uses '
               '5.17.0 for Qwen3.5); supports only llava-1.5/instructblip/shikra; '
               'CT2S needs attention weights at layers 0-2 which are linear-attention '
               '(no attention matrices) in Qwen3.5 - reported, not silently replaced',
    },
    'legacy_local_names': {'mib': 'variant implementation, not M3ID; kept as supplement',
                           'lcd': 'variant implementation, differs from DoLa layer selection'},
    'runs': {},
    'gpu_allocation': GPU_MAP,
    'recording': 'answer_maxp/step_probs under the final distribution each config '
                 'used (post-contrast, post-truncation, renormalized); direct runs '
                 'additionally store the full first-token distribution (float16 npz)',
    'stage6_outputs': sorted(str(p.relative_to(ROOT)) for p in
                             (ROOT / 'outputs').rglob('*stage6*')),
    'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
for model, d in durations.items():
    man['stage6']['runs'][model] = {
        'gpu': GPU_MAP[model], 'log': f'logs/stage6_{model}.log', **d}

json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
print('run_manifest.json updated with stage6 block; durations:', durations)
