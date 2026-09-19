"""Fill firstdist npz entries for samples whose direct decode ran in an earlier
process (sanity) and whose vector was overwritten by the later run's dump.

Recomputes ONLY the first-token distribution via a single prefill forward,
identical to what decode_single captured at t=0 (greedy, clean image).
Usage: ./venv/bin/python code/stage6_npz_repair.py --model q9b --gpu 3 --part eval
"""
import os, sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {
    'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
    'q9b': '/home/g203-4028/Models/Qwen3.5-9B',
    'llava16': '/home/g203-4028/Models/llava-v1.6-mistral-7b-hf',
    'internvl4b': '/home/g203-4028/Models/InternVL3_5-4B',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--part', default='eval')
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from stage6_engine import S6Model
    from stage6_run import MODELS as RUN_MODELS  # path + style

    recs = {}
    for line in open(ROOT / 'outputs' / 'raw' /
                     f'{args.model}_s6_{args.part}_naming.jsonl'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('method') == 'direct' and 'error' not in r and r.get('tokens'):
            recs[r['file']] = r
    npz_path = ROOT / 'outputs' / 'raw' / f'{args.model}_s6_firstdist_{args.part}.npz'
    have = set(np.load(npz_path).files) if npz_path.exists() else set()
    missing = [f for f in sorted(recs) if f not in have]
    print(f'{args.model} {args.part}: {len(missing)} missing vectors')
    if not missing:
        return
    em = S6Model(MODELS[args.model], f'cuda:{args.gpu}')
    man = {m['file']: m for m in
           (json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl'))}
    out = {k: v for k, v in np.load(npz_path).items()} if npz_path.exists() else {}
    for f in missing:
        m = man[f]
        img = Image.open(m['image_path']).convert('RGB')
        inputs = em.build(img, recs[f]['prompt'])
        with torch.no_grad():
            o = em._fwd(inputs=inputs)
            p = F.softmax(o.logits[:, -1, :].float(), -1)
        out[f] = p[0].detach().cpu().half().numpy()
        assert int(p[0].argmax()) == recs[f]['tokens'][0], f'token mismatch {f}'
    np.savez(npz_path, **out)
    print(f'repaired {len(missing)}; npz now {len(out)} entries')


if __name__ == '__main__':
    main()
