"""Load probe: verify a candidate model builds inputs, decodes, and supports LCD."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl')]:
    os.environ[var] = str(ROOT/'cache'/sub)
sys.path.insert(0, str(ROOT/'code'))
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--key', required=True)
ap.add_argument('--path', required=True)
ap.add_argument('--gpu', type=int, required=True)
a = ap.parse_args()
import torch
from PIL import Image
from stage3_engine import get_engine
from prompts import STYLE1
t0=time.time()
try:
    em = get_engine(a.path, f"cuda:{a.gpu}")
    print(f'[{a.key}] loaded {em.mt} in {time.time()-t0:.0f}s lcd_ready={em.lcd_ready} norm={getattr(em,"norm_path",None)}', flush=True)
except Exception as e:
    print(f'[{a.key}] LOAD FAIL: {str(e)[:300]}'); sys.exit(1)
img = Image.open(ROOT/'data/images/pizza_eval_024.jpg').convert('RGB')
try:
    inp = em.build(img, STYLE1)
    r = em.decode_single(inp, 12)
    print(f'[{a.key}] direct decode ok: {r["text"]!r} conf={r["answer_maxp"]:.3f} {r["wall_s"]:.2f}s prefill={r["n_prefill_tokens"]}', flush=True)
except Exception as e:
    print(f'[{a.key}] BUILD/DECODE FAIL: {str(e)[:300]}'); sys.exit(2)
try:
    r2 = em.decode_single(em.build(img, STYLE1), 12, lcd=True)
    print(f'[{a.key}] lcd decode ok: {r2["text"]!r} {r2["wall_s"]:.2f}s', flush=True)
except Exception as e:
    print(f'[{a.key}] LCD FAIL: {str(e)[:300]}')
try:
    from engine import make_variants
    v = make_variants(img)
    r3 = em.decode_contrastive(em.build(v['clean'], STYLE1), em.build(v['blur'], STYLE1), 'mib', 12)
    print(f'[{a.key}] mib decode ok: {r3["text"]!r} conf={r3["answer_maxp"]:.3f}', flush=True)
except Exception as e:
    print(f'[{a.key}] CONTRASTIVE FAIL: {str(e)[:300]}')
print(f'[{a.key}] PROBE DONE')
