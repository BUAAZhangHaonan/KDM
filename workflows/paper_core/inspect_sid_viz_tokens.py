"""CPU-only exact native-processor SID rank100 applicability on Viz512."""
import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from kdm.io import atomic_json, file_hash
from kdm.execution import resolve_image_path
from kdm.models.backbone import FamilyModel
from kdm.prompts import task_prompt
from workflows.paper_core.dev_viz import roster
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
torch.set_num_threads(2)
spec=json.loads((ROOT/'configs/runtime'/f'{a.model}.json').read_text())
engine=object.__new__(FamilyModel)
engine.device='cpu';engine.cfg=AutoConfig.from_pretrained(spec['kwargs']['model_path'],trust_remote_code=True)
engine.mt=engine.cfg.model_type
engine.proc=AutoProcessor.from_pretrained(spec['kwargs']['model_path'],trust_remote_code=True)
tok=getattr(engine.cfg,'image_token_id',getattr(engine.cfg,'image_token_index',None))
assert tok is not None
roster_path,samples=roster('viz512');out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
rows=[];start=time.monotonic()
for i,sample in enumerate(samples):
    path=resolve_image_path(sample['image_path'],ROOT)
    with Image.open(path) as f:
        image=f.convert('RGB');width,height=image.size
        inputs=engine.build(image,task_prompt(sample['question'],guided=False))
    positions=(inputs['input_ids'][0]==tok).nonzero(as_tuple=True)[0]
    n=int(len(positions));contiguous=bool(n and int(positions[-1])-int(positions[0])+1==n)
    rows.append(dict(sample_id=sample['id'],source_image=sample['image_path'],width=width,height=height,
        visual_tokens=n,contiguous=contiguous,fixed_rank100_applicable=n>=100 and contiguous,
        image_grid_thw=inputs.get('image_grid_thw').tolist() if 'image_grid_thw' in inputs else None))
    if i==0 or (i+1)%64==0:
        atomic_json(out/'progress.json',dict(completed=i+1,expected=512,first=rows[0],elapsed_s=time.monotonic()-start))
        print(json.dumps({'completed':i+1,'expected':512,'last_tokens':n}),flush=True)
assert not torch.cuda.is_initialized()
with (out/'native_visual_tokens.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=dict(model=a.model,n=len(rows),native_token_distribution=dict(sorted(Counter(r['visual_tokens'] for r in rows).items())),
    below100=sum(r['visual_tokens']<100 for r in rows),noncontiguous=sum(not r['contiguous'] for r in rows),
    first_input=rows[0],roster_sha256=file_hash(roster_path),processor_source='original FamilyModel.build on CPU',
    cuda_initialized=False,model_weights_loaded=False,algorithm_changed=False,elapsed_s=time.monotonic()-start)
atomic_json(out/'complete.json',summary);print(json.dumps(summary),flush=True)
