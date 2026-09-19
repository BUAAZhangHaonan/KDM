"""GLM probe on the naming-subset files only (targeted, resumable)."""
import sys, json, time
sys.path.insert(0,'/home/g203-4028/projects/knowledge-deficit-mitigation/code')
import torch
from PIL import Image
from stage3_engine import get_engine
from stage4_probe import probe_ranks
from engine import make_variants
from prompts import STYLE1
em = get_engine('/home/g203-4028/Models/GLM-4.6V-Flash','cuda:4')
tok = em.proc.tokenizer
classes = sorted({json.loads(l)['class'] for l in open('/home/g203-4028/projects/knowledge-deficit-mitigation/data/samples_manifest.jsonl')})
cand_names=[c.replace('_',' ') for c in classes]
cand_ids=[tok(' '+n, add_special_tokens=False)['input_ids'] for n in cand_names]
manifest={m['file']:m for m in map(json.loads,open('/home/g203-4028/projects/knowledge-deficit-mitigation/data/samples_manifest.jsonl'))}
nam=[json.loads(l) for l in open('/home/g203-4028/projects/knowledge-deficit-mitigation/outputs/raw/glm46v_stage4_naming_food101.jsonl')][:90]
p='/home/g203-4028/projects/knowledge-deficit-mitigation/outputs/raw/glm46v_stage4_probe_food101.jsonl'
done={json.loads(l)['key'] for l in open(p)}
t0=time.time(); n=0
with open(p,'a') as fo:
    for r in nam:
        key=f"pr4:{r['file']}"
        if key in done: continue
        m=manifest[r['file']]
        img=Image.open(m['image_path']).convert('RGB')
        blank=make_variants(img)['blank']
        gi=classes.index(r['class'])
        ri, li, ti, _=probe_ranks(em, img, STYLE1, cand_ids, cand_names, gi, False)
        rb, lb, tb, _=probe_ranks(em, blank, STYLE1, cand_ids, cand_names, gi, False)
        fo.write(json.dumps({'key':key,'model':'glm46v','domain':'food101','stratum':r['stratum'],
            'class':r['class'],'file':r['file'],'gold_idx':gi,'rank_image':ri,'rank_blank':rb,
            'correct_image':ri==0,'correct_blank':rb==0,'top_image':ti,'top_blank':tb,
            'll_gold_image':round(li,4),'ll_gold_blank':round(lb,4)})+'\n')
        n+=1
        if n%10==0: print(f'{n} {time.time()-t0:.0f}s', flush=True)
print('DONE', flush=True)
