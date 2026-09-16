"""Explicitly synthetic fixtures. These are never evidence about KDM models."""
from pathlib import Path
import json,sys,io
from types import SimpleNamespace
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from stage9_common import write_json
from stage9_geometry import softmax


def cache_fixture(root:Path):
    rng=np.random.default_rng(912)
    folder=root/'outputs/raw';folder.mkdir(parents=True,exist_ok=True)
    for model in ('q9b','llava16'):
        rows=[]
        for c in range(20):
            st='low_acc' if c<10 else 'high_acc'
            for i in range(4):
                f=f'fixture_{c:02}_{i}.jpg';correct=bool(rng.random() < (.35 if c<10 else .8))
                base=float(rng.uniform(.25,.85));common=rng.normal(0,.04)
                for method in ('direct','vcd','m3id','dola','deco'):
                    change=method!='direct' and rng.random()<.12
                    y=not correct if change else correct
                    p=float(np.clip(base+(0 if method=='direct' else .1)+common,.01,.99))
                    rows.append(dict(key=f'{f}:{method}',model=model,task='naming',method=method,
                                     stratum=st,**{'class':f'c{c:02}'},file=f,
                                     text=f'c{c:02}' if y else 'other',outcome='correct' if y else 'wrong',
                                     tokens=[1,2],step_probs=[p,.9],answer_maxp=p,abstained=False,
                                     synthetic_fixture=True))
        with (folder/f'{model}_s6_eval_naming.jsonl').open('w') as h:
            for r in rows:h.write(json.dumps(r)+'\n')
    write_json(root/'FIXTURE_NOTICE.json',dict(kind='synthetic_test_fixture',seed=912,scientific_evidence=False))


def image_fixture(root:Path):
    data=root/'data';data.mkdir(parents=True,exist_ok=True);manifest=[]
    classes=[f'c{i:02}' for i in range(8)]
    for c,cls in enumerate(classes):
        for i in range(2):
            path=data/f'{cls}_{i}.png'
            yy,xx=np.mgrid[:96,:128]
            a=np.stack([(xx*3+c*7)%255,(yy*4+i*11)%255,((xx+yy)*2)%255],axis=-1).astype('uint8')
            Image.fromarray(a).save(path)
            manifest.append(dict(file=path.name,image_path=str(path),part='eval',**{'class':cls}))
    with (data/'samples_manifest.jsonl').open('w') as f:
        for r in manifest:f.write(json.dumps(r)+'\n')
    st={'deficient':classes[:4],'known':classes[4:]}
    write_json(data/'strata_q9b.json',st);write_json(data/'strata_llava16_food101.json',st)
    return manifest


class FakeTokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):return [1]
    def __call__(self,text,add_special_tokens=False):return {'input_ids':[1,1,1]}
    def decode(self,ids,skip_special_tokens=True):return 'c00'


class FakeEngine:
    """CPU tensors exercise the production flow, not a substitute for a real VLM."""
    def __init__(self):self.tokenizer=FakeTokenizer();self.eos={2};self.device='cpu'
    def build(self,image,prompt):return {'reference':False,'prompt':prompt}
    def build_text_only(self,prompt):return {'reference':True,'prompt':prompt}
    def noised_inputs(self,inputs,seed):return {**inputs,'reference':True}
    def _fwd(self,inputs=None,tok=None,pkv=None,**kw):
        import torch
        if inputs is not None:step=0;ref=inputs.get('reference',False)
        else:step,ref=pkv;step+=1
        z=np.zeros(3) if ref else (np.array([1.,2.,-3.]) if step==0 else np.array([0.,-1.,3.]))
        return SimpleNamespace(logits=torch.tensor(z,dtype=torch.float64)[None,None,:],past_key_values=(step,ref))
    def _decode(self,main,ref,alpha):
        a=self._fwd(inputs=main);b=self._fwd(inputs=ref) if ref is not None else None
        toks=[];probs=[]
        for _ in range(12):
            z=a.logits[0,-1].numpy();star=z.copy()
            if b is not None:
                r=b.logits[0,-1].numpy();star=z+alpha*(z-r)
                star=np.where(z>=z.max()+np.log(.1),star,-np.inf)
            p=softmax(star);tok=int(p.argmax());toks.append(tok);probs.append(float(p[tok]))
            if tok==2:break
            a=self._fwd(tok=tok,pkv=a.past_key_values)
            if b is not None:b=self._fwd(tok=tok,pkv=b.past_key_values)
        return dict(text='c00',tokens=toks,step_probs=probs,answer_maxp=probs[0],first_probs=p,
                    first_entropy=0.,first_maxp=probs[0],n_forwards=len(toks),n_prefill_tokens=3,wall_s=0.,layers_selected=[])
    def decode_single(self,inputs,max_new_tokens,method='direct'):return self._decode(inputs,None,0)
    def decode_two_branch(self,inputs,ref,method,max_new_tokens,alpha_override=None,t0_sched=0):
        return self._decode(inputs,ref,1. if alpha_override is None else alpha_override)


def fake_score(text,cls,classes):return 'correct' if text==cls else 'wrong'


def run_persistent(root:Path):
    from stage9_common import initialize
    from stage9_analyze import run as analyze
    from stage9_prepare import prepare
    from stage9_run import run
    from stage9_finalize import finalize
    initialize(root);cache_fixture(root)
    analyze(root,root/'outputs/tables/stage9',('q9b','llava16'),replicates=64)
    image_fixture(root);prepare(root,2,1)
    for model in ('q9b','llava16'):
        run(root,model,FakeEngine(),'Name the object.',fake_score,use_legacy=False)
    finalize(root,replicates=64)
    return root

def visualization_fixture(package:Path):
    """Design-only curves; fixed numbers deliberately do not encode a prediction."""
    import csv, shutil
    output=package/'data_sample';output.mkdir(exist_ok=True)
    shutil.copyfile(package/'verification/fixture_project/outputs/tables/stage9/risk_coverage.csv',output/'synthetic_risk_coverage.csv')
    curves={('low_acc','direct'):[.35,.23,.12],('low_acc','vcd'):[.39,.25,.10],
            ('low_acc','m3id'):[.37,.24,.13],('high_acc','direct'):[.88,.72,.49],
            ('high_acc','vcd'):[.83,.67,.44],('high_acc','m3id'):[.87,.71,.48]}
    rows=[]
    for (st,m),vals in curves.items():
        for c,a in zip(('original','short64','short32'),vals):
            rows.append(dict(model='q9b',stratum=st,condition=c,method=m,n=100,accuracy=a))
    with (output/'synthetic_evidence_accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    kinds=['at_least_one_name_reachable','every_name_excluded_by_support','remaining_score_constraints_incompatible']
    specs=[('q9b','original',[32,40,28]),('q9b','short32',[20,52,28]),
           ('llava16','original',[28,46,26]),('llava16','short32',[18,58,24])]
    for model,c,counts in specs:
        for category,n in zip(kinds,counts):
            for i in range(n):rows.append(dict(model=model,condition=c,file=f'synthetic_{category}_{i}',category=category))
    with (output/'synthetic_name_reachability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(output/'SYNTHETIC_NOTICE.json',dict(scientific_evidence=False,
        risk_source='Seed-912 synthetic per-sample cache, processed by stage9_analyze.py',
        evidence_source='Explicit fixed design-only values in tests/build_fixtures.py',
        reachability_source='Explicit design-only category counts in tests/build_fixtures.py',
        interpretation='Layout and code verification only; no expected or actual VLM findings.'))

if __name__=='__main__':
    root=ROOT/'verification/fixture_project'
    if root.exists():raise SystemExit('Refusing to mix a new fixture run with existing outputs; delete only this fixture directory before rerunning.')
    run_persistent(root);visualization_fixture(ROOT);print(root)
