"""Frozen task expansion, complete response ledgers, and behavioral probes."""
from __future__ import annotations
from dataclasses import asdict,replace
import importlib,json,time,math
from pathlib import Path
from PIL import Image
from kdm.execution import resolve_image_path
import numpy as np
from .io import Ledger,read_jsonl,stable_hash,stable_seed,file_hash,atomic_json
from .prompts import task_prompt,MARKERS,closed_prompt
from .decoding import DecodeConfig,generate,replay
from .probability import complete_response_shift,log_normalize
from .scoring import lexical_label,label_response,food_correct,vqa_score,normalize
ROOT=Path(__file__).resolve().parents[2]


def make_backend(spec,device):
    factory=spec.get('factory')
    if not factory: raise ValueError('Model factory is not configured')
    module,name=factory.split(':');cls=getattr(importlib.import_module(module),name)
    return cls(**spec.get('kwargs',{}),device=device) if not spec.get('api') else cls(**spec.get('kwargs',{}))


def census_tasks(samples,markers=('UNKNOWN',)):
    for sample in samples:
        for marker in markers:
            yield {'sample':sample,'method':'direct','marker':marker,'reference_marker':marker,
                   'guided':True,'reference_guided':True,'replicate':0,'kind':'census'}
        yield {'sample':sample,'method':'direct','marker':'UNKNOWN','reference_marker':'UNKNOWN',
               'guided':False,'reference_guided':False,'replicate':0,'kind':'unguided'}


def experiment_tasks(samples,methods=('vcd','m3id','dola','deco'),markers=MARKERS):
    for sample in samples:
        if sample['split']!='eval':continue
        for marker in markers:
            yield {'sample':sample,'method':'direct','marker':marker,'reference_marker':marker,
                   'guided':True,'reference_guided':True,'replicate':0,'kind':'main'}
            for method in methods:
                refs=markers if method in {'vcd','m3id','sid'} else (marker,)
                for refmarker in refs:
                    yield {'sample':sample,'method':method,'marker':marker,'reference_marker':refmarker,
                           'guided':True,'reference_guided':True,'replicate':0,'kind':'main'}
                if method in {'vcd','m3id'}:
                    yield {'sample':sample,'method':method,'marker':marker,'reference_marker':marker,
                           'guided':True,'reference_guided':False,'replicate':0,'kind':'reference_instruction_removed'}
            for method in ('instruction_vcd','instruction_m3id','cda_visual'):
                yield {'sample':sample,'method':method,'marker':marker,'reference_marker':marker,
                       'guided':True,'reference_guided':False,'replicate':0,'kind':'instruction_preserving'}


def task_id(model,task):
    return stable_hash({'model':model,'task':{k:v for k,v in task.items() if k!='sample'},
                        'sample':task['sample']['id']})


def sessions(backend,image,task,cfg,seed):
    question=task['sample']['question'];method=cfg.method
    main_prompt=task_prompt(question,task['marker'],task['guided'],task.get('attempt',False))
    ref_prompt=task_prompt(question,task['reference_marker'],task['reference_guided'])
    main=backend.session(image,main_prompt,need_layers=method in {'dola','deco'})
    reference=None;neutral=None
    if method in {'vcd','m3id','sid','instruction_vcd','instruction_m3id'}:
        mode='text_only' if 'm3id' in method else ('sid' if method=='sid' else 'noise')
        reference=backend.session(image,ref_prompt,reference=mode,seed=seed)
    if method.startswith('instruction_'):
        neutral=backend.session(image,task_prompt(question,guided=False))
    return main,reference,neutral,main_prompt,ref_prompt


def run_tasks(backend,model,tasks,out,identity,cfg=DecodeConfig(),shard=0,n_shards=1):
    if n_shards<1 or not 0<=shard<n_shards:raise ValueError('Invalid shard')
    ledger=Ledger(out,{**identity,'model':model,'shard':shard,'n_shards':n_shards,'base_config':asdict(cfg)})
    for task in tasks:
        sid=task['sample']['id']
        if int(stable_hash(sid)[:8],16)%n_shards!=shard:continue
        key=task_id(model,task)
        if key in ledger.keys:continue
        seed=stable_seed(sid,model,task['replicate'])
        current=replace(cfg,method=task['method']);offset_prompt_tokens=None
        started=time.perf_counter()
        try:
            with Image.open(resolve_image_path(task['sample']['image_path'], ROOT)) as im:image=im.convert('RGB')
            prompt=task_prompt(task['sample']['question'],task['marker'],task['guided'],task.get('attempt',False))
            if getattr(backend,'generate_text',None):
                if current.method!='direct':raise ValueError('API runner supports direct census only')
                result=backend.generate_text(image,prompt,seed=seed,temperature=current.temperature,
                                             max_tokens=current.max_tokens)
            else:
                main,reference,neutral,prompt,rprompt=sessions(backend,image,task,current,seed)
                if current.method=='m3id':
                    offset_prompt_tokens=list(backend.encode(prompt))
                    current=replace(current,m3id_offset=len(offset_prompt_tokens))
                if current.method=='instruction_m3id':
                    offset_prompt_tokens=list(backend.encode(task_prompt(task['sample']['question'],guided=False)))
                    current=replace(current,m3id_offset=len(offset_prompt_tokens))
                if current.method=='cda_visual':
                    from .cda import generate_cda
                    question=task['sample']['question'];plain=task_prompt(question,guided=False)
                    prior=backend.session(image,plain,reference='text_only')
                    context=backend.session(image,plain)
                    null_prompt=task_prompt('[N/A]',guided=False)
                    null_prior=backend.session(image,null_prompt,reference='text_only')
                    null_context=backend.session(Image.new('RGB',image.size,(127,127,127)),null_prompt)
                    result=generate_cda(prior,context,main,null_prior,null_context,current,
                                        backend.eos,backend.decode,seed)
                    del prior,context,null_prior,null_context
                else:
                    result=generate(main,reference,current,backend.eos,backend.decode,seed,neutral)
                if current.method=='sid' and getattr(backend,'sid_control',None):
                    backend.sid_control.close();backend.sid_control=None
                del main,reference,neutral
            ledger.add(key,{'key':key,**{k:v for k,v in task.items() if k!='sample'},'sample':task['sample'],
                        'model':model,'prompt':prompt,
                        'reference_prompt':task_prompt(task['sample']['question'],task['reference_marker'],task['reference_guided']) if task['method']!='direct' else None,
                        'neutral_prompt':task_prompt(task['sample']['question'],guided=False) if task['method'].startswith('instruction_') else None,
                        'config':asdict(current),'seed':seed,
                        **({'offset_prompt_tokens':offset_prompt_tokens} if offset_prompt_tokens is not None else {}),
                        'wall_s':time.perf_counter()-started,**result})
        except Exception as e:
            error=Path(str(out)+'.errors.jsonl')
            failure={'key':key,'error':type(e).__name__+': '+str(e)}
            if hasattr(e,'cda_diagnostics'):failure['cda_diagnostics']=e.cda_diagnostics
            with error.open('a') as f:f.write(json.dumps(failure,allow_nan=False)+'\n')
            raise
    return len(ledger.keys)


def probe_tasks(samples,repeats=10):
    for s in samples:
        if s['split']!='eval':continue
        for repeat in range(repeats):
            yield {'sample':s,'method':'direct','marker':'UNKNOWN','reference_marker':'UNKNOWN',
                   'guided':False,'reference_guided':False,'attempt':True,
                   'replicate':repeat,'kind':'independent_attempt'}


def closed_rank(backend,image,question,names,gold):
    if gold not in names or len(names)!=len(set(names)): raise ValueError('Invalid candidate labels')
    prompt=closed_prompt(question,names);scores=[]
    session=backend.session(image,prompt)
    for label in names:
        tokens=backend.encode(label.replace('_',' '))
        if not tokens:raise ValueError('Empty label tokenization')
        logps=[]
        for t,tok in enumerate(tokens):logps.append(float(log_normalize(session.next(tuple(tokens[:t])).logits)[tok]))
        scores.append({'label':label,'sum_logp':sum(logps),'mean_logp':float(np.mean(logps)),'n_tokens':len(tokens)})
    target=next(s for s in scores if s['label']==gold)
    rank=1+sum(s['mean_logp']>target['mean_logp'] for s in scores)
    return {'candidate_scores':scores,'gold_rank':rank,'ranking_rule':'mean_log_probability',
            'prompt':prompt,'target':gold}


def select_models(census_paths,annotations,expected_ids):
    """Use complete guided direct records, aggregating authorized source shards."""
    report=[];by_model={};sources={}
    for path in census_paths:
        records=list(read_jsonl(path))
        models={r['model'] for r in records}
        if len(models)!=1:raise ValueError('Census file must contain exactly one model')
        model=next(iter(models))
        by_model.setdefault(model,[]).extend(r for r in records if r['kind']=='census' and r['marker']=='UNKNOWN')
        sources.setdefault(model,[]).append(str(path))
    for model,rows in by_model.items():
        ids=[r['sample']['id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('Incomplete or duplicated model census')
        for dataset in sorted({r['sample']['dataset'] for r in rows}):
            subset=[r for r in rows if r['sample']['dataset']==dataset]
            n_abs=sum(label_response(r['text'],annotations,r['key'])=='abstain' for r in subset)
            report.append({'model':model,'dataset':dataset,'n':len(subset),'n_abstain':n_abs,
                           'selected':n_abs>0,'selection_sources':sources[model]})
    return report


def finite_response_audit(backend,image,question,main_marker,reference_marker,observed_answers,cfg,seed=0):
    """Score a fixed list of exact complete responses, including every EOS path.
    Observed answers are supplied from a fixed donor pool established before
    reference-marker comparisons. No result-dependent candidate removal.
    """
    candidates=[];seen={}
    entries=[{'kind':'abstention','text':text} for text in MARKERS]
    entries += [({'kind':'answer','text':item} if isinstance(item,str) else item) for item in observed_answers]
    for entry in entries:
        kind=entry['kind'];text=entry['text']
        if kind not in {'abstention','answer'}:raise ValueError('Unknown response group')
        base=list(entry['tokens']) if 'tokens' in entry else backend.encode(text)
        if not base:raise ValueError('Empty candidate sequence')
        if any(tok in backend.eos for tok in base[:-1]):raise ValueError('EOS appears before response end')
        if not backend.eos:raise ValueError('Registered termination tokens are required')
        if base[-1] in backend.eos:base=base[:-1]
        if not base:raise ValueError('Empty complete response')
        sequences=[tuple(base+[eos]) for eos in sorted(backend.eos)]
        for toks in sequences:
            if toks in seen:
                if seen[toks]!=kind:raise ValueError('Same token sequence assigned to conflicting response groups')
                continue
            seen[toks]=kind;candidates.append((kind,text,toks))
    if not any(c[0]=='answer' for c in candidates):raise ValueError('Answer candidates required')
    sample={'question':question};task={'sample':sample,'method':cfg.method,'marker':main_marker,
            'reference_marker':reference_marker,'guided':True,'reference_guided':not cfg.method.startswith('instruction_')}
    main,ref,neutral,main_prompt,_=sessions(backend,image,task,cfg,seed=seed)
    if cfg.method=='m3id':cfg=replace(cfg,m3id_offset=len(backend.encode(main_prompt)))
    if cfg.method=='instruction_m3id':cfg=replace(cfg,m3id_offset=len(backend.encode(task_prompt(question,guided=False))))
    result=[]
    for kind,text,tokens in candidates:
        steps=replay(main,ref,cfg,tokens,neutral_main=neutral)
        result.append({'kind':kind,'text':text,'tokens':list(tokens),
                       'base_logp':sum(s['base_logp'] for s in steps),
                       'modified_logp':sum(s['modified_logp'] for s in steps),
                       'steps':steps})
    p=np.array([r['base_logp'] for r in result]);m=np.array([r['modified_logp'] for r in result]);a=np.array([r['kind']=='abstention' for r in result])
    if not np.isfinite(m).any():raise ValueError('Declared response set has no method support')
    closure=complete_response_shift(p,m,a)
    if cfg.method=='sid' and getattr(backend,'sid_control',None):
        backend.sid_control.close();backend.sid_control=None
    return {'responses':result,'finite_response_identity':closure,
            'measurement_scope':'conditional_on_listed_complete_token_sequences'}


def json_safe(value):
    """Represent zero probability as explicit null log-probability, never NaN JSON."""
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return {'nonfinite':str(float(value))}
    if isinstance(value,np.generic):return value.item()
    return value
