#!/usr/bin/env python3
"""Single Food-frozen operating points on all951 blind Hallusion visual questions."""
from __future__ import annotations
import argparse,copy,json,math,os,re,socket,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.decoding import DecodeConfig,generate
from kdm.cda import generate_cda
from kdm.pipeline import make_backend,task_id,json_safe
from kdm.io import atomic_json,file_hash,read_jsonl,stable_hash,stable_seed
from kdm.protocol import validate_native_runtime_files
from workflows.general_vqa_direct.generate import admit,now,relative,write_once,process_identity,ids_from_file
SOURCES=('workflows/hallusion_blind/generate.py','src/kdm/decoding.py','src/kdm/cda.py',
'src/kdm/prompts.py','src/kdm/pipeline.py','src/kdm/probability.py','src/kdm/models/hf.py',
'src/kdm/models/backbone.py','src/kdm/models/remote.py','src/kdm/models/sid.py',
'src/kdm/models/internvl_dual.py','src/kdm/models/internvl_preprocessing.py',
'workflows/general_vqa_direct/generate.py')

def guided_prompt(plain,marker,guided):
    if not guided:return plain
    from kdm.prompts import MARKERS
    if marker not in MARKERS:raise ValueError('Unregistered Food expression')
    return plain+f'\nIf you cannot determine the answer reliably, respond with "{marker}".'

def load_plan(args,root=ROOT):
    protocol_path=relative(root,args.protocol)
    protocol=json.loads(protocol_path.read_text())
    condition_path=protocol_path.parent/protocol['conditions']
    frozen_conditions=json.loads(condition_path.read_text())
    if protocol.get('conditions_sha256') and file_hash(condition_path)!=protocol['conditions_sha256']:raise ValueError('Food condition bytes changed')
    entry=protocol['dataset_entry']
    manifest=relative(root,entry['manifest'])
    if file_hash(manifest)!=entry['manifest_sha256']:raise ValueError('Frozen Hall manifest changed')
    samples=list(read_jsonl(manifest))
    if len(samples)!=951 or len({s['id'] for s in samples})!=951:raise ValueError('Whole951 roster required')
    conditions=[c for c in frozen_conditions if c['model']==args.model]
    if not set(args.methods)<={c['method'] for c in conditions}:raise ValueError('Method outside frozen Food panel')
    tasks={}
    for c in conditions:
        for s in samples:
            if s['split']!='eval' or s['dataset']!='hallusionbench':raise ValueError('Blind eval required')
            item={'sample':s,**{f:c[f] for f in ('method','marker','reference_marker','guided','reference_guided','kind','replicate')},
                  'condition_identity':stable_hash(c)}
            key=task_id(args.model,item)
            if key in tasks:raise ValueError('Duplicate task')
            tasks[key]=item
    by_cond={stable_hash(c):c for c in conditions}
    # Eight fixed source-order inputs per method produce timing evidence before the rest.
    selected=[]
    for c in conditions:
        h=stable_hash(c); selected += [k for k,v in tasks.items() if v['condition_identity']==h][:8]
    for c in conditions:
        h=stable_hash(c); selected += [k for k,v in tasks.items() if v['condition_identity']==h][8:]
    selected=[k for k in selected if tasks[k]['method'] in args.methods]
    assignment_sha=None
    if args.sample_ids:
        assignment=relative(root,args.sample_ids); ids=ids_from_file(assignment)
        if not set(ids)<={s['id'] for s in samples}:raise ValueError('Foreign input assignment')
        selected=[k for k in selected if tasks[k]['sample']['id'] in set(ids)]
        assignment_sha=file_hash(assignment)
    spec_path=root/f'configs/runtime/{args.model}.json'
    spec=json.loads(spec_path.read_text())
    dtype='float16' if args.model=='onevision' else 'bfloat16'
    if spec['key']!=args.model or spec['dtype']!=dtype or spec.get('api'):raise ValueError('Frozen native model required')
    validate_native_runtime_files(root,spec,args.model)
    # Identity uses the full model's frozen conditions, allowing independent method shards.
    allconditions=[c for c in frozen_conditions if c['model']==args.model]
    definition={'schema':'kdm_hallusion_blind_eos_v1','model':args.model,'dtype':dtype,
        'runtime_sha256':file_hash(spec_path),'conditions':allconditions,'dataset_entry':entry,
        'source_sha256':{f:file_hash(root/f) for f in SOURCES},'engine':'registered_hf_session',
        'batch_size':1,'prompt_policy':'exact_manifest_plain_plus_registered_guidance_only',
        'termination':'native_eos_only_no_output_cap','selection':'Food_dev_frozen_no_Hall_dev'}
    output=relative(root,args.output);output.relative_to(root/'outputs/hallusion_blind_20261005')
    cfgs={h:DecodeConfig(**{**c['config'],'max_tokens':None}) for h,c in by_cond.items()}
    return {'definition':definition,'identity':stable_hash(definition),'spec':spec,'conditions':by_cond,
        'cfgs':cfgs,'datasets':{'hallusionbench':{'entry':entry,'identity':stable_hash(entry)}},
        'tasks':tasks,'selected':selected,'output':output/args.model,'assignment_sha256':assignment_sha}

def expected_config(plan,item,row=None):
    cfg=plan['cfgs'][item['condition_identity']]
    if cfg.method=='instruction_m3id' and row is not None:
        tokens=row.get('offset_prompt_tokens')
        if not isinstance(tokens,list) or any(type(t)is not int for t in tokens):
            raise ValueError('Missing registered M3ID prompt offset')
        cfg=replace(cfg,m3id_offset=len(tokens))
    return asdict(cfg)

def validate_rows(path,plan,keys,claim_identity,eos):
    rows=list(read_jsonl(path))
    if [r.get('key') for r in rows]!=keys or len(set(keys))!=len(keys):raise ValueError('Chunk key coverage differs')
    for r,key in zip(rows,keys):
        item=plan['tasks'][key];s=item['sample'];plain=s['prompt']
        if (any(r.get(f)!=v for f,v in item.items()) or r.get('identity')!=plan['identity']
            or r.get('claim_identity')!=claim_identity or r.get('model')!=plan['definition']['model']
            or r.get('dataset_identity')!=plan['datasets']['hallusionbench']['identity'] or r.get('status')!='ok' or r.get('config')!=expected_config(plan,item,r)
            or r.get('prompt')!=guided_prompt(plain,item['marker'],item['guided'])
            or r.get('seed')!=stable_seed(s['id'],r['model'],0)):
            raise ValueError('Frozen model/condition/prompt/source binding differs')
        tok=r.get('tokens')
        if (not isinstance(tok,list) or not tok or any(type(t)is not int or t<0 for t in tok)
            or r.get('terminated') is not True or tok[-1] not in eos or any(t in eos for t in tok[:-1])):
            raise ValueError('Only naturally ended full responses may seal as complete')
        if not math.isfinite(r.get('wall_s',float('nan'))) or r['wall_s']<0:raise ValueError('Invalid measured time')
        trace=r.get('trace')
        if r.get('generation_source')=='new_eos_only' and (not isinstance(trace,list) or len(trace)!=len(tok) or [t.get('token') for t in trace]!=tok):
            raise ValueError('Token trace does not match actual generation')
        if r.get('generation_source')=='new_eos_only':
            branches=r.get('branch_inputs',{})
            expected={'main':r['prompt']}
            method=item['method']
            if method in {'vcd','sid','instruction_vcd','instruction_m3id'}:
                expected['reference']=guided_prompt(plain,item['reference_marker'],item['reference_guided'])
            if method.startswith('instruction_'):expected['neutral']=plain
            if method=='cda_visual':expected.update(prior=plain,context=plain,
                null_prior='[N/A]\nGive a concise answer.',null_context='[N/A]\nGive a concise answer.')
            if set(branches)!=set(expected) or any(branches[n].get('prompt')!=v for n,v in expected.items()):
                raise ValueError('Actual branch prompts differ from the registered condition')
            if method in {'vcd','instruction_vcd'} and not branches['reference'].get('noise_tensor_sha256'):
                raise ValueError('Missing actual processed-noise tensor evidence')
        lp=r.get('selected_log_probabilities')
        if r.get('generation_source')=='new_eos_only' and (not isinstance(lp,list) or len(lp)!=len(tok) or not all(math.isfinite(x) for x in lp)):
            raise ValueError('Invalid selected-token probabilities')
    return {'rows':len(rows),'truncated_rows':0,'dataset_counts':{'hallusionbench':len(rows)}}
def completed_keys(plan):
    ledger = plan["output"] / "completed_keys.jsonl"
    if not ledger.exists():
        return set()
    grouped, seen = {}, set()
    for row in read_jsonl(ledger):
        if row["key"] in seen or row["identity"] != plan["identity"]:
            raise ValueError("Duplicate/foreign completed-key ledger entry")
        seen.add(row["key"])
        if row["dataset"] in plan["datasets"]:
            if row["key"] not in plan["tasks"]:
                raise ValueError("Completed key lies outside its frozen manifest")
            grouped.setdefault(row["receipt"], []).append(row)
    complete = set()
    for name, entries in grouped.items():
        receipt_path = relative(plan["output"], name)
        receipt = json.loads(receipt_path.read_text())
        raw = relative(plan["output"], receipt["raw_path"])
        owner_path = relative(plan["output"], receipt["owner_path"])
        owner = json.loads(owner_path.read_text())
        keys = [entry["key"] for entry in entries]
        if (receipt["status"] != "complete" or receipt["identity"] != plan["identity"]
                or receipt["keys"] != keys or file_hash(raw) != receipt["raw_sha256"]
                or file_hash(owner_path) != receipt["owner_sha256"]
                or stable_hash(owner) != receipt["claim_identity"]
                or any(entry["receipt_sha256"] != file_hash(receipt_path) for entry in entries)):
            raise ValueError("Completed ledger receipt/source binding differs")
        checked = validate_rows(raw, plan, keys, receipt["claim_identity"], set(receipt["eos_token_ids"]))
        if checked != receipt["validation"]:
            raise ValueError("Completed chunk no longer passes its validation")
        complete.update(keys)
    return complete

def claim(args, plan, admission):
    import fcntl
    folder = plan["output"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        definition = {"identity": plan["identity"], "definition": plan["definition"]}
        path = folder / "identity.json"
        if path.exists():
            if json.loads(path.read_text()) != definition:
                raise ValueError("Output directory belongs to another model/source identity")
        else:
            write_once(path, definition)
        complete = completed_keys(plan)
        pending = set(plan["selected"]) - complete
        claims = folder / "claims"
        claims.mkdir(exist_ok=True)
        for other in claims.iterdir():
            if not other.is_dir():
                continue
            previous = json.loads((other / "owner.json").read_text())
            released = set()
            if (other / "released.json").exists():
                release = json.loads((other / "released.json").read_text())
                if release["claim_identity"] != stable_hash(previous) or release["reason"] != "stop_after_completed_chunk":
                    raise ValueError("Unbound ownership release")
                released = set(release["keys"])
                if not released <= set(previous["keys"]):
                    raise ValueError("Ownership release contains keys outside its claim")
            if pending & (set(previous["keys"]) - complete - released):
                raise ValueError("A pending key already has an owner; no retry or automatic takeover")
        if not pending:
            return None, [], None
        run = claims / args.claim_id
        run.mkdir(exist_ok=False)
        keys = [key for key in plan["selected"] if key in pending]
        owner = {"claim_id": args.claim_id, "owner": args.owner, "started_utc": now(),
                 **process_identity(), "keys": keys, "identity": plan["identity"],
                 "dataset_entries": {name: row["entry"] for name, row in plan["datasets"].items()},
                 "admission": admission, "batch_size": 1, "chunk_rows": args.chunk_rows,
                 "assignment_path": args.sample_ids, "assignment_sha256": plan["assignment_sha256"],
                 "source_sha256": plan["definition"]["source_sha256"], "argv": sys.argv,
                 "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_once(run / "owner.json", owner)
        return run, keys, owner

def seal(plan, run, owner, number, pending_path, keys, eos):
    import fcntl
    checked = validate_rows(pending_path, plan, keys, stable_hash(owner), eos)
    raw = run / f"chunk_{number:05d}.jsonl"
    if raw.exists():
        raise ValueError("Refusing to replace a completed chunk")
    pending_path.rename(raw)
    raw.chmod(0o444)
    path = run / f"chunk_{number:05d}.complete.json"
    receipt = {"status": "complete", "completed_utc": now(), "identity": plan["identity"],
               "claim_identity": stable_hash(owner), "keys": keys, "eos_token_ids": sorted(eos),
               "raw_path": str(raw.relative_to(plan["output"])), "raw_sha256": file_hash(raw),
               "owner_path": str((run / "owner.json").relative_to(plan["output"])),
               "owner_sha256": file_hash(run / "owner.json"), "validation": checked}
    write_once(path, receipt)
    with (plan["output"] / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = plan["output"] / "completed_keys.jsonl"
        recorded = {row["key"] for row in read_jsonl(ledger)} if ledger.exists() else set()
        if recorded & set(keys):
            raise ValueError("Completed ledger already contains a chunk key")
        with ledger.open("a", encoding="utf-8") as stream:
            for key in keys:
                stream.write(json.dumps({"key": key, "dataset": plan["tasks"][key]["sample"]["dataset"],
                    "identity": plan["identity"], "receipt": str(path.relative_to(plan["output"])),
                    "receipt_sha256": file_hash(path)}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"event": "sealed_chunk", "chunk": number, **checked}), flush=True)

def input_evidence(session,prompt):
    ids=session.inputs['input_ids'].detach().cpu().tolist()
    evidence={'prompt':prompt,'input_ids_sha256':stable_hash(ids),'input_token_count':sum(map(len,ids))}
    pixels=session.inputs.get('pixel_values')
    if pixels is not None:
        def shapes(v):
            if isinstance(v,(list,tuple)):return [shapes(x) for x in v]
            return {'shape':list(v.shape),'dtype':str(v.dtype)}
        evidence['processed_image_tensors']=shapes(pixels)
    return evidence

def run_one(backend,item,cfg,model):
    from PIL import Image
    sample=item['sample']
    if len(sample['image_paths'])!=1:raise ValueError('Hall single-image source required')
    path=relative(ROOT,sample['image_paths'][0])
    if file_hash(path)!=sample['image_sha256'][0]:raise ValueError('Frozen image bytes changed')
    with Image.open(path) as im:image=im.convert('RGB')
    seed=stable_seed(sample['id'],model,0);plain=sample['prompt']
    prompt=guided_prompt(plain,item['marker'],item['guided'])
    refprompt=guided_prompt(plain,item['reference_marker'],item['reference_guided'])
    created=[];evidence={};offset=None
    def session(name,image,prompt,**kw):
        value=backend.session(image,prompt,**kw);created.append(value)
        evidence[name]=input_evidence(value,prompt)
        if kw.get('reference')=='noise':
            import hashlib
            digest=hashlib.sha256()
            def add(v):
                if isinstance(v,(list,tuple)):
                    for x in v:add(x)
                else:
                    digest.update(str((tuple(v.shape),str(v.dtype))).encode())
                    digest.update(v.detach().contiguous().view(backend.torch.uint8).cpu().numpy().tobytes())
            add(value.inputs['pixel_values'])
            evidence[name]['noise_tensor_sha256']=digest.hexdigest()
        return value
    try:
        main=session('main',image,prompt,need_layers=cfg.method in {'dola','deco'})
        reference=neutral=None
        if cfg.method in {'vcd','sid','instruction_vcd','instruction_m3id'}:
            mode='text_only' if 'm3id' in cfg.method else 'sid' if cfg.method=='sid' else 'noise'
            reference=session('reference',image,refprompt,reference=mode,seed=seed)
        if cfg.method.startswith('instruction_'):
            neutral=session('neutral',image,plain)
        if cfg.method=='instruction_m3id':
            offset=list(backend.encode(plain));cfg=replace(cfg,m3id_offset=len(offset))
        if cfg.method=='cda_visual':
            prior=session('prior',image,plain,reference='text_only')
            context=session('context',image,plain)
            # Preserve the registered CDA no-input placeholder convention.
            nullprompt='[N/A]\nGive a concise answer.'
            nullprior=session('null_prior',image,nullprompt,reference='text_only')
            nullcontext=session('null_context',Image.new('RGB',image.size,(127,127,127)),nullprompt)
            result=generate_cda(prior,context,main,nullprior,nullcontext,cfg,backend.eos,backend.decode,seed)
        else:result=generate(main,reference,cfg,backend.eos,backend.decode,seed,neutral)
        if not result['terminated']:raise ValueError('EOS-only run returned without natural EOS')
        return {**result,'prompt':prompt,'reference_prompt':refprompt if reference else None,
            'neutral_prompt':plain if neutral else None,'config':asdict(cfg),'seed':seed,
            'offset_prompt_tokens':offset,'branch_inputs':evidence,
            'noise':{'mode':'processed_visual_tensor_diffusion','timestep':500,'seed':seed}
                if cfg.method in {'vcd','instruction_vcd'} else None}
    finally:
        if cfg.method=='sid' and getattr(backend,'sid_control',None):
            backend.sid_control.close();backend.sid_control=None
        created.clear()

def execute(args,plan):
    actual,admission=admit(args,plan)
    run,keys,owner=claim(args,plan,admission)
    if run is None:return {'status':'already_complete','selected_rows':len(plan['selected'])}
    started=time.perf_counter();active_key=None
    try:
        backend=make_backend(actual,'cuda:0');loaded=time.perf_counter()
        eos=set(backend.eos)
        if not eos:raise ValueError('Missing registered natural termination tokens')
        number=0;chunk_keys=[];timings={}
        for index,key in enumerate(keys):
            active_key=key;item=plan['tasks'][key];method=item['method'];cfg=plan['cfgs'][item['condition_identity']]
            before=time.perf_counter()
            result=run_one(backend,item,cfg,args.model);wall=time.perf_counter()-before
            row={'key':key,**item,'model':args.model,'identity':plan['identity'],
                'dataset_identity':plan['datasets']['hallusionbench']['identity'],
                'claim_identity':stable_hash(owner),'wall_s':wall,'generation_source':'new_eos_only',**result}
            pending=run/f'chunk_{number:05d}.pending.jsonl'
            with pending.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(json_safe(row),ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
            chunk_keys.append(key)
            timing=timings.setdefault(method,[])
            if len(timing)<8:
                timing.append({'key':key,'wall_s':wall,'n_tokens':len(row['tokens'])})
                if len(timing)==8:
                    measurement={'status':'actual_first8','method':method,'model':args.model,
                        'rows':timing,'mean_input_wall_s':sum(x['wall_s'] for x in timing)/8,
                        'mean_output_tokens':sum(x['n_tokens'] for x in timing)/8,
                        'model_load_wall_s':loaded-started,'measured_utc':now(),'max_tokens':None}
                    write_once(run/f'first8_{method}.json',measurement)
                    print(json.dumps({'event':'actual_first8',**measurement}),flush=True)
            nextmethod=plan['tasks'][keys[index+1]]['method'] if index+1<len(keys) else None
            if len(chunk_keys)>=args.chunk_rows or nextmethod!=method:
                seal(plan,run,owner,number,pending,chunk_keys,eos);number+=1;chunk_keys=[]
                if (run/'STOP_AFTER_CHUNK').exists() and index+1<len(keys):
                    write_once(run/'released.json',{'claim_identity':stable_hash(owner),
                        'reason':'stop_after_completed_chunk','released_utc':now(),'keys':keys[index+1:]})
                    return {'status':'stopped_after_completed_chunk','completed_rows':index+1}
        complete=completed_keys(plan)
        if not set(keys)<=complete:raise ValueError('Claim lacks sealed full responses')
        receipt={'status':'complete','completed_rows':len(keys),'completed_utc':now(),
            'identity':plan['identity'],'claim_identity':stable_hash(owner),'wall_s':time.perf_counter()-started}
        write_once(run/'complete.json',receipt);return receipt
    except BaseException as exc:
        write_once(run/'failed.json',{'status':'failed','failed_utc':now(),'key':active_key,
            'claim_identity':stable_hash(owner),'error':type(exc).__name__+': '+str(exc),
            'traceback':traceback.format_exc(),'cda_diagnostics':getattr(exc,'cda_diagnostics',None),
            'automatic_retry':False,'automatic_parameter_change':False})
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true');mode.add_argument('--verify',action='store_true')
    mode.add_argument('--check-plan',action='store_true')
    p.add_argument('--model',required=True);p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--protocol',default='outputs/hallusion_blind_20261005/registration/protocol.json')
    p.add_argument('--output',default='outputs/hallusion_blind_20261005/runs')
    p.add_argument('--registry',default='workflows/general_vqa_direct/host_registry.json')
    p.add_argument('--cards',required=True);p.add_argument('--claim-id',required=True);p.add_argument('--owner',default='/root')
    p.add_argument('--sample-ids');p.add_argument('--chunk-rows',type=int,choices=(1,8,16),default=8)
    p.add_argument('--k100-intern',action='store_true')
    args=p.parse_args();plan=load_plan(args)
    if args.execute:result=execute(args,plan)
    elif args.verify:
        complete=completed_keys(plan);expected=set(plan['selected'])
        result={'status':'complete' if expected<=complete else 'incomplete',
            'expected_rows':len(expected),'completed_rows':len(expected&complete)}
    else:result={'status':'CPU_plan_checked','model_identity':plan['identity'],'selected_rows':len(plan['selected']),
        'max_tokens':None,'methods':args.methods,'gpu_initialized':False}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2),flush=True)
    return 2 if args.verify and result['status']!='complete' else 0
if __name__=='__main__':raise SystemExit(main())
