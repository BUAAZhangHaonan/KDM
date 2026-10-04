"""Original SID on the frozen VizWiz512 roster; no algorithm adaptation."""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(Path(__file__).parent)]
from kdm.io import atomic_json, file_hash, read_jsonl, stable_seed, stable_hash, within
from kdm.execution import resolve_image_path
from kdm.frozen import frozen_path
from kdm.pipeline import make_backend, sessions, task_id
from kdm.decoding import DecodeConfig, generate
from author_baseline_pilot import load_inputs
from native_baselines import admission as core_admission
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.native_audit import input_description

MODELS = ('qwen25vl','llava16_mistral','gemma3_4b','internvl35_8b','onevision','phi35','qwen3vl')
CORE = ('qwen25vl','llava16_mistral','gemma3_4b')
REVISION = 'sid_author_Viz512_20261004'
REVIEWED_SID_SHA = 'e1b8457e6dc0f470c62e333a161fca1b65370dd2383b6c9633c6d32b79fa1299'
K100_GATE = 'outputs/supplemental/remaining4/intern_k100_single_admission_launch2_20261003_1355/admission/k100_intern_single_actual_duplicate8_launch2_20261003_1355/condition_gate.json'
K100_GATE_SHA = '1ebd535c708a202be45f951bc9057e903b34263c77c126e271a2eb758a754336'


def now(): return datetime.now(timezone.utc).isoformat()


def task(sample):
    return dict(sample=sample, method='sid', kind='native_unguided_author_core', marker='NONE',
                reference_marker='NONE', guided=False, reference_guided=False, replicate=0,
                implementation_revision=REVISION)


def admit(args, spec):
    cards = args.physical_gpus.split(',')
    if args.model in CORE and not args.supplemental_location:
        return core_admission(spec, args.model, cards, args.registry)
    from workflows.supplemental.remaining11 import execution
    execution.REGISTRY = args.registry
    os.environ['KDM_SUPPLEMENTAL_DATASET'] = 'vizwiz'
    os.environ['KDM_SUPPLEMENTAL_METHOD'] = 'sid'
    if args.k100_intern:
        from workflows.supplemental.remaining4.k100_intern_registered_matrix import single_spec
        path = ROOT / K100_GATE
        gate = json.loads(path.read_text())
        assert args.model == 'internvl35_8b' and cards == ['0']
        assert file_hash(path) == K100_GATE_SHA and gate['passed'] and gate['production_allowed']
        assert gate['completed'] == 8
        assert file_hash(ROOT/'workflows/supplemental/remaining4/internvl_k100_single.py') == gate['factory_sha256']
        assert file_hash(ROOT/gate['operator_audit_path']) == gate['operator_audit_sha256']
        value = execution.validate_supplemental_runtime(ROOT, single_spec(spec), args.model,
                    cards, 'independent', args.run_id, args.owner)
        value['execution']['reused_hardware_gate'] = {'path':K100_GATE,'sha256':K100_GATE_SHA,
            'scope':'existing single-map checkpoint/software/load-capacity admission only',
            'SID_operator_admitted_by_prior_gate':False, 'new_SID_eight_input_gate_required':True}
    else:
        value = execution.validate_supplemental_runtime(ROOT, spec, args.model, cards,
                                                       'formal', args.run_id, args.owner)
    return value['runtime_spec'], value['execution']


def run(args):
    # Reuse frozen checkpoint/forward proof checks; Food representatives returned
    # by this helper are intentionally not used as this experiment's inputs.
    _, spec, original_proof = load_inputs(args.model, 'vcd')
    # The legacy SID receipts bind the entire older sid.py file. Gemma support
    # added at 523d35fb changed that file without changing the six other model
    # branches. Keep the old receipt as history, not as current admission. The
    # present source is pinned here and admitted on eight real Viz inputs below.
    assert file_hash(ROOT/'src/kdm/models/sid.py') == REVIEWED_SID_SHA
    legacy_sid = spec.get('mechanism_validation',{}).get('sid_reference')
    original_proof['current_sid_source_sha256'] = REVIEWED_SID_SHA
    original_proof['SID_current_admission'] = 'this run real eight-input rank100/mask/session gate'
    original_proof['native_interface_proof_method'] = 'vcd'
    if legacy_sid:
        legacy_path = frozen_path(ROOT,legacy_sid['record'])
        original_proof['legacy_sid_reference'] = {'path':legacy_sid['record'],
            'resolved_path':str(legacy_path),'sha256':file_hash(legacy_path),'current_source_admission':False}
    roster_path, all_samples = roster('viz512')
    pilot_ids = [sample['id'] for sample in all_samples[:8]]
    cfg = DecodeConfig(method='sid', alpha=.5)
    assert cfg.max_tokens==32 and cfg.temperature==0 and cfg.top_p==1
    signature = dict(model=args.model, dataset='vizwiz', split='eval', revision=REVISION,
        config=asdict(cfg), roster_sha256=file_hash(roster_path),
        sid_sha256=file_hash(ROOT/'src/kdm/models/sid.py'),
        decoder_sha256=file_hash(ROOT/'src/kdm/decoding.py'),
        frozen_dtype=spec['kwargs'].get('dtype',spec['dtype']))
    samples = all_samples[:8] if args.mode=='pilot' else [sample for sample in all_samples
               if sample['id'] not in pilot_ids and int(stable_hash(sample['id'])[:8],16)%args.parts==args.part]
    if args.sample_list:
        listing = json.loads(within(ROOT,args.sample_list).read_text())
        ids = listing['sample_ids']
        assert listing['model']==args.model and len(ids)==len(set(ids))
        assert not set(ids)&set(pilot_ids) and set(ids)<={sample['id'] for sample in samples}
        samples = [sample for sample in samples if sample['id'] in set(ids)]
    if args.mode=='plan':
        print(json.dumps(dict(signature=signature,total=512,pilot=8,remaining=len(samples),
                              GPU_initialized=False,original_proof=original_proof)),flush=True)
        return
    out = within(ROOT,args.output)
    if args.mode=='full':
        prior = within(ROOT,args.admission)
        gate = json.loads((prior/'complete.json').read_text())
        identity = json.loads((prior/'identity.json').read_text())
        assert gate['passed'] and gate['completed']==8 and gate['pilot']
        assert identity['signature']==signature
        assert identity['sample_ids']==pilot_ids
        assert file_hash(prior/'new_predictions.jsonl')==gate['raw_sha256']
        assert file_hash(prior/'attention_audit.jsonl')==gate['attention_audit_sha256']
    actual, admitted = admit(args,spec)
    assert actual['kwargs'].get('dtype',actual['dtype'])==signature['frozen_dtype']
    out.mkdir(parents=True,exist_ok=False)
    identity = dict(signature=signature,model=args.model,method='sid',config=asdict(cfg),
        sample_ids=[sample['id'] for sample in samples],pilot=args.mode=='pilot',
        frozen_forward_proof=original_proof,runtime_spec=actual,runtime_admission=admitted,
        official_sid_commit='127dd412fa6b61ab1c9babf6979ec4da98002438',
        fixed_operator=dict(aggregation_block_1based=2,rank=100,alpha=.5,greedy_support='full_vocab'),
        prior_six_model_Food_SID_identity='legacy alpha1/beta-supported full runs remain separate',
        runner_sha256=file_hash(Path(__file__)),partition=dict(part=args.part,parts=args.parts,
        excluded_pilot_ids=[] if args.mode=='pilot' else pilot_ids))
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'claim.json',dict(pid=os.getpid(),starttick=Path('/proc/self/stat').read_text().split()[21],
        owner=args.owner,run_id=args.run_id,host=socket.gethostname(),started_utc=now(),
        physical_gpus=args.physical_gpus,identity_sha256=file_hash(out/'identity.json')))
    started=time.perf_counter(); rows=[]
    try:
        backend=make_backend(actual,'cuda:0');load_s=time.perf_counter()-started
        torch=backend.torch
        from PIL import Image
        for index,sample in enumerate(samples):
            one=time.perf_counter()
            with Image.open(resolve_image_path(sample['image_path'],ROOT)) as image_file:
                image=image_file.convert('RGB')
            seed=stable_seed(sample['id'],args.model,0); item=task(sample)
            main,reference,neutral,prompt,reference_prompt=sessions(backend,image,item,cfg,seed)
            control=reference.control;checks=[]
            if prompt!=reference_prompt:raise ValueError('SID branch prompts differ')
            prompt_ids=reference.inputs['input_ids'].detach().cpu().tolist()
            if args.mode=='pilot':
                if input_description(main.inputs)!=input_description(reference.inputs):
                    raise ValueError('SID clean and reference processor inputs differ')
                def audit(event):
                    attention=event['attention'];mask=event['mask'];original=event['original_mask']
                    if not bool(torch.isfinite(attention).all()):raise ValueError('Nonfinite actual attention')
                    oracle=attention.mean(dim=0)[-1,event['start']:event['start']+event['length']].topk(100,largest=False).indices.sort().values
                    if not torch.equal(oracle,event['selected']):raise ValueError('Original rank100 differs')
                    blocked=torch.zeros(mask.shape[-1],device=mask.device,dtype=torch.bool)
                    blocked[event['start']:event['start']+event['length']]=True
                    blocked[event['start']+oracle.to(mask.device)]=False
                    if not bool(torch.isneginf(mask[...,blocked]).all()):raise ValueError('Dropped visual keys leaked')
                    if original is not None:
                        if not torch.equal(mask[...,~blocked],original[...,~blocked]):raise ValueError('Native mask changed outside dropped visuals')
                    else:
                        query=torch.arange(event['query_length'],device=mask.device)[:,None]+event['kv_length']-event['query_length']
                        keys=torch.arange(event['kv_length'],device=mask.device)[None,:]
                        expected=torch.zeros_like(mask).masked_fill((keys>query)[None,None,:,:],float('-inf'))
                        if not torch.equal(mask[...,~blocked],expected[...,~blocked]):raise ValueError('Original causal mask changed')
                    checks.append(dict(layer=event['layer'],query_length=event['query_length'],
                        kv_length=event['kv_length'],visual_start=event['start'],visual_length=event['length'],
                        selected=oracle.detach().cpu().tolist(),rank100_equal=True,native_mask_retained=True,
                        original_mask_present=original is not None,attention_finite=True))
                control.audit=audit
            before=main.next(()).logits.copy() if args.mode=='pilot' and index==0 else None
            result=generate(main,reference,cfg,backend.eos,backend.decode,seed,neutral)
            if result['status']!='ok' or not 0<len(result['tokens'])<=32:
                raise ValueError('Invalid actual generation')
            if not all(math.isfinite(value) for value in result['selected_log_probabilities']):
                raise ValueError('Nonfinite selected probabilities')
            assert len(result['selected_log_probabilities'])==len(result['tokens'])
            assert result['terminated']==(result['tokens'][-1] in backend.eos)
            if args.mode=='pilot':
                if len(checks)!=(len(control.layers)-2)*len(result['tokens']):
                    raise ValueError('Some generated token/layer lacks actual SID evidence')
                if getattr(backend,'_sid_active_control',None):raise ValueError('SID control leaked')
                if before is not None and not np.array_equal(backend.session(image,prompt).next(()).logits,before):
                    raise ValueError('Clean branch changed after SID')
            row=dict(**item,model=args.model,key=task_id(args.model,item),config=asdict(cfg),seed=seed,
                prompt=prompt,reference_prompt=reference_prompt,neutral_prompt=None,**result,generation_wall_s=time.perf_counter()-one,
                prompt_token_ids=prompt_ids,visual_start=control.start,visual_length=control.length,
                score_status='pending_exact_QA_or_semantic')
            with (out/'new_predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if args.mode=='pilot':
                with (out/'attention_audit.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(sample_id=sample['id'],steps=len(result['tokens']),checks=checks))+'\n')
            rows.append(row)
            if getattr(backend,'sid_control',None):backend.sid_control.close();backend.sid_control=None
            del main,reference,neutral
            mean=sum(row['generation_wall_s'] for row in rows)/len(rows)
            progress=dict(completed=len(rows),expected=len(samples),remaining=len(samples)-len(rows),
                mean_sample_s=mean,eta_s=mean*(len(samples)-len(rows)),estimated_512_generation_s=mean*512,
                updated_utc=now(),actual_gpu_generation=True)
            atomic_json(out/'progress.json',progress)
            if args.mode=='pilot' or len(rows)%16==0 or len(rows)==len(samples):print(json.dumps(progress),flush=True)
        assert len(rows)==len({row['key'] for row in rows})==len(samples)
        receipt=dict(passed=True,model=args.model,dataset='vizwiz',completed=len(rows),expected=len(samples),
            raw_sha256=file_hash(out/'new_predictions.jsonl'),unique_keys=len(rows),
            mean_sample_s=sum(row['generation_wall_s'] for row in rows)/len(rows),
            elapsed_s=time.perf_counter()-started,load_s=load_s,completed_utc=now(),
            pilot=args.mode=='pilot',score_status='pending',full_eval_source_denominator=512,
            official_unanswerable_n=166,official_answerable_n=346,algorithm_changed=False)
        if args.mode=='pilot':receipt['attention_audit_sha256']=file_hash(out/'attention_audit.jsonl')
        atomic_json(out/'complete.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=type(exc).__name__+': '+str(exc),completed=len(rows),
            observed_utc=now(),elapsed_s=time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--mode',choices=('plan','pilot','full'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--physical-gpus',required=True)
    parser.add_argument('--registry',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--owner',default='/root/sid_minicpm_full')
    parser.add_argument('--admission');parser.add_argument('--sample-list')
    parser.add_argument('--parts',type=int,default=1);parser.add_argument('--part',type=int,default=0)
    parser.add_argument('--k100-intern',action='store_true')
    parser.add_argument('--supplemental-location',action='store_true',
                        help='Use existing exclusive supplemental host admission for an unchanged runtime path move')
    args=parser.parse_args()
    if args.mode=='full' and not args.admission:parser.error('Full run requires its actual eight-input admission')
    if not 0<=args.part<args.parts:parser.error('Invalid exclusive partition')
    run(args)
