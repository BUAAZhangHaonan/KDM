"""VizWiz continuation: original accepted generator; new explicit dataset/key scheduling."""
from __future__ import annotations
import argparse,fcntl,importlib.metadata as metadata,json,os,socket,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows/independent_v5')]
from kdm.io import atomic_json,file_hash,stable_hash,stable_seed,read_jsonl,Ledger
from kdm.pipeline import task_id
BASE=ROOT/'outputs/paper_core_20261002_dev_viz/closeout_20261004/viz'
GPU_UUIDS={'6403':{'0':'GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402','1':'GPU-5c45e961-7442-eb90-9b8c-295a1cf14995'},'RTX_Pro_6000':{'0':'GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea'}}

def evidence(model):
    recovery=json.loads((BASE/'source_recovery.json').read_text())
    for item in recovery['source_files']:
        assert file_hash(ROOT/item['original_path'])==item['sha256'],item['original_path']
    matches=[x for x in recovery['historical_evidence'] if x['member'].endswith('/'+model+'/identity.json')]
    assert len(matches)==1
    item=matches[0]; p=ROOT/item['path']; assert file_hash(p)==item['sha256']
    old=json.loads(p.read_text()); definition=old['definition']
    assert stable_hash(definition)==old['identity']
    assert definition['model']==model and definition['repeats']==10 and definition['dtype']=='bfloat16'
    assert definition['independent_verification']['separate_engine_cohort'] is True
    versions={p:metadata.version(p) for p in ('vllm','torch','transformers')}
    assert versions==definition['independent_verification']['versions'],versions
    return definition,{'path':str(p),'sha256':item['sha256'],'accepted_historical_identity':old['identity'],'versions':versions,'native_numerical_equivalence_claimed':False}

def checkpoint(model,path):
    spec_path=ROOT/'configs/runtime'/f'{model}.json'; spec=json.loads(spec_path.read_text()); path=Path(path)
    assert file_hash(path/'config.json')==spec['model_config_sha256']
    for name,digest in spec['processor']['files'].items():assert file_hash(path/name)==digest,name
    for w in spec['weights']:
        assert (path/w['filename']).stat().st_size==w['size_bytes'],w['filename']
        hub=path/'.cache/huggingface/download'/(w['filename']+'.metadata')
        if hub.exists():assert hub.read_text().splitlines()[:2]==[w['hub_revision'],w['hub_recorded_sha256']]
    return {'spec_sha256':file_hash(spec_path),'checkpoint':str(path),'model_config_sha256':spec['model_config_sha256'],'processor':spec['processor'],'weights':spec['weights']}

def inputs(model,pieces,base):
    samples={s['id']:s for s in read_jsonl(BASE/'inputs/viz_eval3501.jsonl')}
    registered={s['id']:s for s in read_jsonl(ROOT/'data/current/all.jsonl') if s['dataset']=='vizwiz' and s['split']=='eval'}
    assert len(samples)==3501 and samples==registered
    keys={};groups={}
    for piece in pieces:
        path=BASE/'inputs'/model/(piece+'_keys.jsonl')
        for row in read_jsonl(path):
            sid=row['sample_id']; rep=row['replicate']; key=task_id(model,base.task(samples[sid],rep))
            assert row['model']==model and row['key']==key and row['seed']==stable_seed(sid,model,rep)
            assert key not in keys
            keys[key]=row;groups.setdefault(sid,[]).append(key)
    assert all(len(v)==10 and {keys[k]['replicate'] for k in v}==set(range(10)) for v in groups.values())
    return samples,keys,groups

def resource(host,gpu,out):
    import subprocess
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==GPU_UUIDS[host][gpu]
    lock=ROOT/'outputs/locks'/f'gpu_{gpu}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o644);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if fd!=20:os.dup2(fd,20);os.close(fd)
    tick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
    proof={'host':host,'hostname':socket.gethostname(),'gpu':gpu,'uuid':uuid,'pid':os.getpid(),'starttick':tick,'python':sys.executable,'lock':str(lock),'created_unix':time.time()}
    atomic_json(out/'claim.json',proof)
    return proof

def claim_keys(model,keys,res,out):
    """Serialize key admission, checking completed and live shards before load."""
    directory=BASE/'generation'/model; path=directory/'key_admission.lock'
    with open(path,'a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for other in directory.glob('*/independent.jsonl'):
            existing={r['key'] for r in read_jsonl(other)}
            assert not existing.intersection(keys),'Already generated keys: '+str(other)
        for other in directory.glob('*/key_claim.json'):
            old=json.loads(other.read_text())
            assert not set(old['keys']).intersection(keys),'Keys already claimed: '+str(other)
        atomic_json(out/'key_claim.json',{'keys':sorted(keys),'pid':res['pid'],'starttick':res['starttick'],'host':res['host'],'gpu':res['gpu'],'created_unix':time.time()})

def run(a):
    import independent as base
    old,history=evidence(a.model)
    assert file_hash(ROOT/'data/current/all.jsonl')==old['manifest_sha256']
    migration=(a.host=='RTX_Pro_6000' and a.model=='minicpm26')
    assert old['execution']['host']==a.host or migration,(old['execution']['host'],a.host)
    path=str(ROOT/'cache/models/MiniCPM-V-2_6_core20260930') if migration else old['checkpoint_path'];base.closed.PATHS[a.model]=path
    cp=checkpoint(a.model,path);samples,keys,groups=inputs(a.model,a.pieces,base)
    if a.plan:
        print(json.dumps({'model':a.model,'images':len(groups),'answers':len(keys),'history':history,'checkpoint':cp['checkpoint'],'generation_started':False}));return
    out=BASE/'generation'/a.model/a.run;out.mkdir(parents=True,exist_ok=False);bridge=None;sc=None
    try:
        res=resource(a.host,a.gpu,out)
        claim_keys(a.model,keys,res,out)
        if a.host=='RTX_Pro_6000':
            sys.path.insert(0,str(ROOT/'workflows/independent_k100_v1'))
            import admission
            current=admission.register('llava16_mistral' if migration else a.model)
            base.resolve_image_path=admission.image_resolver;base.closed.resolve_image_path=admission.image_resolver
            res['original_k100_host_runtime_admission']=current
            if migration:
                res['model_admission']={'model':a.model,'checkpoint':cp,'source_host':old['execution']['host'],'target_host':a.host,'new_host_input_pilot_images':8,'numerical_equivalence_claimed':False}
        if a.model=='gemma3_4b':
            import gemma_native_pixels_v1 as pixels
            import gemma_range_audit_v1 as ranges
            import gemma_native_pixels_entry_v4 as entry
            ranges.install(base);bridge=pixels.install();res['gemma_native_pixels']=entry.binding_for(bridge)
            assert res['gemma_native_pixels']==old['execution']['native_pixels_bridge']
        eos=old['eos_token_ids'];assert eos==old['independent_verification']['eos_token_ids']
        identity={'schema':'kdm_viz_independent_continuation_v1','model':a.model,'dataset':'vizwiz','split':'eval','repeats':10,
          'historical_evidence':history,'checkpoint':cp,'source_recovery_sha256':file_hash(BASE/'source_recovery.json'),
          'runner_sha256':file_hash(__file__),'manifest_sha256':file_hash(ROOT/'data/current/all.jsonl'),
          'pieces':a.pieces,'keys_sha256':stable_hash(sorted(keys)),'eos_token_ids':eos,'dtype':'bfloat16',
          'sampling_example':base.parameters(a.model,next(iter(groups)),0,eos),'host':a.host,
          'sampling_backend':'vllm_separate_cohort_not_numpy_draw_equivalence'}
        ledger=Ledger(out/'independent.jsonl',identity);atomic_json(out/'identity.json',{'identity':ledger.identity,'definition':identity,'resource':res})
        sc=base.Generator(a.model)
        if a.model=='gemma3_4b':sc.audit_path=out/'gemma_attention_ranges.json'
        started=time.time();progress={'status':'loading','model':a.model,'expected':len(keys),'completed':0,'images':0,'started_unix':started}
        atomic_json(out/'progress.json',progress);sc.load();loaded=time.time();walls=[];cache=0
        with open(out/'inputs.jsonl','x') as proof:
            prepare=sc.prepare
            def captured_prepare(sample):
                result=prepare(sample);_,prompt,base_ids,inp=result
                proof.write(json.dumps({'sample_id':sample['id'],'prompt':prompt,'unexpanded_prompt_ids':base_ids,'expanded_prompt_ids':inp['input_ids'][0].tolist(),'processor_summary':base.closed.summarize(inp)},ensure_ascii=False)+'\n');proof.flush()
                return result
            sc.prepare=captured_prepare
            for sid in groups:
                sample=samples[sid]
                rows=sc.generate_ten(sample,eos)
                assert {task_id(a.model,base.task(sample,r['replicate'])) for r in rows}==set(groups[sid])
                assert len({r['seed'] for r in rows})==10
                for row in rows:ledger.add(task_id(a.model,base.task(sample,row['replicate'])),row)
                walls.append(rows[0]['wall_s']);cache+=sum(r['num_cached_tokens'] for r in rows)
                if migration and len(walls)==8:
                    assert len(ledger.keys)==80 and cache>0
                    atomic_json(out/'migration_admission8.json',{'model':a.model,'actual_images':8,'actual_answers':80,'input_expansion_native_eos_finite_logp_checked':True,'ten_unique_seeds_per_image':True,'num_cached_tokens':cache,'image_ids':list(groups)[:8],'counted_in_original_shard':True,'new_runtime_admission':res['model_admission'],'historical_source':history,'gpu':res,'warm_mean_image_s':sum(walls[1:])/7,'generation_parameters_changed':False})
                progress.update(status='running',completed=len(ledger.keys),images=len(walls),last_sample_id=sid,group_wall_s=sum(walls),load_s=loaded-started,mean_image_s=sum(walls)/len(walls),remaining_estimate_s=(len(groups)-len(walls))*sum(walls)/len(walls),updated_unix=time.time())
                atomic_json(out/'progress.json',progress)
        assert ledger.keys==set(keys) and cache>0
        atomic_json(out/'complete.json',{'generation_complete':True,'answers':len(keys),'images':len(groups),'zero_duplicates':True,'zero_missing':True,'finite_logprobs':True,'input_expansion_checks':'all_actual_requests','native_eos_checks':'all_actual_requests','num_cached_tokens':cache,'load_s':loaded-started,'group_wall_s':sum(walls),'elapsed_s':time.time()-started,'output_sha256':file_hash(out/'independent.jsonl'),'identity':ledger.identity,'semantic_labels_complete':False})
        progress.update(status='generation_complete',updated_unix=time.time());atomic_json(out/'progress.json',progress)
    except BaseException as e:
        atomic_json(out/'error.json',{'error':repr(e),'traceback':traceback.format_exc(),'no_automatic_retry':True});raise
    finally:
        if sc is not None and hasattr(sc,'llm'):
            sc.llm.llm_engine.engine_core.shutdown(timeout=30)
        if bridge is not None:bridge.close()

if __name__=='__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--host',required=True,choices=GPU_UUIDS);p.add_argument('--gpu',required=True);p.add_argument('--pieces',nargs='+',required=True);p.add_argument('--run',required=True);p.add_argument('--plan',action='store_true');run(p.parse_args())
