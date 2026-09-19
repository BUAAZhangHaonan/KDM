"""CPU-only valid-source mutations for portable non-census admission."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from kdm.decoding import DecodeConfig
from kdm.io import file_hash,stable_hash,stable_seed,read_jsonl
from kdm.pipeline import experiment_tasks,probe_tasks,task_id
from kdm.prompts import task_prompt,closed_prompt
from kdm.task_provenance import manifest_identity,validate_task_inputs,validate_measurement_methods


@pytest.fixture
def task_source(tmp_path,monkeypatch):
    import kdm.execution as execution
    monkeypatch.setattr(execution.subprocess,'check_output',lambda *a,**k:pytest.fail('Portable consumer queried GPUs'))
    monkeypatch.setattr(execution.socket,'gethostname',lambda:'different-central-consumer')
    root=tmp_path;frozen={}
    def save(name,value,jsonl=False):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(x)+'\n' for x in value) if jsonl else json.dumps(value))
        frozen[name]=file_hash(p);return p
    samples=[{'id':f's{i}','dataset':'food101','split':'eval' if i<2 else 'dev','question':'Which food?',
        'class':'class000','image_path':f'/logical/s{i}.jpg'} for i in range(3)]
    manifest=save('data/current/all.jsonl',samples,True)
    save('configs/kdm/models.json',[{'key':'m'}])
    methods=['vcd','m3id','dola','deco'];plan={'m':{'food101':methods}}
    save('configs/kdm/method_plan.json',plan)
    names=[f'class{i:03}' for i in range(101)];save('configs/kdm/food_aliases.json',{n:[n] for n in names})
    spec={'key':'m','factory':'synthetic:NeverLoaded','gpu_count':1}
    spec_path=save('configs/runtime/m.json',spec)
    hosts={'schema':1,'hosts':{'6403':{'hostname':'remote','root':'/remote/project','allowed_gpus':[1],'gpu_uuids':{'1':'remote-uuid'}}},
        'model_hosts':{'m':'6403'},'image_prefixes':{'/logical':'data/images'},'image_catalog':'catalog.json'}
    save('configs/runtime/hosts.json',hosts);save('catalog.json',{'synthetic':True});save('src/kdm/execution.py',{'fixture':'resolver'})
    save('src/kdm/models/official_vqa_normalizer.py',{'fixture':'never imported by rejection tests'})
    receipt={'host':'6403','hostname':'remote','project_root':'/remote/project','physical_gpus':['1'],'gpu_uuids':{'1':'remote-uuid'},
        'registry_sha256':frozen['configs/runtime/hosts.json'],'resolver_sha256':frozen['src/kdm/execution.py'],'image_catalog_sha256':frozen['catalog.json']}
    freeze={'status':'frozen','source_blobs':[{'fixture':'frozen-source'}],'files':frozen.copy()}
    fp=root/'outputs/records/preregistration_freeze.json';fp.parent.mkdir(parents=True);fp.write_text(json.dumps(freeze))
    def write(stage='experiment',shard=0,n_shards=1,suffix=''):
        definition={'backend':spec,'backend_spec_sha256':file_hash(spec_path),'schema':'kdm_current_v2','model':'m','stage':stage,
            'source_blobs':freeze['source_blobs'],'freeze_receipt_sha256':file_hash(fp),'execution':receipt,
            'manifest_sha256':file_hash(manifest),**manifest_identity(samples)}
        cfg=DecodeConfig(temperature=1.) if stage=='probe' else DecodeConfig()
        if stage=='closed':
            definition['names']=names
            rows=[{'key':stable_hash(['m',s['id'],'closed']),'status':'ok','model':'m','sample':s,'target':s['class'],
                'prompt':closed_prompt(s['question'],names),'candidate_scores':[{'label':n,'sum_logp':-1.,'mean_logp':-1.,'n_tokens':1} for n in names],
                'ranking_rule':'mean_log_probability','gold_rank':1} for s in samples if s['split']=='eval']
        else:
            definition.update(shard=shard,n_shards=n_shards,base_config=asdict(cfg))
            if stage=='experiment':definition.update(method_plan_sha256=frozen['configs/kdm/method_plan.json'],planned_dataset='food101',planned_methods=methods)
            tasks=list(experiment_tasks(samples,methods)) if stage=='experiment' else list(probe_tasks(samples))
            rows=[]
            for t in tasks:
                if int(stable_hash(t['sample']['id'])[:8],16)%n_shards!=shard:continue
                config=asdict(replace(cfg,method=t['method']));offset=[7,8] if t['method'] in {'m3id','instruction_m3id'} else None
                if offset is not None:config['m3id_offset']=len(offset)
                rows.append({**t,'key':task_id('m',t),'model':'m','status':'ok','text':'food','tokens':[7,1],'terminated':True,
                    'config':config,'offset_prompt_tokens':offset,'seed':stable_seed(t['sample']['id'],'m',t['replicate']),
                    'prompt':task_prompt(t['sample']['question'],t['marker'],t['guided'],t.get('attempt',False)),
                    'reference_prompt':task_prompt(t['sample']['question'],t['reference_marker'],t['reference_guided']) if t['method']!='direct' else None,
                    'neutral_prompt':task_prompt(t['sample']['question'],guided=False) if t['method'].startswith('instruction_') else None})
        identity=stable_hash(definition)
        path=root/f'{stage}_{shard}{suffix}.jsonl'
        path.write_text(''.join(json.dumps({**r,'identity':identity})+'\n' for r in rows))
        path.with_suffix('.identity.json').write_text(json.dumps({'definition':definition,'identity':identity}))
        return path
    return SimpleNamespace(root=root,freeze=freeze,spec_path=spec_path,manifest=manifest,write=write,samples=samples)


def rewrite(path,definition_mutation=None,row_mutation=None):
    meta=path.with_suffix('.identity.json');env=json.loads(meta.read_text());rows=list(read_jsonl(path))
    if definition_mutation:
        definition_mutation(env['definition']);env['identity']=stable_hash(env['definition'])
        for r in rows:r['identity']=env['identity']
    if row_mutation:row_mutation(rows)
    meta.write_text(json.dumps(env));path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def check(c,paths,stages=('experiment',)):
    return validate_task_inputs(c.root,paths,c.freeze,stages=stages,model='m')


def test_complete_legal_shards_and_separate_stages(task_source):
    c=task_source;paths=[c.write(shard=i,n_shards=2) for i in range(2)]
    r=check(c,paths);assert r['record_count']==128 and len(r['sources'])==2
    probe=c.write('probe');closed=c.write('closed')
    assert check(c,[probe],['probe'])['record_count']==20
    assert check(c,[closed],['closed'])['record_count']==2


def test_same_identity_fragments_are_accepted_but_missing_or_duplicate_keys_rejected(task_source):
    c=task_source;p=c.write();rows=list(read_jsonl(p));other=p.with_name('fragment.jsonl')
    other.with_suffix('.identity.json').write_bytes(p.with_suffix('.identity.json').read_bytes())
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows[:60]));other.write_text(''.join(json.dumps(r)+'\n' for r in rows[60:]))
    assert check(c,[p,other])['record_count']==128
    with pytest.raises(ValueError,match='Incomplete'):check(c,[p])
    other.write_text(other.read_text()+json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='duplicate'):check(c,[p,other])


@pytest.mark.parametrize('mutation',[
    'missing_sidecar','bad_digest','row_identity','source','freeze','backend','spec_hash','host','card','registry','resolver',
    'temperature','alpha','method','seed','prompt','sample','terminated','tokens','offset','missing','extra','shard','base_config','manifest','plan'])
def test_valid_source_mutations_are_rejected(task_source,mutation):
    c=task_source;p=c.write()
    if mutation=='missing_sidecar':p.with_suffix('.identity.json').unlink()
    elif mutation=='bad_digest':
        meta=p.with_suffix('.identity.json');env=json.loads(meta.read_text());env['identity']='bad';meta.write_text(json.dumps(env))
    elif mutation in {'source','freeze','backend','spec_hash','host','card','registry','resolver','base_config','manifest','plan','shard'}:
        def change(d):
            if mutation=='source':d['source_blobs']=['old']
            elif mutation=='freeze':d['freeze_receipt_sha256']='old'
            elif mutation=='backend':d['backend']={**d['backend'],'key':'other'}
            elif mutation=='spec_hash':d['backend_spec_sha256']='old'
            elif mutation=='host':d['execution']['host']='4028'
            elif mutation=='card':d['execution']['physical_gpus']=['0']
            elif mutation=='registry':d['execution']['registry_sha256']='old'
            elif mutation=='resolver':d['execution']['resolver_sha256']='old'
            elif mutation=='base_config':d['base_config']['temperature']=9
            elif mutation=='manifest':d['manifest_content_sha256']='old'
            elif mutation=='plan':d['planned_methods'].append('sid')
            elif mutation=='shard':d['shard']=1;d['n_shards']=2
        rewrite(p,change)
    else:
        def change(rows):
            r=rows[0]
            if mutation=='row_identity':r['identity']='another'
            elif mutation in {'temperature','alpha'}:r['config'][mutation]=9
            elif mutation=='method':r['method']='sid'
            elif mutation=='seed':r['seed']+=1
            elif mutation=='prompt':r['prompt']='changed'
            elif mutation=='sample':r['sample']['question']='changed'
            elif mutation=='terminated':r.pop('terminated')
            elif mutation=='tokens':r['tokens']=[]
            elif mutation=='offset':next(r for r in rows if r['method']=='m3id')['config']['m3id_offset']+=1
            elif mutation=='missing':rows.pop()
            elif mutation=='extra':rows.append(dict(rows[0]))
        rewrite(p,row_mutation=change)
    with pytest.raises(ValueError):check(c,[p])


def test_mixed_shard_definitions_and_concatenated_identities_are_rejected(task_source):
    c=task_source;paths=[c.write(shard=i,n_shards=2) for i in range(2)]
    rewrite(paths[1],lambda d:d.update(manifest_sha256='a'*64))
    with pytest.raises(ValueError,match='Mixed'):check(c,paths)
    merged=c.root/'concatenated.jsonl';merged.write_bytes(b''.join(p.read_bytes() for p in paths))
    merged.with_suffix('.identity.json').write_bytes(paths[0].with_suffix('.identity.json').read_bytes())
    with pytest.raises(ValueError,match='identity'):check(c,[merged])


@pytest.mark.parametrize('stage,mutation',[('probe','repeat'),('probe','temperature'),('closed','names'),('closed','prompt')])
def test_probe_and_closed_keep_frozen_definitions(task_source,stage,mutation):
    c=task_source;p=c.write(stage)
    def change(rows):
        if mutation=='repeat':rows[0]['replicate']=11
        elif mutation=='temperature':rows[0]['config']['temperature']=0
        elif mutation=='names':rows[0]['candidate_scores'][0]['label']='forged'
        else:rows[0]['prompt']='forged'
    rewrite(p,row_mutation=change)
    with pytest.raises(ValueError):check(c,[p],[stage])


def test_measurement_sid_requires_task_level_applicability(task_source):
    c=task_source;r=check(c,[c.write()])
    validate_measurement_methods(c.root,r,['vcd','instruction_m3id'])
    with pytest.raises(ValueError,match='undefined'):validate_measurement_methods(c.root,r,['sid'])


@pytest.mark.parametrize('command',['mechanism','replay'])
def test_cli_input_gate_precedes_backend(task_source,monkeypatch,command):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    c=task_source;p=c.write();rewrite(p,row_mutation=lambda rows:rows[0]['config'].update(temperature=9))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1');monkeypatch.setattr(protocol,'validate_runtime',lambda *a:None)
    monkeypatch.setattr(protocol,'validate_freeze',lambda *a:c.freeze)
    monkeypatch.setattr(pipeline,'make_backend',lambda *a:pytest.fail('Backend loaded before input provenance'))
    with pytest.raises(ValueError,match='configuration'):
        cli.main(['--root',str(c.root),command,'--records',str(p),'--model-spec',str(c.spec_path),'--model','m','--gpu','1','--out','result.jsonl'])


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_complete_input_gate_precedes_backend_and_human_review(task_source,monkeypatch):
    c=task_source;p=c.write();rewrite(p,lambda d:d.update(source_blobs=['old']))
    mod=load_script('complete_response_audit')
    monkeypatch.setattr(mod,'execution_identity',lambda *a:{'source_blobs':c.freeze['source_blobs']})
    monkeypatch.setattr(mod,'make_backend',lambda *a:pytest.fail('Backend loaded before input provenance'))
    monkeypatch.setattr('sys.argv',['complete','--root',str(c.root),'--records',str(p),'--manifest',str(c.manifest),
        '--annotations','missing','--model-spec',str(c.spec_path),'--model','m','--gpu','1','--out','result.jsonl'])
    with pytest.raises(ValueError,match='source or freeze'):mod.main()


@pytest.mark.parametrize('relative',[False,True])
def test_report_gate_precedes_scoring_and_output(task_source,monkeypatch,relative):
    import kdm.protocol as protocol
    c=task_source;p=c.write();rewrite(p,lambda d:d['execution'].update(host='4028'))
    mod=load_script('build_reports');monkeypatch.setattr(protocol,'validate_freeze',lambda *a:c.freeze)
    def arg(path):return str(Path(path).relative_to(c.root)) if relative else str(path)
    if relative:monkeypatch.chdir(c.root.parent)
    monkeypatch.setattr('sys.argv',['report','--root',str(c.root),'--records',arg(p),'--closed',arg(c.write('closed')),
        '--manifest',arg(c.manifest),'--selection','missing','--method-plan',arg(c.root/'configs/kdm/method_plan.json'),
        '--annotations','missing','--aliases',arg(c.root/'configs/kdm/food_aliases.json'),
        '--normalizer',arg(c.root/'src/kdm/models/official_vqa_normalizer.py'),'--out-dir','report'])
    with pytest.raises(ValueError,match='Execution host'):mod.main()
    assert not (c.root/'report').exists()


def test_actual_cpu_producer_rows_satisfy_consumer_contract(task_source):
    from PIL import Image
    from kdm.models.mock import MockBackend
    from kdm.pipeline import run_tasks
    c=task_source;image=c.root/'outputs/verification/image.png';image.parent.mkdir(parents=True);Image.new('RGB',(8,8),(80,90,100)).save(image)
    for sample in c.samples:sample['image_path']=str(image)
    c.manifest.write_text(''.join(json.dumps(s)+'\n' for s in c.samples))
    c.freeze['files']['data/current/all.jsonl']=file_hash(c.manifest)
    (c.root/'outputs/records/preregistration_freeze.json').write_text(json.dumps(c.freeze))
    template=c.write();definition=json.loads(template.with_suffix('.identity.json').read_text())['definition']
    out=c.root/'CPU_SYNTHETIC_PRODUCER.jsonl'
    run_tasks(MockBackend(),'m',experiment_tasks(c.samples),out,definition)
    assert check(c,[out])['record_count']==128
    offsets=[r for r in read_jsonl(out) if r['method'] in {'m3id','instruction_m3id'}]
    assert offsets and all(r['config']['m3id_offset']==len(r['offset_prompt_tokens'])==1 for r in offsets)


@pytest.mark.parametrize('command',['mechanism','replay'])
def test_valid_shard_union_reaches_backend_after_portable_gate(task_source,monkeypatch,command):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    c=task_source;paths=[c.write(shard=i,n_shards=2) for i in range(2)]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1');monkeypatch.setattr(protocol,'validate_runtime',lambda *a:None)
    monkeypatch.setattr(protocol,'validate_freeze',lambda *a:c.freeze)
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda *a:None)
    def admitted(*args):raise RuntimeError('validated legal union reached backend')
    monkeypatch.setattr(pipeline,'make_backend',admitted)
    with pytest.raises(RuntimeError,match='validated legal union'):
        cli.main(['--root',str(c.root),command,'--records',*[str(p) for p in paths],
            '--model-spec',str(c.spec_path),'--model','m','--gpu','1','--out','result.jsonl'])


@pytest.mark.parametrize('stage',['experiment','probe','closed'])
def test_cli_cpu_production_and_portable_consumption_agree(task_source,monkeypatch,stage):
    from PIL import Image
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    from kdm.models.mock import MockBackend
    c=task_source;image=c.root/'outputs/verification/image.png';image.parent.mkdir(parents=True);Image.new('RGB',(8,8),(80,90,100)).save(image)
    for sample in c.samples:sample['image_path']=str(image)
    c.manifest.write_text(''.join(json.dumps(s)+'\n' for s in c.samples))
    c.freeze['files']['data/current/all.jsonl']=file_hash(c.manifest)
    (c.root/'outputs/records/preregistration_freeze.json').write_text(json.dumps(c.freeze))
    template=c.write();receipt=json.loads(template.with_suffix('.identity.json').read_text())['definition']['execution']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *a:receipt)
    monkeypatch.setattr(protocol,'validate_freeze',lambda *a:c.freeze)
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda *a:None)
    monkeypatch.setattr(pipeline,'make_backend',lambda *a:MockBackend())
    out=c.root/f'CPU_SYNTHETIC_CLI_{stage}.jsonl'
    args=['--root',str(c.root),'closed-probe' if stage=='closed' else 'run','--manifest',str(c.manifest),
        '--model-spec',str(c.spec_path),'--model','m','--gpu','1','--out',str(out)]
    if stage!='closed':args+=['--mode',stage]
    if stage=='experiment':args+=['--method-plan',str(c.root/'configs/kdm/method_plan.json')]
    cli.main(args)
    assert check(c,[out],[stage])['record_count']=={'experiment':128,'probe':20,'closed':2}[stage]
