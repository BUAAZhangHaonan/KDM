"""Portable frozen provenance for experiment, independent-probe and closed ledgers."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from .decoding import DecodeConfig
from .io import file_hash, read_jsonl, stable_hash, stable_seed, within
from .pipeline import experiment_tasks, probe_tasks, task_id
from .prompts import task_prompt, closed_prompt


def manifest_identity(samples):
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Empty or duplicate task manifest')
    return {'datasets': sorted({s['dataset'] for s in samples}),
            'manifest_content_sha256': stable_hash(sorted(samples, key=lambda s:s['id']))}


def validate_task_inputs(root, paths, freeze, *, stages, model=None):
    """Check complete legal shard unions against the frozen producing identity.

    The caller must obtain freeze from validate_freeze. This checks recorded host
    receipts without loading a model, checking local weights, or querying GPUs.
    Different models/stages/dataset scopes may coexist as separate ledgers;
    fragments of one scope may differ only in shard. Every row and sidecar hash
    is preserved in the returned source receipt.
    """
    root=Path(root).resolve(); stages=set(stages)
    if not stages or not stages <= {'experiment','probe','closed'}:
        raise ValueError('Unsupported task provenance stage')
    if freeze.get('status')!='frozen' or not freeze.get('source_blobs'):
        raise ValueError('Task provenance requires a validated frozen source anchor')
    frozen=freeze['files']
    def frozen_path(relative):
        path=within(root,relative)
        if relative not in frozen or file_hash(path)!=frozen[relative]:
            raise ValueError('Task frozen input changed or is unregistered: '+relative)
        return path
    original=list(read_jsonl(frozen_path('data/current/all.jsonl')))
    panel=json.loads(frozen_path('configs/kdm/models.json').read_text())
    candidates={r['key'] for r in panel}
    plan=json.loads(frozen_path('configs/kdm/method_plan.json').read_text())
    names=sorted(json.loads(frozen_path('configs/kdm/food_aliases.json').read_text()))
    freeze_sha=file_hash(root/'outputs/records/preregistration_freeze.json')
    paths=[within(root,p) for p in paths]
    if not paths or len(paths)!=len(set(paths)):raise ValueError('Empty or duplicate task input path')
    groups={};sources=[];seen_global=set();specs={}
    for path in paths:
        meta=path.with_suffix('.identity.json')
        if not meta.is_file():raise ValueError('Formal task input requires its original ledger identity sidecar')
        before=file_hash(path);meta_before=file_hash(meta)
        envelope=json.loads(meta.read_text());definition=envelope.get('definition');identity=envelope.get('identity')
        if not isinstance(definition,dict) or not identity or stable_hash(definition)!=identity:
            raise ValueError('Task ledger identity digest mismatch')
        current=definition.get('model');stage=definition.get('stage')
        if current not in candidates or (model is not None and current!=model):raise ValueError('Unexpected task input model')
        if stage not in stages:raise ValueError('Unexpected task input stage')
        if current not in specs:
            relative=f'configs/runtime/{current}.json'
            specs[current]=(json.loads(frozen_path(relative).read_text()),frozen[relative])
        spec,spec_sha=specs[current]
        if spec.get('purpose')=='CPU_TEST_ONLY' or spec.get('factory')=='kdm.models.mock:MockBackend':
            raise ValueError('Formal task provenance cannot admit mock evidence')
        if definition.get('schema')!='kdm_current_v2' or spec.get('key')!=current or definition.get('backend')!=spec or definition.get('backend_spec_sha256')!=spec_sha:
            raise ValueError('Task backend differs from its frozen model specification')
        if definition.get('freeze_receipt_sha256')!=freeze_sha or definition.get('source_blobs')!=freeze['source_blobs']:
            raise ValueError('Task source or freeze identity differs from frozen evidence')
        from .execution import validate_execution_receipt
        validate_execution_receipt(root,definition.get('execution'),current,spec.get('gpu_count'))
        datasets=definition.get('datasets')
        if not isinstance(datasets,list) or datasets!=sorted(set(datasets)) or not datasets or not set(datasets)<={s['dataset'] for s in original}:
            raise ValueError('Invalid task manifest dataset declaration')
        samples=[s for s in original if s['dataset'] in datasets]
        if definition.get('manifest_content_sha256')!=manifest_identity(samples)['manifest_content_sha256']:
            raise ValueError('Task manifest differs from complete frozen dataset contents')
        if not isinstance(definition.get('manifest_sha256'),str) or len(definition['manifest_sha256'])!=64:
            raise ValueError('Task ledger lacks original manifest byte identity')
        scope=(current,stage,tuple(datasets))
        if stage=='experiment':
            if len(datasets)!=1:raise ValueError('Experiment requires one complete dataset condition')
            methods=plan[current][datasets[0]]
            declared=definition.get('planned_methods')
            if definition.get('planned_dataset')!=datasets[0] or not isinstance(declared,list) or len(declared)!=len(set(declared)) or set(declared)!=set(methods) or definition.get('method_plan_sha256')!=frozen['configs/kdm/method_plan.json']:
                raise ValueError('Task methods differ from frozen model/dataset plan')
            expected={task_id(current,t):t for t in experiment_tasks(samples,methods)}
            config=DecodeConfig()
        elif stage=='probe':
            expected={task_id(current,t):t for t in probe_tasks(samples)}
            config=DecodeConfig(temperature=1.,top_p=1.)
        else:
            if definition.get('names')!=names or len(names)!=101:raise ValueError('Closed ledger must retain the frozen 101 candidates')
            expected={stable_hash([current,s['id'],'closed']):s for s in samples if s['dataset']=='food101' and s['split']=='eval'}
            config=None
        if not expected:raise ValueError('Empty frozen task scope')
        if stage=='closed':
            shard=0;n_shards=1
        else:
            if definition.get('base_config')!=asdict(config):raise ValueError('Task base configuration differs from the fixed protocol')
            shard=definition.get('shard');n_shards=definition.get('n_shards')
            if type(shard) is not int or type(n_shards) is not int or n_shards<1 or not 0<=shard<n_shards:
                raise ValueError('Invalid task shard declaration')
        common=stable_hash({k:v for k,v in definition.items() if k!='shard'})
        if scope not in groups:groups[scope]={'common':common,'expected':expected,'seen':set()}
        group=groups[scope]
        if group['common']!=common:raise ValueError('Mixed task run definitions for one model/stage/dataset scope')
        count=0
        for row in read_jsonl(path):
            key=row.get('key')
            if row.get('identity')!=identity:raise ValueError('Raw task record identity differs from its sidecar')
            if key not in expected or key in seen_global:raise ValueError('Unexpected or duplicate task across input files')
            if row.get('model')!=current or row.get('status')!='ok':raise ValueError('Task model or completion status mismatch')
            task=expected[key];sample=task if stage=='closed' else task['sample']
            if stage=='closed':
                if row.get('sample')!=sample or row.get('target')!=sample['class'] or row.get('prompt')!=closed_prompt(sample['question'],names):
                    raise ValueError('Closed record differs from its frozen task')
                scores=row.get('candidate_scores',[])
                if len(scores)!=101 or {s.get('label') for s in scores}!=set(names):raise ValueError('Closed candidates differ from frozen names')
            else:
                if any(row.get(k)!=v for k,v in task.items()):raise ValueError('Raw task differs from frozen task definition')
                if int(stable_hash(sample['id'])[:8],16)%n_shards!=shard:raise ValueError('Task belongs to a different shard')
                expected_config=asdict(replace(config,method=task['method']))
                if task['method'] in {'m3id','instruction_m3id'}:
                    offset_tokens=row.get('offset_prompt_tokens')
                    if not isinstance(offset_tokens,list) or not offset_tokens or any(type(t) is not int or t<0 for t in offset_tokens):
                        raise ValueError('M3ID requires recorded original offset prompt tokens')
                    expected_config['m3id_offset']=len(offset_tokens)
                elif row.get('offset_prompt_tokens') is not None:raise ValueError('Unexpected derived offset tokens')
                if row.get('config')!=expected_config:raise ValueError('Task row configuration differs from fixed decoding')
                if row.get('seed')!=stable_seed(sample['id'],current,task['replicate']):raise ValueError('Task seed differs from fixed seed')
                prompt=task_prompt(sample['question'],task['marker'],task['guided'],task.get('attempt',False))
                reference=task_prompt(sample['question'],task['reference_marker'],task['reference_guided']) if task['method']!='direct' else None
                neutral=task_prompt(sample['question'],guided=False) if task['method'].startswith('instruction_') else None
                if row.get('prompt')!=prompt or row.get('reference_prompt')!=reference or row.get('neutral_prompt')!=neutral:
                    raise ValueError('Task prompts differ from frozen protocol')
                if not isinstance(row.get('text'),str) or not isinstance(row.get('tokens'),list) or not row['tokens'] or any(type(t) is not int or t<0 for t in row['tokens']) or type(row.get('terminated')) is not bool:
                    raise ValueError('Task lacks successful original token and termination evidence')
            seen_global.add(key);group['seen'].add(key);count+=1
        if file_hash(path)!=before or file_hash(meta)!=meta_before:raise ValueError('Task input changed during provenance verification')
        sources.append({'path':str(path.relative_to(root)),'sha256':before,'sidecar_path':str(meta.relative_to(root)),
            'sidecar_sha256':meta_before,'identity':identity,'model':current,'stage':stage,'datasets':datasets,'shard':shard,'n_shards':n_shards,'records':count})
    for scope,group in groups.items():
        if group['seen']!=set(group['expected']):raise ValueError('Incomplete frozen task union: '+str(scope))
    return {'schema':'kdm_task_input_provenance_v1','freeze_receipt_sha256':freeze_sha,'source_blobs':freeze['source_blobs'],
        'sources':sources,'formal_evidence':True,'record_count':len(seen_global)}


def validate_measurement_methods(root, provenance, methods):
    """Retain the frozen task-level applicability while measuring stored paths."""
    methods=tuple(methods)
    allowed={'vcd','m3id','dola','deco','sid','instruction_vcd','instruction_m3id'}
    if not methods or len(methods)!=len(set(methods)) or not set(methods)<=allowed:
        raise ValueError('Unsupported or duplicate measurement methods')
    plan=json.loads((Path(root)/'configs/kdm/method_plan.json').read_text())
    for source in provenance['sources']:
        for dataset in source['datasets']:
            if not (set(methods)-{'instruction_vcd','instruction_m3id'})<=set(plan[source['model']][dataset]):
                raise ValueError('Measurement method is undefined for the frozen model/dataset task')
