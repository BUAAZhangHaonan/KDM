"""Read-only provenance checks for frozen census ledgers and legal shard unions."""
from dataclasses import asdict
import json
from pathlib import Path
from .decoding import DecodeConfig
from .io import file_hash, read_jsonl, stable_hash, stable_seed, within
from .pipeline import census_tasks, task_id
from .prompts import task_prompt


def validate_census_inputs(root, paths, manifest_path, freeze, require_complete_panel=True, *, allow_mock=False):
    """Validate every model's complete census, permitting a subset of models for annotation.

    ``freeze`` must come from the caller's validate_freeze gate. Source identity is
    checked against that receipt, never replaced by the current Git state here.
    Multiple paths for one model are aggregated without rewriting their ledgers.
    Shard identities may differ only in ``shard``. Same-identity fragments and
    merges are accepted when their union is complete and contains no duplicate key.
    Mock admission is explicit and additionally restricted to verification inputs
    and a frozen CPU_TEST_ONLY MockBackend spec; callers must keep mock outputs in
    verification too. This function performs no writes and never loads a model.
    """
    root=Path(root).resolve()
    if freeze.get('status')!='frozen' or not isinstance(freeze.get('source_blobs'),list) or not freeze['source_blobs']:
        raise ValueError('Census provenance requires a validated frozen source anchor')
    frozen=freeze.get('files',{})
    def frozen_file(relative):
        path=within(root,relative)
        if relative not in frozen or file_hash(path)!=frozen[relative]:
            raise ValueError('Census frozen input changed or is unregistered: '+relative)
        return path
    panel=json.loads(frozen_file('configs/kdm/models.json').read_text())
    candidates=[r['key'] for r in panel]
    if not candidates or len(set(candidates))!=len(candidates):raise ValueError('Invalid frozen census panel')
    manifest_path=within(root,manifest_path)
    expected_manifest=file_hash(frozen_file('data/current/all.jsonl'))
    if file_hash(manifest_path)!=expected_manifest:raise ValueError('Census manifest differs from the frozen complete manifest')
    samples=list(read_jsonl(manifest_path))
    if not samples or len({s['id'] for s in samples})!=len(samples):raise ValueError('Census manifest is empty or duplicated')
    paths=[within(root,path) for path in paths]
    if not paths or len(set(paths))!=len(paths):raise ValueError('Empty or duplicate census input path')
    freeze_sha=file_hash(root/'outputs/records/preregistration_freeze.json')
    expected_config=asdict(DecodeConfig())
    models={};sources=[];specs={}
    for path in paths:
        meta=path.with_suffix('.identity.json')
        if not meta.is_file():raise ValueError('Census input requires its original ledger identity sidecar')
        meta_sha=file_hash(meta)
        envelope=json.loads(meta.read_text());definition=envelope.get('definition');identity=envelope.get('identity')
        if not isinstance(definition,dict) or not identity or stable_hash(definition)!=identity:
            raise ValueError('Census ledger identity digest mismatch')
        model=definition.get('model')
        if model not in candidates:raise ValueError('Unexpected census model in ledger definition')
        if model not in specs:
            relative=f'configs/runtime/{model}.json';spec_path=frozen_file(relative)
            specs[model]=(json.loads(spec_path.read_text()),frozen[relative])
        spec,spec_sha=specs[model]
        if definition.get('schema')!='kdm_current_v2':raise ValueError('Unexpected census ledger schema')
        if spec.get('key')!=model or definition.get('backend')!=spec or definition.get('backend_spec_sha256')!=spec_sha:
            raise ValueError('Census backend differs from its frozen model specification')
        is_mock=spec.get('purpose')=='CPU_TEST_ONLY' or spec.get('factory')=='kdm.models.mock:MockBackend'
        if is_mock:
            if not allow_mock or not path.is_relative_to(root/'outputs/verification') or spec.get('purpose')!='CPU_TEST_ONLY' or spec.get('factory')!='kdm.models.mock:MockBackend':
                raise ValueError('Mock census requires explicit verification scope and MockBackend')
            expected_sources={'software_fixture':True,'formal_evidence':False}
        else:
            expected_sources=freeze['source_blobs']
            if definition.get('freeze_receipt_sha256')!=freeze_sha:raise ValueError('Census ledger freeze receipt differs from the active frozen receipt')
        if definition.get('source_blobs')!=expected_sources:raise ValueError('Census source blobs differ from the frozen source anchor')
        if definition.get('manifest_sha256')!=expected_manifest:raise ValueError('Census ledger manifest differs from the frozen manifest')
        if definition.get('base_config')!=expected_config:raise ValueError('Census base configuration differs from fixed greedy decoding')
        shard=definition.get('shard');n_shards=definition.get('n_shards')
        if type(shard) is not int or type(n_shards) is not int or n_shards<1 or not 0<=shard<n_shards:
            raise ValueError('Invalid census shard declaration')
        common={k:v for k,v in definition.items() if k!='shard'}
        common_hash=stable_hash(common)
        if model not in models:
            expected={task_id(model,t):t for t in census_tasks(samples)}
            models[model]={'paths':[],'identities':[],'common_definition_sha256':common_hash,'record_count':0,'formal_evidence':not is_mock,'_expected':expected,'_seen':set()}
        group=models[model]
        if group['common_definition_sha256']!=common_hash:raise ValueError('Mixed census run definitions for one model')
        before=file_hash(path);count=0
        for row in read_jsonl(path):
            if row.get('identity')!=identity:raise ValueError('Raw census record identity differs from its sidecar')
            key=row.get('key');task=group['_expected'].get(key)
            if task is None or key in group['_seen']:raise ValueError('Unexpected or duplicate census task across input files')
            if row.get('model')!=model or any(row.get(k)!=v for k,v in task.items()):raise ValueError('Census task differs from the frozen manifest')
            if int(stable_hash(task['sample']['id'])[:8],16)%n_shards!=shard:raise ValueError('Census record belongs to a different shard')
            if row.get('config')!=expected_config:raise ValueError('Census row configuration differs from fixed greedy decoding')
            if row.get('seed')!=stable_seed(task['sample']['id'],model,task['replicate']):raise ValueError('Census seed differs from the fixed task seed')
            prompt=task_prompt(task['sample']['question'],task['marker'],task['guided'])
            if row.get('prompt')!=prompt or row.get('reference_prompt') is not None or row.get('neutral_prompt') is not None:
                raise ValueError('Census prompt differs from the fixed task prompt')
            if row.get('status')!='ok' or not isinstance(row.get('text'),str) or not isinstance(row.get('tokens'),list) or not row['tokens'] or not all(type(t) is int for t in row['tokens']) or type(row.get('terminated')) is not bool:
                raise ValueError('Census lacks successful real token and termination evidence')
            group['_seen'].add(key);count+=1
        if file_hash(path)!=before:raise ValueError('Census input changed while its provenance was checked')
        if file_hash(meta)!=meta_sha:raise ValueError('Census identity sidecar changed while its provenance was checked')
        relative=str(path.relative_to(root));group['paths'].append(relative);group['identities'].append(identity);group['record_count']+=count
        sources.append({'path':relative,'sha256':before,'sidecar_path':str(meta.relative_to(root)),'sidecar_sha256':meta_sha,'identity':identity,'model':model,'shard':shard,'n_shards':n_shards,'records':count})
    for model,group in models.items():
        if group.pop('_seen')!=set(group.pop('_expected')):raise ValueError('Incomplete guided or unguided census across input files: '+model)
    if require_complete_panel and set(models)!=set(candidates):raise ValueError('Missing fixed candidate census: '+', '.join(sorted(set(candidates)-set(models))))
    return {'schema':'kdm_census_input_provenance_v1','manifest_sha256':expected_manifest,'freeze_receipt_sha256':freeze_sha,'source_blobs':freeze['source_blobs'],'models':models,'sources':sources,'complete_panel':set(models)==set(candidates),'formal_evidence':all(v['formal_evidence'] for v in models.values())}


def census_input_paths(paths):
    """Find census files in a mixed-stage annotation/report input list."""
    result=[]
    for path in paths:
        kinds={row.get('kind') for row in read_jsonl(path)}
        if kinds & {'census','unguided'}:
            if not kinds <= {'census','unguided'}:raise ValueError('Census and other stages cannot share one raw ledger')
            result.append(path)
    return result
