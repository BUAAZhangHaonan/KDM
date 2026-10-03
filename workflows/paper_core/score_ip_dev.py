#!/usr/bin/env python3
"""CPU scoring of explicit sealed nine-model IP-only fixed-dev pilot sources."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, within
from kdm.prompts import MARKERS
from workflows.main_results import score as frozen
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.score_dev_viz import (audit_raw, food_metrics, native_decisions,
    FIELDS, COND, REFERENCE, AUTHORITY, NATIVE_RECEIPT)
from workflows.paper_core.score_native import save_json, save_rows
from workflows.supplemental.remaining11.score import load_authority, infer_qa, rows, score_target
CORE = ('qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b')
EXTRA = ('internvl35_8b', 'onevision', 'phi35', 'qwen3vl')
EXTRA_REF = 'outputs/supplemental/remaining4/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference'
DEV_RECEIPT = 'outputs/paper_core_20261002_dev_viz/scoring_dev_completed80_closed_root1_20261003_0005/receipt.json'
EXECUTOR = dict(agent='/root/final_cpu_intake', model='gpt-6.1-sol', effort='max', call_id='')


def pilot_sources(manifest_path, samples, roster_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'kdm_nine_model_IP_only_dev404_all416_actual_pilot_budget_v1':
        raise ValueError('Requires the explicit actually received immutable pilot manifest')
    selected, sources, seen = [], [], set()
    expected_ids = list(samples)[:8]
    for source in manifest['sources']:
        model = source['model']
        if model not in CORE + EXTRA or source['producer_active_at_collection'] is not False or source['central_copies_have_identical_bytes'] is not True:
            raise ValueError('A producer is active, unregistered or not actually transferred')
        receipt_path = within(ROOT, source['source_receipt'])
        if file_hash(receipt_path) != source['source_receipt_sha256']:
            raise ValueError('Actual pilot receipt changed')
        receipt = json.loads(receipt_path.read_text())
        identity_path = receipt_path.with_name('identity.json')
        identity = json.loads(identity_path.read_text())
        plan = identity['plan']
        spec = json.loads((ROOT / f'configs/runtime/{model}.json').read_text())
        if (receipt['passed'] is not True or receipt['plan_sha256'] != stable_hash(plan)
                or plan['schema'] != 'kdm_ip_only_dev404_pilot_plan_v1' or plan['model'] != model
                or plan['stage'] != 'dev404' or plan['fixed_roster_sha256'] != file_hash(roster_path)
                or plan['sample_ids'] != expected_ids or plan['authorized_inputs_per_condition'] != 8
                or plan['frozen_spec'] != spec or plan['scientific_parameters_changed'] is not False
                or set(plan['methods']) - {'instruction_vcd', 'instruction_m3id'}
                or set(plan['markers']) - set(MARKERS)
                or (model in CORE and plan['methods'] != ['instruction_m3id'])):
            raise ValueError('Pilot identity differs from the frozen roster, model or allowed IP scope')
        actual = identity['runtime_spec']
        if actual['hf_model_id'] != spec['hf_model_id'] or actual['dtype'] != spec['dtype']:
            raise ValueError('Actual checkpoint or registered precision differs')
        admitted = identity['actual_admission']
        if model in EXTRA and admitted['runtime_spec'] != actual:
            raise ValueError('The actual runtime differs from its admission')
        if model in CORE and admitted['fixed_identity'] is None:
            raise ValueError('Actual original-core admission is absent')
        wanted_keys = set(plan['pilot_keys'])
        source_count = 0
        for part in source['conditions']:
            raw = within(ROOT, part['raw'])
            complete_path = raw.with_suffix('.complete.json')
            sidecar_path = raw.with_suffix('.identity.json')
            complete = json.loads(complete_path.read_text())
            sidecar = json.loads(sidecar_path.read_text())
            if (part['scope'] != 'pilot_only' or part['rows'] != 8 or part['passed'] is not True
                    or complete['scope'] != 'pilot_only' or complete['rows'] != 8 or complete['passed'] is not True
                    or file_hash(raw) != part['raw_sha256'] or complete['raw_sha256'] != part['raw_sha256']
                    or sidecar['definition'] != {**identity, 'model':model, 'shard':0, 'n_shards':1, 'base_config':asdict(DecodeConfig())} or sidecar['identity'] != stable_hash(sidecar['definition'])
                    or complete['method'] != part['method'] or complete['marker'] != part['marker']
                    or not any(p == complete for p in receipt['conditions'])):
                raise ValueError('Immutable completed pilot part or original ledger identity differs')
            if complete.get('operator_audit_sha256'):
                operator_path = raw.with_suffix('.operator.json')
                if file_hash(operator_path) != complete['operator_audit_sha256']:
                    raise ValueError('Original K100 three-route operator evidence changed')
            group = list(rows(raw))
            if len(group) != 8 or {r['sample']['id'] for _, r, _ in group} != set(expected_ids):
                raise ValueError('Actual pilot part is not the fixed original eight samples')
            if {r['key'] for _, r, _ in group} != set(complete['completed_keys']):
                raise ValueError('Actual raw keys differ from the completed receipt')
            for line, row, line_sha in group:
                sample = samples.get(row['sample']['id'])
                if sample is None or row['identity'] != sidecar['identity'] or row['key'] in seen or row['key'] not in wanted_keys:
                    raise ValueError('Raw pilot sample, ledger identity or unique key differs')
                audit_raw(row, sample, model)
                if (row['method'] != part['method'] or row['marker'] != part['marker']
                        or row['kind'] != 'instruction_preserving' or row['reference_marker'] != row['marker']
                        or row['guided'] is not True or row['reference_guided'] is not False or row['replicate'] != 0):
                    raise ValueError('The recorded IP condition differs from its admitted pilot')
                seen.add(row['key'])
                selected.append((row, dict(source_host=source['host'], source_original_root=source['source_root'],
                    source_path=str(raw), source_line=line, raw_line_sha256=line_sha,
                    raw_source_sha256=part['raw_sha256'], source_identity=row['identity'],
                    complete_source_path=str(complete_path), complete_source_sha256=file_hash(complete_path),
                    identity_source_path=str(sidecar_path), identity_source_sha256=file_hash(sidecar_path),
                    source_scope='pilot_partial', expected_n=404, source_raw_complete=False,
                    model_checkpoint=spec['hf_model_id'], model_dtype=spec['dtype'],
                    actual_admission_identity_sha256=file_hash(identity_path))))
                source_count += 1
            sources.append(dict(model=model, method=part['method'], marker=part['marker'], rows=8,
                expected_n=404, source_scope='pilot_partial', raw_complete=False,
                raw_path=str(raw), raw_sha256=part['raw_sha256'], source_host=source['host'],
                source_original_root=source['source_root'], complete_path=str(complete_path),
                complete_sha256=file_hash(complete_path), identity_path=str(sidecar_path),
                identity_sha256=file_hash(sidecar_path), runner_sha256=plan['runner_sha256']))
        if source_count != source['actual_rows'] or source_count != receipt['actual_rows']:
            raise ValueError('Actual completed pilot source row count differs')
    if len(selected) != manifest['actual_unique_pilot_rows']:
        raise ValueError('Actual unique pilot total differs')
    return selected, sources


def completed_dev_sources(manifest_path, samples, roster_path):
    """Only explicit actually copied complete parts; never enumerate active raw."""
    from kdm.pipeline import task_id
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'kdm_IP_dev_completed_parts_source_index_v1':
        raise ValueError('An explicit received completed-part index is required')
    selected, sources, seen = [], [], set()
    for source in manifest['sources']:
        model = source['model']
        if model not in CORE + EXTRA:
            raise ValueError('An unregistered model entered the IP-dev source index')
        files = {}
        for field in ('raw','complete','identity','claim_identity','claim_plan','missing_keys','budget_release'):
            p = within(ROOT,source[field])
            if file_hash(p) != source[field+'_sha256']:
                raise ValueError('A received completed producer member changed: '+field)
            files[field] = p
        complete = json.loads(files['complete'].read_text())
        sidecar = json.loads(files['identity'].read_text())
        identity = json.loads(files['claim_identity'].read_text())
        plan = json.loads(files['claim_plan'].read_text())
        budget = json.loads(files['budget_release'].read_text())
        spec = json.loads((ROOT/f'configs/runtime/{model}.json').read_text())
        if (plan != identity['plan'] or plan['schema'] != 'kdm_IP_only_dev404_budgeted_missing_claim_v1'
                or plan['model'] != model or plan['claim_id'] != source['claim_id']
                or plan['stage'] != 'dev404' or plan['sample_roster_sha256'] != file_hash(roster_path)
                or plan['original_spec'] != spec or plan['scientific_parameters_changed'] is not False
                or plan['reused_pilot_key_overlap'] != 0
                or plan['missing_keys_sha256'] != source['missing_keys_sha256']
                or plan['inherited_budget_release_sha256'] != source['budget_release_sha256']
                or set(plan['methods']) - {'instruction_vcd','instruction_m3id'}
                or set(plan['markers']) - set(MARKERS)
                or (model in CORE and plan['methods'] != ['instruction_m3id'])
                or budget['schema'] != 'kdm_nine_model_IP_only_dev404_all416_actual_pilot_budget_v1'
                or budget['released'] is not True or budget['actual_unique_pilot_rows'] != 416
                or budget['authorized_total_GPU_hours_cap'] != 24
                or budget['estimated_total_GPU_hours'] > 24):
            raise ValueError('The source claim differs from registered scope or actual budget release')
        actual = identity['runtime_spec']
        if actual['hf_model_id'] != spec['hf_model_id'] or actual['dtype'] != spec['dtype']:
            raise ValueError('The completed-part actual checkpoint or original precision differs')
        if model in EXTRA and identity['actual_admission']['runtime_spec'] != actual:
            raise ValueError('The actual supplemental admission differs from its recorded runtime')
        if (sidecar['definition'] != {**identity,'model':model,'shard':0,'n_shards':1,'base_config':asdict(DecodeConfig())}
                or sidecar['identity'] != stable_hash(sidecar['definition'])
                or complete['scope'] != 'budgeted_dev404' or complete['passed'] is not True
                or complete['model'] != model or not 1 <= complete['rows'] <= 128
                or complete['rows'] != source['rows'] or complete['raw_sha256'] != source['raw_sha256']
                or complete['completed_keys'] != source['completed_keys']):
            raise ValueError('The immutable completed part or original ledger differs')
        missing = [r for _,r,_ in rows(files['missing_keys'])]
        allowed = {r['key'] for r in missing}
        if len(allowed) != len(missing) or len(allowed) != plan['expected_rows'] or allowed.intersection(budget['reused_pilot_keys']):
            raise ValueError('The original missing-key claim duplicates or repeats the accepted pilot')
        raw_rows = list(rows(files['raw']))
        if len(raw_rows) != complete['rows'] or {r['key'] for _,r,_ in raw_rows} != set(complete['completed_keys']):
            raise ValueError('Actual complete part row count or scientific keys differ')
        for line,row,line_sha in raw_rows:
            sample = samples.get(row['sample']['id'])
            if (sample is None or row['identity'] != sidecar['identity'] or row['key'] in seen
                    or row['key'] not in allowed or row['method'] not in plan['methods'] or row['marker'] not in plan['markers']
                    or row['method'] != complete['method'] or row['marker'] != complete['marker']
                    or row['kind'] != 'instruction_preserving' or row['reference_marker'] != row['marker']
                    or row['guided'] is not True or row['reference_guided'] is not False or row['replicate'] != 0):
                raise ValueError('Actual raw lies outside its completed original missing-key claim')
            task = audit_raw(row,sample,model)
            if row['key'] != task_id(model,task):
                raise ValueError('Actual task identity differs from the original entry')
            seen.add(row['key'])
            selected.append((row,dict(source_host=source['host'],source_original_root=source['source_root'],
                source_path=str(files['raw']),source_line=line,raw_line_sha256=line_sha,
                raw_source_sha256=source['raw_sha256'],source_identity=row['identity'],
                complete_source_path=str(files['complete']),complete_source_sha256=source['complete_sha256'],
                identity_source_path=str(files['identity']),identity_source_sha256=source['identity_sha256'],
                source_claim=source['claim_id'],source_scope='registered_dev_completed_part',expected_n=404,
                source_raw_complete=True,source_part_complete=True,
                model_checkpoint=spec['hf_model_id'],model_dtype=spec['dtype'],
                actual_admission_identity_sha256=source['claim_identity_sha256'],
                original_missing_keys_path=str(files['missing_keys']),original_missing_keys_sha256=source['missing_keys_sha256'])))
        sources.append(dict(model=model,method=complete['method'],marker=complete['marker'],rows=complete['rows'],
            expected_n=404,source_scope='registered_dev_completed_part',raw_complete=True,
            raw_path=str(files['raw']),raw_sha256=source['raw_sha256'],source_host=source['host'],
            source_original_root=source['source_root'],complete_path=str(files['complete']),complete_sha256=source['complete_sha256'],
            identity_path=str(files['identity']),identity_sha256=source['identity_sha256'],runner_sha256=plan['runner_sha256'],
            source_claim=source['claim_id'],original_missing_keys_sha256=source['missing_keys_sha256']))
    if len(selected) != manifest['rows'] or len(sources) != manifest['source_parts']:
        raise ValueError('Received completed-part source index differs from its actual contents')
    return selected,sources

def frozen_references(samples):
    core_path = ROOT / REFERENCE
    frame = pd.read_parquet(core_path)
    if (len(frame) != 24240 or frame.duplicated(['model','sample_id']).any()
            or not frame.independent_attempts.eq(10).all()
            or not ((frame.gold_rank > 1) & frame.independent_correct_attempts.eq(0)).eq(frame.uniform_reference).all()):
        raise ValueError('Original core Food references are not complete or use a different rule')
    refs = {}
    for r in frame[frame.sample_id.isin(samples)].to_dict('records'):
        refs[(r['model'],r['sample_id'])] = {**r, 'reference_source_path':str(core_path)}
    extra_dir = ROOT / EXTRA_REF
    receipt_path = extra_dir / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    ref_path = extra_dir / 'reference_G.jsonl'
    if not receipt['passed'] or receipt['reference_complete'] != 19392 or receipt['reference_pending'] != 0 or file_hash(ref_path) != receipt['outputs']['reference_G.jsonl']:
        raise ValueError('Original extra-model reference closure or source bytes differ')
    total = 0
    for line,r,line_sha in rows(ref_path):
        total += 1
        if (r['reference_complete'] is not True or r['attempt_count'] != 10
                or r['missing_replicates'] or r['reference_G'] != (r['gold_rank'] > 1 and r['correct_count'] == 0)):
            raise ValueError('A original extra reference differs from rank>1 and ten correct_count=0')
        if r['split'] == 'dev' and r['sample_id'] in samples:
            key = (r['model'],r['sample_id'])
            if key in refs or r['model'] not in EXTRA or r['target_class'] != samples[r['sample_id']]['class']:
                raise ValueError('A frozen extra reference sample or unique key differs')
            refs[key] = dict(model=r['model'],sample_id=r['sample_id'],uniform_reference=r['reference_G'],
                accepted_reference=r['historical_accepted_reference'],gold_rank=r['gold_rank'],
                independent_correct_attempts=r['correct_count'],independent_attempts=r['attempt_count'],
                uniform_source_line=line,reference_source_path=str(ref_path),reference_line_sha256=line_sha)
    if total != 19392 or len(refs) != 9*404:
        raise ValueError('The same registered dev404 input reference coverage is incomplete')
    provenance = [dict(path=str(core_path),sha256=file_hash(core_path),rows=24240),
        dict(path=str(ref_path),sha256=receipt['outputs']['reference_G.jsonl'],rows=19392,
             receipt_path=str(receipt_path),receipt_sha256=file_hash(receipt_path))]
    return refs, provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-manifest',required=True)
    parser.add_argument('--authority',required=True)
    parser.add_argument('--decision-file',action='append',default=[])
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only scorer requires explicitly empty CUDA_VISIBLE_DEVICES')
    started=time.perf_counter()
    output=within(ROOT,args.out);output.relative_to(ROOT/'outputs/paper_core_20261002_dev_viz')
    if output.exists():raise FileExistsError('New scoring results require an exclusive output directory')
    roster_path, selected_samples=roster('dev404');samples={s['id']:s for s in selected_samples}
    manifest_path=within(ROOT,args.producer_manifest)
    input_schema=json.loads(manifest_path.read_text())['schema']
    source_records,sources=(completed_dev_sources(manifest_path,samples,roster_path) if input_schema=='kdm_IP_dev_completed_parts_source_index_v1' else pilot_sources(manifest_path,samples,roster_path))
    references,reference_proofs=frozen_references(samples)
    canonical_sha=file_hash(Path(frozen.__file__))
    accepted_paths,accepted_receipts=native_decisions(NATIVE_RECEIPT,file_hash(ROOT/REFERENCE),canonical_sha)
    dev_path=ROOT/DEV_RECEIPT;dev=json.loads(dev_path.read_text())
    if not dev['passed'] or any(dev[k] for k in ['canonical_pending_rows','literal_pending_rows','abstain_pending_rows','pending_QA']) or dev['reference_sha256'] != file_hash(ROOT/REFERENCE) or dev['canonical_scorer_sha256'] != canonical_sha:
        raise ValueError('Reused original-core dev authority is not closed or differs')
    authority_path=within(ROOT,args.authority);authority=json.loads(authority_path.read_text())
    path_entries=[dict(path=str(p),sha256=file_hash(p)) for p in accepted_paths]+dev['decision_files']+authority['decision_files']
    path_entries += [dict(path=str(within(ROOT,p)),sha256=file_hash(within(ROOT,p))) for p in args.decision_file]
    decision_paths=[]
    for e in path_entries:
        p=within(ROOT,e['path'])
        if file_hash(p) != e['sha256']:raise ValueError('A closed annotation authority source changed')
        decision_paths.append(p)
    output.mkdir(parents=True,exist_ok=False)
    original=json.loads((ROOT/AUTHORITY).read_text())
    reviews,behavior,behavior_sources,_history,decisions=load_authority(output,{'census_final_labels':original['census_final_labels'],'historical_labels':[]},decision_paths)
    classes=json.loads((ROOT/'data/manifest.json').read_text())['canonical_classes'];patterns=frozen.compile_classes(classes)
    scored=[];pending={};memberships=[];cache={}
    for row,source in source_records:
        sample=row['sample'];answer=row['text'];qkey=frozen.qah(sample['question'],answer)
        if qkey not in cache:cache[qkey]=infer_qa(sample['question'],answer,patterns,reviews,behavior,decisions)
        inferred=cache[qkey]
        canonical,literal,reason=score_target(answer,sample['class'],inferred,patterns)
        decision=inferred['decision'];root_pending=bool(decision and (decision.get('needs_root') or decision.get('root_review_required')))
        if root_pending:canonical=literal=None
        abstain=inferred['abstain'] if not root_pending else None
        ref=references[(row['model'],sample['id'])]
        condition=dict(model=row['model'],dataset='food101',split='dev',**{f:row[f] for f in FIELDS})
        record=dict(**condition,**source,**{k:row[k] for k in ['prompt','reference_prompt','neutral_prompt','config','seed','tokens','terminated','selected_log_probabilities','first_probability']},
            condition_id=stable_hash(condition)[:16],key=row['key'],sample_id=sample['id'],qa_key=qkey,
            question=sample['question'],answer=answer,target_class=sample['class'],image_source=sample['image_path'],
            main_marker=row['marker'],correct_canonical=canonical,canonical_name_in_primary_score=canonical,
            correct_literal=literal,literal_extracted_name_score=literal,abstain=abstain,
            **{k:ref[k] for k in ['uniform_reference','accepted_reference','gold_rank','independent_correct_attempts','independent_attempts','reference_source_path','uniform_source_line']},
            score_reason=reason,behavior_source=inferred['behavior_source'],primary_extraction=inferred['parsed'],
            decision_source=decision,rule_executor=EXECUTOR)
        scored.append(record)
        needs=dict(canonical=canonical is None,literal=literal is None,abstain=abstain is None)
        if any(needs.values()):
            pending.setdefault(qkey,dict(qa_key=qkey,question=sample['question'],answer=answer,needs=needs))
            memberships.append(dict(qa_key=qkey,key=row['key'],model=row['model'],sample_id=sample['id'],target_class=sample['class'],**source))
    frame=pd.DataFrame(scored)
    metrics=[food_metrics(g,dict(zip(COND,k))) for k,g in frame.groupby(list(COND),dropna=False,observed=True)]
    if input_schema=='kdm_IP_dev_completed_parts_source_index_v1':
        for metric in metrics:
            if not metric['raw_complete']:metric['report_scope']='registered_dev_partial'
    save_rows(output/'score_rows.jsonl.gz',scored);save_rows(output/'pending_targetblind.jsonl',list(pending.values()));save_rows(output/'pending_private_memberships.jsonl',memberships)
    pd.DataFrame(metrics).to_csv(output/'metrics_actual_pilot.csv',index=False);pd.DataFrame(sources).to_csv(output/'sources.csv',index=False)
    pd.DataFrame([references[k] for k in sorted(references)]).to_csv(output/'frozen_dev404_references.csv',index=False)
    receipt=dict(schema='kdm_nine_model_IP_only_dev_cpu_scoring_v1',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        rows=len(scored),unique_keys=frame.key.nunique(),conditions=len(metrics),full404_complete_conditions=sum(m['primary_complete'] for m in metrics),
        pilot_partial_conditions=sum(m['report_scope']=='pilot_partial' for m in metrics),registered_dev_partial_conditions=sum(m['report_scope']=='registered_dev_partial' for m in metrics),canonical_pending_rows=int(frame.correct_canonical.isna().sum()),literal_pending_rows=int(frame.correct_literal.isna().sum()),
        abstain_pending_rows=int(frame.abstain.isna().sum()),pending_QA=len(pending),pending_memberships=len(memberships),
        label_completion_claimed=not memberships,selection_performed=False,selected_operating_point=False,expected_n_per_condition=404,
        producer_manifest=str(manifest_path),producer_manifest_sha256=file_hash(manifest_path),reference_sources=reference_proofs,
        reference_join_missing=0,accepted_native_authority_receipts=accepted_receipts,accepted_dev_authority_receipt=str(dev_path),
        accepted_dev_authority_receipt_sha256=file_hash(dev_path),decision_files=path_entries,authority_path=str(authority_path),authority_sha256=file_hash(authority_path),
        canonical_scorer_sha256=canonical_sha,reused_inference_source_sha256=file_hash(ROOT/'workflows/supplemental/remaining11/score.py'),
        source_audit_sha256=file_hash(ROOT/'workflows/paper_core/score_dev_viz.py'),runner_sha256=file_hash(Path(__file__)),actual_command=sys.argv,
        GPU_initialized=False,new_generation=0,new_API_calls=0,elapsed_s=time.perf_counter()-started,
        outputs={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    save_json(output/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['passed','rows','conditions','full404_complete_conditions','canonical_pending_rows','literal_pending_rows','abstain_pending_rows','pending_QA','pending_memberships']}))

if __name__=='__main__':main()
