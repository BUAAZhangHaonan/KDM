"""Score sealed native Viz512 additions with existing official/semantic functions."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse, json, math, os, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from workflows.main_results import score as frozen
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.score_dev_viz import viz_references,viz_inferences,viz_metrics,FIELDS,COND,AUTHORITY
from workflows.paper_core import score_selected_viz as accepted
from workflows.paper_core.score_native import save_json,save_rows

BASE=ROOT/'outputs/paper_core_20261002_dev_viz'
SOURCE_PACKAGE=BASE/'final_review_package_complete_accepted_sources_v2_20261003/package'
FROZEN_SCORES=[('support/vizwiz/core5_512/new_scores.parquet','0ba5aa7874b8a0ffcc9e0bd1603093b36c38463f1ffbc590ae597e8955baedf8',15872),
               ('vizwiz_final/new25_complete_records.parquet','d5e5822c7ff90e226bab91ce12f7a9e6c1c0716b9957ceaa6d9d863262dc919b',12800)]

def now():return datetime.now(timezone.utc).isoformat()

def sealed_part(part):
    """Read either a finished part or the original bytes sealed at handoff."""
    raw=part/'new_predictions.jsonl'; ip=part/'identity.json'
    identity=json.loads(ip.read_text()); rows=[json.loads(s) for s in raw.read_text().splitlines()]
    complete=part/'complete.json'
    if complete.exists():
        end=json.loads(complete.read_text())
        assert end['passed'] and len(rows)==end['completed']==end['expected']
        assert set(identity['sample_ids'])=={r['sample']['id'] for r in rows}
    else:
        complete=part.parent/'sealed_for_handoff.json'
        seal=json.loads(complete.read_text())
        assert seal['passed'] and seal['producer_exited'] and seal['zero_completed_remaining_intersection']
        matches=[x for x in seal['source_parts'] if within(ROOT,x['raw'])==raw]
        assert len(matches)==1
        source=matches[0]
        assert len(rows)==source['completed']>0
        assert source['original_identity_sha256']==file_hash(ip)
        assert {r['sample']['id'] for r in rows}<=set(identity['sample_ids'])
        end={'raw_sha256':source['raw_sha256']}
    assert file_hash(raw)==end['raw_sha256']
    return raw,ip,identity,rows,end,complete

def reuse(path):
    if path.exists():
        receipt=json.loads((path.parent/'receipt.json').read_text())
        assert receipt['passed'] and receipt['outputs'][path.name]==file_hash(path)
        return path
    decisions=defaultdict(list);sources=[]
    for name,sha,n in FROZEN_SCORES:
        source=SOURCE_PACKAGE/name
        assert file_hash(source)==sha
        frame=pd.read_parquet(source)
        if name=='vizwiz_final/new25_complete_records.parquet':
            unpacked=[json.loads(s) for s in frame.complete_original_record_json_line]
            assert [r['key'] for r in unpacked]==frame.key.tolist()
            frame=pd.DataFrame(unpacked)
        assert len(frame)==n and not frame[['abstain','quality_score']].isna().any().any()
        for line,row in enumerate(frame.to_dict('records'),1):
            key=frozen.qah(row['question'],row['answer'])
            assert key==row['qa_key']
            span=row['answer_text']
            if pd.isna(span):span=''
            if span and span not in row['answer']:raise ValueError('Frozen span is not an original substring')
            value=dict(qa_key=key,question=row['question'],answer=row['answer'],abstain=bool(row['abstain']),
                label=row['label'],answer_text=span,quality_span_resolved=True,behavior_resolved=True,
                source_score_path=str(source),source_score_sha256=sha,source_parquet_row_1based=line,
                new_scientific_judgment=False)
            decisions[key].append(value)
        sources.append(dict(path=str(source),sha256=sha,rows=n))
    closed=[];conflicts=[]
    for key,items in decisions.items():
        variants={(x['abstain'],x['answer_text'],x['label']) for x in items}
        if len(variants)==1:
            closed.append({**items[0],'reused_score_rows':len(items)})
        else:conflicts.append(dict(qa_key=key,existing_variants=items))
    path.parent.mkdir(parents=True,exist_ok=False)
    save_rows(path,closed);save_rows(path.parent/'existing_conflicts.jsonl',conflicts)
    save_json(path.parent/'receipt.json',dict(schema='kdm_finite_viz_exact_QA_authority_v1',passed=True,
        created_utc=now(),closed_QA=len(closed),existing_conflicts_excluded=len(conflicts),sources=sources,
        new_scientific_judgments=0,outputs={path.name:file_hash(path),
        'existing_conflicts.jsonl':file_hash(path.parent/'existing_conflicts.jsonl')}))
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',action='append',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--reuse-cache',required=True)
    p.add_argument('--decision-file',action='append',default=[])
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=within(ROOT,args.out)
    if out.exists():raise FileExistsError(out)
    rp,listed=roster('viz512');samples={s['id']:s for s in listed}
    refs,reference_path=viz_references(samples);reference_map={r.sample_id:r for r in refs.itertuples()}
    records=[];sources=[];keys=set()
    for item in args.part:
        part=within(ROOT,item)
        raw,identity_path,identity,rows,end,complete_path=sealed_part(part)
        identity_sha=file_hash(identity_path)
        for line,row in enumerate(rows,1):
            sample=samples[row['sample']['id']]
            assert row['sample']==sample
            assert row['method'] in {'sid','dola','deco'} and not row['guided'] and not row['reference_guided']
            assert row['replicate']==0 and row['marker']==row['reference_marker']=='NONE'
            assert row['model']==identity['model'] and row['config']==identity['config']
            assert row['config']['max_tokens']==32 and row['config']['temperature']==0
            assert row['prompt']==task_prompt(sample['question'],guided=False)
            assert row['seed']==stable_seed(sample['id'],row['model'],0)
            task={f:row[f] for f in FIELDS}|{'sample':sample}
            if 'implementation_revision' in row:task['implementation_revision']=row['implementation_revision']
            assert row['key']==task_id(row['model'],task) and row['key'] not in keys
            assert 0<len(row['tokens'])<=32 and len(row['tokens'])==len(row['selected_log_probabilities'])
            assert all(math.isfinite(v) for v in row['selected_log_probabilities'])
            keys.add(row['key'])
            spec=identity['runtime_spec']
            provenance=dict(source_path=str(raw),source_line=line,raw_source_sha256=end['raw_sha256'],
                source_identity=identity_sha,generation_identity=identity_sha,expected_n=512,
                source_raw_complete=True,model_checkpoint=spec['hf_model_id'],model_dtype=spec['dtype'],
                identity_source_path=str(identity_path),complete_source_path=str(complete_path),
                implementation_revision=row.get('implementation_revision',identity.get('revision','')))
            records.append((row,provenance))
        sources.append(dict(path=str(raw),sha256=end['raw_sha256'],identity_sha256=identity_sha,n=len(rows),
                            complete_path=str(complete_path),complete_sha256=file_hash(complete_path)))
    cache=reuse(within(ROOT,args.reuse_cache))
    old_decisions,old_viz,oldproof=accepted.old_closed_authorities()
    # Frozen complete56 exact-QA authority is sufficient; ambiguous old duplicate
    # variants stay excluded and return to the existing census/explicit path.
    dec=old_decisions+[within(ROOT,x) for x in args.decision_file]
    authority=json.loads((ROOT/AUTHORITY).read_text())
    inference,census=viz_inferences(records,authority,dec,[cache])
    accepted.EXECUTOR=dict(agent='/root',model=None,effort=None,call_id='',role='deterministic CPU scoring adapter')
    scored,pending,memberships=accepted.score_records(records,inference,reference_map,reference_path)
    frame=pd.DataFrame(scored)
    metrics=[viz_metrics(g,dict(zip(COND,k))) for k,g in frame.groupby(list(COND),dropna=False,observed=True)]
    for metric in metrics:
        metric['J']=(metric['answer_quality_sum']+metric['TP'])/metric['n'] if metric['primary_scoring_complete'] else None
        metric['J_reason']=None if metric['J'] is not None else 'unresolved_answer_span_or_abstention'
    out.mkdir(parents=True,exist_ok=False)
    save_rows(out/'score_rows.jsonl.gz',scored)
    save_rows(out/'pending_targetblind.jsonl',[{k:x[k] for k in ('qa_key','question','answer','needs')} for x in pending.values()])
    save_rows(out/'pending_memberships.jsonl',memberships)
    pd.DataFrame(metrics).to_csv(out/'metrics_actual.csv',index=False)
    refs.to_csv(out/'official_fixed512_references.csv',index=False)
    save_json(out/'receipt.json',dict(passed=True,created_utc=now(),rows=len(scored),conditions=len(metrics),
        complete_conditions=sum(x['primary_complete'] for x in metrics),pending_QA=len(pending),
        quality_pending_rows=int(frame.quality_score.isna().sum()),abstain_pending_rows=int(frame.abstain.isna().sum()),
        sources=sources,old_semantic_authority=oldproof,exact_reuse=str(cache),census=census,
        decision_files=[dict(path=str(x),sha256=file_hash(x)) for x in dec],
        roster_sha256=file_hash(rp),official_reference_sha256=file_hash(reference_path),
        official_unanswerable_n=166,official_answerable_n=346,old_scores_changed=False,GPU_initialized=False,
        outputs={x.name:file_hash(x) for x in out.iterdir() if x.is_file()}))
    print(json.dumps(dict(rows=len(scored),pending_QA=len(pending),complete_conditions=sum(x['primary_complete'] for x in metrics))))

if __name__=='__main__':main()
