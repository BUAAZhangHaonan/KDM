"""Score only sealed native Gemma SID parts using frozen canonical authorities."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import file_hash,read_jsonl,atomic_json,within,stable_hash,stable_seed
from workflows.main_results import score as frozen
from workflows.paper_core.score_native_baseline_parts import reference_map,condition_metrics
from workflows.paper_core.score_dev_viz import native_decisions,NATIVE_RECEIPT,AUTHORITY
from workflows.supplemental.remaining11.score import infer_qa,score_target,load_authority

def run(a):
    out=within(ROOT,a.output);out.mkdir(parents=True,exist_ok=True)
    refs,ref_sources=reference_map()
    authority=json.loads((ROOT/AUTHORITY).read_text())
    existing,authority_receipts=native_decisions(NATIVE_RECEIPT,ref_sources[0]['sha256'],file_hash(Path(frozen.__file__)))
    extra=[within(ROOT,p) for p in a.decision_file]
    reviews,behavior,history,_,decisions=load_authority(out,
        {'census_final_labels':authority['census_final_labels'],'historical_labels':[]},existing+extra)
    samples={s['id']:s for s in read_jsonl(ROOT/'data/current/all.jsonl') if s['dataset']=='food101' and s['split']=='eval'}
    patterns=frozen.compile_classes(sorted({s['class'] for s in samples.values()}))
    seen=set();cache={};pending={};scores=[];sources=[];identities=set()
    for relative in a.part:
        part=within(ROOT,relative);receipt=json.loads((part/'complete.json').read_text())
        identity=json.loads((part/'identity.json').read_text());raw=part/'new_predictions.jsonl'
        if not receipt['passed'] or file_hash(raw)!=receipt['raw_sha256']:
            raise ValueError('Raw source is not sealed and verified')
        values=list(read_jsonl(raw))
        if len(values)!=receipt['completed'] or {r['sample']['id'] for r in values}!=set(identity['sample_ids']):
            raise ValueError('Sealed part identity does not cover actual rows')
        sources.append({'path':str(raw),'sha256':receipt['raw_sha256'],'n':len(values),
                        'identity_sha256':file_hash(part/'identity.json'),'receipt_sha256':file_hash(part/'complete.json')})
        for line,row in enumerate(values,1):
            sample=row['sample'];sid=sample['id']
            if row['model']!='gemma3_4b' or row['method']!='sid' or sample!=samples[sid] or sid in seen:
                raise ValueError('Source model/input overlap or identity differs')
            if row['config']!=identity['config'] or row['sid_adapter_sha256']!=identity['sid_adapter_sha256']:
                raise ValueError('Actual decoder/adapter identity differs')
            if row['seed']!=stable_seed(sid,'gemma3_4b',0):raise ValueError('Seed differs')
            seen.add(sid);identities.add(stable_hash({'config':row['config'],'revision':row['implementation_revision']}))
            qkey=frozen.qah(sample['question'],row['text'])
            if qkey not in cache:cache[qkey]=infer_qa(sample['question'],row['text'],patterns,reviews,behavior,decisions)
            inferred=cache[qkey];canonical,literal,reason=score_target(row['text'],sample['class'],inferred,patterns)
            rec={'model':'gemma3_4b','dataset':'food101','split':'eval',
                 **{k:row[k] for k in ('method','kind','marker','reference_marker','guided','reference_guided','replicate','implementation_revision')},
                 'main_marker':row['marker'],'sample_id':sid,'key':row['key'],'target_class':sample['class'],
                 'question':sample['question'],'answer':row['text'],'qa_key':qkey,
                 'canonical_name_in_primary_score':canonical,'literal_extracted_name_score':literal,
                 'correct_canonical':canonical,'correct_literal':literal,'abstain':inferred['abstain'],
                 'uniform_reference':refs[('gemma3_4b',sid)],'reference_complete':True,
                 'score_reason':reason,'primary_extraction':inferred['parsed'],
                 'behavior_source':inferred['behavior_source'],'behavior_history_sources':history.get(qkey,[]),
                 'decision':inferred['decision'],'source_path':str(raw),'source_line':line,
                 'source_sha256':receipt['raw_sha256'],'source_identity_sha256':sources[-1]['identity_sha256'],
                 'tokens':row['tokens'],'terminated':row['terminated'],'config':row['config'],'seed':row['seed'],
                 'sid_adapter_sha256':row['sid_adapter_sha256'],
                 'annotation_model':inferred['annotation_model'],'annotation_effort':inferred['annotation_effort'],
                 'annotation_call_id':inferred['annotation_call_id']}
            scores.append(rec)
            if canonical is None or literal is None or inferred['abstain'] is None:
                p=pending.setdefault(qkey,{'qa_key':qkey,'question':sample['question'],'answer':row['text'],
                                          'reasons':[],'memberships':[]})
                p['reasons']=sorted(set(p['reasons']+[reason,inferred['behavior_source']]))
                p['memberships'].append({'sample_id':sid,'source_path':str(raw),'source_line':line})
    if len(identities)!=1:raise ValueError('Mixed conditions')
    if len(scores)==2424 and Counter(r['target_class'] for r in scores)!=Counter({c:24 for c in {s['class'] for s in samples.values()}}):
        raise ValueError('Class quotas differ')
    metrics=condition_metrics(scores)
    for name,values in [('score_rows.jsonl',scores),('pending_complete_QA.jsonl',list(pending.values()))]:
        temp=out/(name+'.new')
        with temp.open('w') as f:
            for row in values:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        temp.replace(out/name)
    compact=[]
    for row in scores:
        item=dict(row)
        for field in ('decision','primary_extraction','behavior_history_sources','config'):
            item[field+'_json']=json.dumps(item.pop(field),ensure_ascii=False,sort_keys=True)
        compact.append(item)
    pd.DataFrame(compact).to_parquet(out/'score_rows.parquet',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics_all.csv',index=False)
    receipt={'passed':True,'rows':len(scores),'expected':2424,'unique_keys':len(seen),
             'missing_sample_ids':sorted(set(samples)-seen),'pending_unique_QA':len(pending),
             'primary_complete':len(scores)==2424 and not pending,'sources':sources,
             'reference_sources':ref_sources,'authority_receipts':authority_receipts,
             'new_semantic_judgments':0,'extra_decisions':[{'path':str(p),'sha256':file_hash(p)} for p in extra],
             'canonical_scorer_sha256':file_hash(Path(frozen.__file__)),
             'outputs':{p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}}
    atomic_json(out/'receipt.json',receipt)
    print(json.dumps({'rows':len(scores),'pending_unique_QA':len(pending),'metrics':metrics}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',action='append',required=True)
    p.add_argument('--decision-file',action='append',default=[]);p.add_argument('--output',required=True)
    run(p.parse_args())
