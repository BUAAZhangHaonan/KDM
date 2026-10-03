"""Accept one sealed, exact-key core5 Viz shard and reuse existing exact-QA scoring."""
import argparse,hashlib,json,math,sys,tarfile
from pathlib import Path
ROOT=Path('/home/g203-4028/projects/knowledge-deficit-mitigation')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.main_results.score import qah
from workflows.paper_core.score_selected_viz import old_closed_authorities
from workflows.paper_core.score_dev_viz import viz_inferences,AUTHORITY
from kdm.scoring import vqa_score
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.io import stable_seed
B=ROOT/'outputs/paper_core_20261002_dev_viz/closeout_20261004/viz'
parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True);parser.add_argument('--piece',required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--source-host',required=True);parser.add_argument('--source-root',required=True)
args=parser.parse_args();model=args.model;piece=args.piece;dest=B/'received_generation'
assert model in ['llava16_mistral','minicpm26','gemma3_4b','qwen25vl','qwen35_4b'] and piece in [f'tail_s{i}of4' for i in range(4)]
with tarfile.open(args.archive) as tar:
    for member in tar:
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        assert member.name.startswith(model+'/'+piece+'/')
        p=dest/member.name;data=tar.extractfile(member).read()
        if p.exists():assert p.read_bytes()==data
        else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
part=dest/model/piece;receipt=json.loads((part/'complete.json').read_text());raw=part/'independent.jsonl'
assert receipt['generation_complete'] and hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['output_sha256']
expected={json.loads(s)['key'] for s in (B/'inputs'/model/(piece+'_keys.jsonl')).read_text().splitlines()}
seen=set();records=[];rawhash=receipt['output_sha256']
for n,line in enumerate(raw.read_text().splitlines(),1):
    row=json.loads(line);assert row['key'] not in seen;seen.add(row['key'])
    assert row['model']==model and row['status']=='ok' and row['sample']['dataset']=='vizwiz' and row['sample']['split']=='eval'
    assert row['seed']==stable_seed(row['sample']['id'],model,row['replicate'])
    assert row['config']['temperature']==1 and row['config']['top_p']==1 and row['config']['max_tokens']==32
    assert len(row['tokens'])==len(row['selected_log_probabilities']) and all(math.isfinite(x) for x in row['selected_log_probabilities'])
    source={'path':str(raw),'original_host':args.source_host,'original_path':args.source_root+'/outputs/paper_core_20261002_dev_viz/closeout_20261004/viz/generation/'+model+'/'+piece+'/independent.jsonl','source_sha256':rawhash,'source_line':n,'source_line_sha256':hashlib.sha256(line.encode()).hexdigest()}
    records.append((row,source))
assert seen==expected and len(records)==receipt['answers']==len(expected)
pilot={json.loads(s)['key'] for s in (B/'inputs'/model/'pilot8_keys.jsonl').read_text().splitlines()}
assert not seen.intersection(pilot)
decisions,authorities,oldproof=old_closed_authorities()
authorities.append(B/'annotation/authority/qa_authority.jsonl')
manifest=json.loads((ROOT/AUTHORITY).read_text())
qa,census=viz_inferences(records,manifest,decisions,authorities)
existing_pending={json.loads(s)['qa_key'] for s in (B/'annotation/pending_QA_with_sources.jsonl').read_text().splitlines()}
out=B/'scored_generation'/model/piece;out.mkdir(parents=True,exist_ok=True)
scored=[];pending={};covered=set()
normalizer=VQAEval(None,None)
normalize=lambda text:normalizer.processDigitArticle(normalizer.processPunctuation(text))
for row,source in records:
    key=qah(row['sample']['question'],row['text']);value=qa[key]
    abstain=value['abstain'];span=value['answer_text']
    decided=abstain is not None and span is not None
    quality=(0.0 if abstain or value['label']=='invalid' else vqa_score(span,row['sample']['official_answers'],normalize)) if decided else None
    if decided:covered.add(key)
    else:pending.setdefault(key,{'qa_key':key,'question':row['sample']['question'],'answer':row['text'],'already_in_extension4_queue':key in existing_pending,'memberships':[]})['memberships'].append({'key':row['key'],'model':model,'sample_id':row['sample']['id'],'replicate':row['replicate'],'source':source})
    scored.append({'key':row['key'],'model':model,'sample_id':row['sample']['id'],'replicate':row['replicate'],'qa_key':key,'abstain':abstain,'answer_text':span,'quality_score':quality,'official_raw_answer_score':vqa_score(row['text'],row['sample']['official_answers'],normalize),'official_answerable':row['sample']['annotated_answerable'],'inference':value,'source':source})
for name,values in [('score_rows.jsonl',scored),('pending_QA.jsonl',list(pending.values()))]:
    tmp=out/(name+'.tmp');tmp.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values));tmp.replace(out/name)
report={'generation_accepted':True,'model':model,'piece':piece,'raw_rows':len(records),'zero_duplicate_missing':True,'pilot_overlap':0,'closed_QA':len(covered),'pending_QA':len(pending),'pending_QA_already_queued':sum(v['already_in_extension4_queue'] for v in pending.values()),'new_pending_QA':sum(not v['already_in_extension4_queue'] for v in pending.values()),'closed_rows':sum(v['quality_score'] is not None for v in scored),'pending_rows':sum(v['quality_score'] is None for v in scored),'source_sha256':rawhash,'GPU_initialized':False,'all_shard_semantics_complete':not pending,'accepted_semantic_authorities':[str(p) for p in authorities],'census_source':census,'full_reply_fallback_used':False,'Food_reference_substituted':False,'official_continuous_scores_binarized':False}
(out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['census_source','accepted_semantic_authorities']}))
