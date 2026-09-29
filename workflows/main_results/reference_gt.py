"""Build the reference-GT tables from frozen compact source bundles."""
from __future__ import annotations
import collections,gzip,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'outputs/annotations/reference_gt';OUT=Path(os.environ.get('MAIN_RESULTS_REFERENCE_GT_OUT',str(DEFAULT_OUT)));RESP=ROOT/'data/responses';REF=ROOT/'data/reference_gt'
MODELS=['qwen25vl','minicpm26','qwen35_4b','gemma3_4b','llava16_mistral']
AUTH=ROOT/'data/reference_gt/final_target_blind_authority'
LEGACY_MODELS={'qwen25vl','minicpm26','llava16_mistral'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(p,items):
 with Path(p).open('w',encoding='utf8',newline='\n') as f:
  for d in items:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
def qah(q,a):return hashlib.sha256(json.dumps([q,a],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def load_score():
 p=ROOT/'workflows/main_results/score.py';spec=importlib.util.spec_from_file_location('main_results_score',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 scorer=load_score();OUT.mkdir(parents=True,exist_ok=True);source_manifest=json.loads((AUTH/'compact_source_manifest.json').read_text())
 classes=[]
 for d in rows(ROOT/'data/current/all.jsonl'):
  if d.get('class') and d['class'] not in classes:classes.append(d['class'])
 if len(classes)!=101:raise ValueError(f'canonical classes: {len(classes)}')
 patterns=scorer.compile_classes(classes)
 # Final build reads only the frozen reference authority copies.
 legacy_path=REF/'legacy_accepted_reference_gt.jsonl';conflict_path=REF/'legacy_accepted_conflicts.jsonl'
 legacy={(d['model'],d['sample_id']):d for d in rows(legacy_path)}
 conflicts=list(rows(conflict_path))
 if len(legacy)!=14544 or len(conflicts)!=157 or len({(d['model'],d['sample_id']) for d in conflicts})!=130:raise ValueError('copied legacy authority coverage mismatch')
 for (m,_),d in legacy.items():
  if m not in LEGACY_MODELS or type(d.get('gt')) is not bool:raise ValueError('bad legacy GT copy')
 # Fixed current registry and finite exact-QA/root target-blind decisions.
 reviewed_path=AUTH/'reviewed_answers.jsonl'
 if not reviewed_path.exists():raise FileNotFoundError('Stable target-blind reviewed-answer registry is absent')
 review_map={(d['question'],d['answer']):d.get('variants',[]) for d in rows(reviewed_path)}
 def keyed(p):return {(d['question'],d['answer']):d for d in rows(p)}
 closure_path=AUTH/'reference_gt_rule_closure_179.jsonl'
 closure_map=keyed(closure_path)
 a_overlap_path=AUTH/'reference_gt_a_overlap3.jsonl'
 a_overlap_map=keyed(a_overlap_path)
 parser_pass_path=AUTH/'reference_gt_correct_count_parser_pass.jsonl'
 parser_pass_map=keyed(parser_pass_path) if parser_pass_path.exists() else {}
 boundary_path=AUTH/'reference_gt_last_boundary_root.jsonl'
 boundary_map=keyed(boundary_path) if boundary_path.exists() else {}
 resolution_path=AUTH/'reference_gt_last_boundary_resolution.jsonl'
 resolution_map=keyed(resolution_path) if resolution_path.exists() else {}
 b_boundary_path=AUTH/'reference_gt_last_boundary_b.jsonl'
 b_boundary_map=keyed(b_boundary_path) if b_boundary_path.exists() else {}
 final24_path=AUTH/'reference_gt_final24_decisions.jsonl'
 final24_map=keyed(final24_path) if final24_path.exists() else {}
 root_files={
  'root_decisions':AUTH/'root_decisions.jsonl',
  'cross_review':AUTH/'cross_review_decisions.jsonl',
  'gap_review':AUTH/'independent_gap_review.jsonl',
  'rule_review':AUTH/'rule_parser_review.jsonl',
 }
 root_maps={k:keyed(p) for k,p in root_files.items()}
 def root_prediction(d,source):
  pc=d.get('predicted_class');span=d.get('primary_canonical_span') or d.get('source_span') or d.get('literal_primary_name') or d.get('primary_answer_span')
  decision=d.get('decision')
  if d.get('abstain') is True:st='abstain'
  elif d.get('multiple_primary') is True or decision in {'multiple_primary','multiple_explicit_primary','multiple_canonical_primary'}:st='multiple_primary'
  elif pc:st='canonical'
  elif decision in {'explicit_outside_101','outside_101','outside_vocabulary'} or (d.get('abstain') is False and d.get('multiple_primary') is False):st='outside_101'
  else:st='unresolved'
  abst=d.get('abstain') if type(d.get('abstain')) is bool else (False if st in {'canonical','outside_101','multiple_primary'} else None)
  return {'prediction_state':st,'canonical_candidates':[pc] if pc else [],'literal_primary_names':[span] if span else [],'abstain':abst,'multiple_primary':st=='multiple_primary','source':source}
 def decision_from_rule(d,source):
  status=d.get('status');cands=d.get('canonical_candidates') or [];name=d.get('primary_name')
  if status=='unique_class' and len(cands)==1:st='canonical'
  elif status=='multiple_classes' or len(cands)>1:st='multiple_primary'
  elif status=='explicit_outside_101':st='outside_101'
  else:st='unresolved'
  return {'prediction_state':st,'canonical_candidates':cands,'literal_primary_names':[name] if name else [],'abstain':None,'multiple_primary':st=='multiple_primary','source':source,'parser_rule':d.get('rule')}
 def predict(q,a):
  key=(q,a)
  # Root's final exact-QA authority wins; earlier exact authority sources are
  # fallback only, with source identity recorded in the output.
  if key in final24_map:return root_prediction(final24_map[key],'root_final24_correct_count_review')
  if key in root_maps['root_decisions']:return root_prediction(root_maps['root_decisions'][key],'root_decisions')
  if key in resolution_map:return root_prediction(resolution_map[key],'root_last_boundary_resolution')
  if key in boundary_map:return root_prediction(boundary_map[key],'root_last_boundary_45')
  if key in b_boundary_map:return root_prediction(b_boundary_map[key],'B_last_boundary_50')
  if key in parser_pass_map:
   d=parser_pass_map[key];status=d.get('status');cands=d.get('canonical_candidates') or []
   if status=='unique_class' and len(cands)==1:state='canonical'
   elif status=='multiple_classes' or len(cands)>1:state='multiple_primary'
   elif status=='explicit_outside_101':state='outside_101'
   else:state='unresolved'
   return {'prediction_state':state,'canonical_candidates':cands,'literal_primary_names':[d['primary_name']] if d.get('primary_name') else [],'abstain':False,'multiple_primary':state=='multiple_primary','source':'A_correct_count_parser_pass','parser_rule':d.get('rule')}
  if key in a_overlap_map:
   d=a_overlap_map[key]
   status=d.get('status');cands=d.get('canonical_class_candidates') or [];state='canonical' if status=='unique_canonical_primary' and len(cands)==1 else ('outside_101' if status=='outside_vocabulary' else ('multiple_primary' if status in {'multiple_primary','multiple_explicit_primary'} else 'unresolved'))
   return {'prediction_state':state,'canonical_candidates':cands,'literal_primary_names':d.get('primary_names',[]),'abstain':False,'multiple_primary':state=='multiple_primary','source':'A_target_blind_250_rule_decisions','parser_rule':d.get('rule')}
  if key in closure_map:
   d=closure_map[key]
   return {'prediction_state':d['prediction_state'],'canonical_candidates':d.get('canonical_candidates',[]),'literal_primary_names':d.get('literal_primary_names',[]),'abstain':False,'multiple_primary':d.get('multiple_primary',False),'source':'agent_target_blind_rule_closure','parser_rule':d.get('rule_name')}
  for auth in ['cross_review','gap_review','rule_review']:
   if key in root_maps[auth]:return root_prediction(root_maps[auth][key],auth)
  variants=review_map.get(key,[])
  if variants:
   states=[]
   for v in variants:
    z=scorer.extract(v,patterns)
    if z.get('abstain') is True:st='abstain'
    elif z.get('multi'):st='multiple_primary'
    elif len(z.get('classes',[]))==1:st='canonical'
    elif len(z.get('classes',[]))>1:st='multiple_primary'
    elif z.get('names') and z.get('ambiguous') is False:st='outside_101'
    else:st='unresolved'
    states.append((st,tuple(z.get('classes',[])),tuple(z.get('literal',[])),z.get('abstain'),z.get('ambiguous')))
   sig={(st,cs,ab,amb) for st,cs,names,ab,amb in states}
   names=sorted({n for _,_,ns,_,_ in states for n in ns},key=str.casefold);cands=sorted({c for _,cs,_,_,_ in states for c in cs})
   if len(sig)==1:
    st,cs,ab,amb=next(iter(sig));return {'prediction_state':st,'canonical_candidates':list(cs),'literal_primary_names':names,'abstain':ab,'multiple_primary':st=='multiple_primary','source':'reviewed_answers','variant_count':len(variants),'scope_ambiguous':amb}
   return {'prediction_state':'unresolved_variant_disagreement','canonical_candidates':cands,'literal_primary_names':names,'abstain':None,'multiple_primary':None,'source':'reviewed_answers_variant_disagreement','variant_count':len(variants)}
  z=scorer.parse_target_blind_primary(a,patterns)
  return decision_from_rule(z,'shared_target_blind_parser')

 # Load source-bound compact generations and candidate ranks. The original
 # experiment files are provenance only; this build does not rescan them.
 attempts=collections.defaultdict(list);meta={};ranks={};qa_set=set()
 for model in MODELS:
  sm=source_manifest['models'][model]
  response=RESP/'independent'/f'{model}.jsonl.gz';rank_path=RESP/'candidate'/f'{model}.jsonl.gz'
  if sha(response)!=sm['responses_sha256'] or sha(rank_path)!=sm['candidate_output_sha256']:raise ValueError(f'compact bundle SHA mismatch {model}')
  with gzip.open(response,'rt',encoding='utf8') as f:
   for d in map(json.loads,f):
    if d['model']!=model or d.get('source_identity')!=sm['identity'] or len(d.get('source_row_sha256',''))!=64:raise ValueError(f'bad response identity/hash {model}')
    key=(model,d['sample_id']);meta[key]=(d['split'],d['cluster'],d['target']);attempts[key].append(d);qa_set.add((d['question'],d['answer']))
  with gzip.open(rank_path,'rt',encoding='utf8') as f:
   for d in map(json.loads,f):
    if d['model']!=model or len(d.get('source_row_sha256',''))!=64:raise ValueError(f'bad candidate rank row {model}')
    key=(model,d['sample_id'])
    if key in ranks:raise ValueError('duplicate candidate row '+str(key))
    ranks[key]=d
 if len(attempts)!=24240 or len(ranks)!=24240 or sum(map(len,attempts.values()))!=242400:raise ValueError('five-model source coverage mismatch')
 # First create the complete target-blind QA decision table. No sample target,
 # rank, model, or sample identity enters the parser call.
 predictions={key:predict(*key) for key in sorted(qa_set)}
 pred_path=OUT/'target_blind_primary_predictions.jsonl'
 if pred_path.exists():raise FileExistsError(pred_path)
 dump(pred_path,({'qa_key':qah(q,a),'question':q,'answer':a,**predictions[(q,a)]} for q,a in sorted(qa_set)))
 uniform=[];review_queue={};correct_dist=collections.Counter();attempt_ranges=collections.Counter()
 for model in MODELS:
  for key,rank in sorted(ranks.items()):
   if key[0]!=model:continue
   _,sid=key;split,cluster,target=meta[key];at=attempts[key]
   if rank['target']!=target or rank['split']!=split:raise ValueError(f'rank/response mismatch {key}')
   if len(at)!=10 or {a['replicate'] for a in at}!=set(range(10)):raise ValueError(f'10 attempt denominator mismatch {key}')
   known_correct=0;pending=[]
   for item in at:
    if scorer.target_absent(item['answer'],target):continue
    p=predictions[(item['question'],item['answer'])]
    if p['prediction_state']=='canonical' and len(p['canonical_candidates'])==1:
     known_correct+=int(p['canonical_candidates'][0]==target)
    elif p['prediction_state'] in {'unresolved','unresolved_variant_disagreement'}:pending.append((item,p))
   exact_correct_count=None if pending else known_correct
   rank_one=rank['gold_rank']==1
   if rank_one:gt=False;reason='rank1_reference_candidate'
   elif known_correct>0:gt=False;reason='at_least_one_target_blind_primary_matches_candidate_target'
   elif pending:gt=None;reason='target_present_primary_scope_unresolved'
   else:gt=True;reason='all_10_resolved_without_candidate_target'
   row={'model':model,'sample_id':sid,'split':split,'cluster':cluster,'target':target,'gold_rank':rank['gold_rank'],'gt':gt,'gt_reason':reason,'attempts':10,'independent_correct_attempts_exact':exact_correct_count,'independent_correct_attempts_lower':known_correct,'independent_correct_attempts_upper':known_correct+len(pending),'unresolved_target_present_attempts':len(pending),'reference_authority':'uniform_current_primary_rule'}
   uniform.append(row)
   correct_dist[(('rank1' if rank_one else 'rank_gt1'),str(exact_correct_count) if exact_correct_count is not None else 'unresolved')]+=1
   attempt_ranges[(('rank1' if rank_one else 'rank_gt1'),known_correct,known_correct+len(pending))]+=1
   if gt is None:
    for item,p in pending:
     q,a=item['question'],item['answer'];keyhash=qah(q,a)
     review_queue[keyhash]={'qa_key':keyhash,'question':q,'answer':a,'prediction_state':p['prediction_state'],'canonical_candidates':p['canonical_candidates'],'literal_primary_names':p['literal_primary_names'],'source':p['source']}
 # Preserve original accepted GT for the first three models. Uniform GT remains
 # available for every model and the transition report exposes differences.
 main=[];migration=[]
 for u in uniform:
  k=(u['model'],u['sample_id']);old=legacy.get(k)
  if old is not None:
   main.append({**u,'gt':old['gt'],'gt_reason':'legacy_authority_preserved:'+old['gt_reason'],'reference_authority':'accepted_legacy_three_model'})
   ug=u['gt'];lg=old['gt']
   category='pending_uniform' if ug is None else ('unchanged' if ug==lg else ('legacy_true_uniform_false' if lg else 'legacy_false_uniform_true'))
   migration.append({'model':u['model'],'sample_id':u['sample_id'],'split':u['split'],'cluster':u['cluster'],'legacy_gt':lg,'uniform_gt':ug,'uniform_reason':u['gt_reason'],'transition':category,'changed':None if ug is None else ug!=lg})
  else:main.append(u)
 outputs={'uniform_reference_gt.jsonl':uniform,'reference_gt.jsonl':main,'uniform_gt_migration.jsonl':migration,'target_blind_gt_review_queue.jsonl':list(review_queue.values())}
 for name,data in outputs.items():
  p=OUT/name
  if p.exists():raise FileExistsError(p)
  dump(p,data)
 for model in MODELS:
  for split in ['dev','eval']:
   if sum(x['model']==model and x['split']==split for x in uniform)!=2424:raise ValueError(f'{model}/{split} denominator mismatch')
 count_csv=OUT/'independent_correctness_count_distribution.csv'
 if count_csv.exists():raise FileExistsError(count_csv)
 import csv
 with count_csv.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['rank_group','exact_correct_count','n_samples']);w.writeheader()
  for rank_group in ['rank1','rank_gt1']:
   for n in range(11):w.writerow({'rank_group':rank_group,'exact_correct_count':n,'n_samples':correct_dist.get((rank_group,str(n)),0)})
   w.writerow({'rank_group':rank_group,'exact_correct_count':'unresolved','n_samples':correct_dist.get((rank_group,'unresolved'),0)})
 bounds_csv=OUT/'independent_correctness_bounds.csv'
 if bounds_csv.exists():raise FileExistsError(bounds_csv)
 with bounds_csv.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['rank_group','known_correct_lower','possible_correct_upper','n_samples']);w.writeheader()
  for (rank_group,lo,hi),n in sorted(attempt_ranges.items()):w.writerow({'rank_group':rank_group,'known_correct_lower':lo,'possible_correct_upper':hi,'n_samples':n})
 outputs['independent_correctness_count_distribution.csv']=None
 outputs['independent_correctness_bounds.csv']=None
 src_manifest_path=AUTH/'compact_source_manifest.json'
 input_paths={'final_review_authority_manifest':AUTH/'authority_manifest.json','compact_source_manifest':src_manifest_path,'legacy_reference_gt':legacy_path,'legacy_conflicts':conflict_path,'reviewed_answers':reviewed_path,'canonical_classes':ROOT/'data/current/all.jsonl','scorer':ROOT/'workflows/main_results/score.py'}
 input_paths.update(root_files);input_paths['rule_closure_179']=closure_path;input_paths['A_overlap3_decisions']=a_overlap_path
 input_paths['A_correct_count_parser_pass']=parser_pass_path;input_paths['root_last_boundary_45']=boundary_path
 input_paths['root_last_boundary_resolution']=resolution_path;input_paths['B_last_boundary_50']=b_boundary_path
 input_paths['root_final24_correct_count_review']=final24_path
 manifest={'schema':'main_results_reference_gt_manifest','status':'generated_pending_root_review','rules':{'legacy_authority':'accepted copy values preserved exactly for Qwen25/MiniCPM/LLaVA','uniform_reference':'all five models under current unique primary-canonical-name rule; rank1 is GT false; rank>1 is GT false when any of 10 independently sampled primary predictions matches candidate target; GT true when all ten are resolved and none match; otherwise GT null','scoring_distribution':'exact primary candidate match only; target-absent response is certainly not a candidate match; unresolved target-present answer remains pending','multi_primary':'multiple/coequal classes cannot count as a single canonical correct answer'},'models':MODELS,'rows':len(main),'uniform_rows':len(uniform),'attempts':242400,'candidate_rank_rows':24240,'rows_per_model':4848,'dev_eval_denominator_per_model':2424,'unique_target_blind_QA':len(qa_set),'unresolved_QA_relevant_after_known_correct_exclusion':len(review_queue),'root_exact_QA_matched':len(set(root_maps['root_decisions']) & qa_set),'root_QA_loaded':len(root_maps['root_decisions']),'root_conflict_highest_priority':'root_decisions.jsonl','legacy_conflict_rows':len(conflicts),'legacy_conflict_unique_model_sample':130,'legacy_gt_transition_counts':dict(collections.Counter(x['transition'] for x in migration)),'uniform_gt_by_model_split':{f'{m}|{sp}|{g}':sum(x['model']==m and x['split']==sp and x['gt'] is g for x in uniform) for m in MODELS for sp in ['dev','eval'] for g in [True,False,None]},'legacy_gt_unknown_by_model_split':{f'{m}|{sp}':sum(x['model']==m and x['split']==sp and x['gt'] is None for x in main) for m in MODELS for sp in ['dev','eval']},'independent_correct_count_distribution_exact':{f'{r}|{n}':v for (r,n),v in sorted(correct_dist.items())},'independent_correct_count_lower_upper':{f'{r}|{lo}|{hi}':v for (r,lo,hi),v in sorted(attempt_ranges.items())},'inputs':{(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)):sha(p) for p in input_paths.values()},'outputs':{str((DEFAULT_OUT/n).relative_to(ROOT)):sha(OUT/n) for n in outputs if n not in {'independent_correctness_count_distribution.csv','independent_correctness_bounds.csv'}},'independent_correctness_count_distribution_sha256':sha(count_csv),'independent_correctness_bounds_sha256':sha(bounds_csv),'build_execution_output_dir':str(OUT),'legacy_copy_sha256':{'path':str(legacy_path.relative_to(ROOT)),'sha256':sha(legacy_path)},'legacy_conflict_copy_sha256':{'path':str(conflict_path.relative_to(ROOT)),'sha256':sha(conflict_path)},'notes':['Build reads only compact response/rank bundles and the copied accepted reference authority.','Target-blind predictions are frozen before sample targets and candidate ranks are joined.','Review queue contains only exact Q/A rows affecting a still-unresolved sample, and excludes target/model/sample identity.','This engineering artifact does not itself constitute global semantic acceptance or release.']}
 mp=OUT/'reference_gt_manifest.json'
 if mp.exists():raise FileExistsError(mp)
 mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'uniform_rows':len(uniform),'main_rows':len(main),'unresolved_samples':sum(x['gt'] is None for x in uniform),'unresolved_QA':len(review_queue),'legacy_transitions':manifest['legacy_gt_transition_counts'],'attempt_count_distribution':manifest['independent_correct_count_distribution_exact'],'GT_sha256':sha(OUT/'reference_gt.jsonl'),'uniform_sha256':sha(OUT/'uniform_reference_gt.jsonl'),'manifest_sha256':sha(mp)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
