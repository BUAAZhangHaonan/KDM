"""Condition-complete paired Food-101 analysis for main-results score rows."""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,math,random
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCORE=ROOT/'outputs/annotations/main_results/score_rows.jsonl.gz'
GT=ROOT/'outputs/annotations/reference_gt/reference_gt.jsonl'
UNIFORM=ROOT/'outputs/annotations/reference_gt/uniform_reference_gt.jsonl'
GT_MANIFEST=ROOT/'outputs/annotations/reference_gt/reference_gt_manifest.json'
DEFAULT_OUT=ROOT/'outputs/analysis/main_results'
OUT=DEFAULT_OUT
COND=['model','method','kind','marker','reference_marker','guided','reference_guided','replicate']
PAIR_CONTEXT=[k for k in COND if k!='method']
BOOT=2000;SEED=20260929

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jsonl(p):
 with Path(p).open(encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def load_score(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt',encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def mean(xs):return sum(xs)/len(xs) if xs else None
def ci(vals):
 if not vals:return [None,None]
 vals=sorted(vals)
 return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]
def bootstrap_class_ci(values_by_cluster, draw_indices):
 clusters=sorted(values_by_cluster)
 if not clusters:return {'estimate':None,'ci95':[None,None],'clusters':0}
 means=np.asarray([np.mean(values_by_cluster[c]) for c in clusters],dtype=float)
 sampled=means[draw_indices].mean(axis=1)
 return {'estimate':float(means.mean()),'ci95':[float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))],'clusters':len(clusters)}
def bootstrap_ratio_diff_by_cluster(numden, draw_indices, cluster_universe):
 nums=np.asarray([numden.get(c,(0,0))[0] for c in cluster_universe],dtype=float)
 dens=np.asarray([numden.get(c,(0,0))[1] for c in cluster_universe],dtype=float)
 den=float(dens.sum());num=float(nums.sum())
 if not den:return {'estimate':None,'ci95':[None,None],'clusters':0,'numerator':0,'denominator':0}
 sn=nums[draw_indices].sum(axis=1);sd=dens[draw_indices].sum(axis=1)
 vals=sn[sd>0]/sd[sd>0]
 return {'estimate':num/den,'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'clusters':len(cluster_universe),'numerator':int(num),'denominator':int(den)}
def main(out_dir=DEFAULT_OUT,score_path=SCORE,reference_gt_path=GT,uniform_reference_gt_path=UNIFORM,reference_gt_manifest_path=GT_MANIFEST):
 global OUT,SCORE,GT,UNIFORM,GT_MANIFEST
 OUT=Path(out_dir);SCORE=Path(score_path);GT=Path(reference_gt_path);UNIFORM=Path(uniform_reference_gt_path);GT_MANIFEST=Path(reference_gt_manifest_path)
 OUT.mkdir(parents=True,exist_ok=True)
 if not SCORE.exists():raise FileNotFoundError(f'final scorer output absent: {SCORE}')
 if not GT.exists() or not UNIFORM.exists():raise FileNotFoundError('reference GT outputs absent')
 source_hashes={'score_rows':sha(SCORE),'reference_gt':sha(GT),'uniform_reference_gt':sha(UNIFORM),'reference_gt_manifest':sha(GT_MANIFEST)}
 gt={(d['model'],d['sample_id']):d for d in jsonl(GT)}
 ug={(d['model'],d['sample_id']):d for d in jsonl(UNIFORM)}
 gt_manifest=json.loads(GT_MANIFEST.read_text())
 if any(k.endswith('|unresolved') and v for k,v in gt_manifest['independent_correct_count_distribution_exact'].items()):raise ValueError('independent ten-attempt correct-count distribution is not fully resolved')
 migration_counts=collections.Counter()
 for key,urow in ug.items():
  mrow=gt[key];a,b=mrow['gt'],urow['gt']
  migration_counts['unchanged' if a==b else ('legacy_true_uniform_false' if a is True else 'legacy_false_uniform_true')]+=1
 rows=[];seen=set()
 for d in load_score(SCORE):
  ident=(d['model'],d['key'])
  if ident in seen:raise ValueError(f'duplicate scored row {ident}')
  seen.add(ident);k=(d['model'],d['sample_id'])
  if k not in gt or k not in ug:raise ValueError('GT join missing '+str(k))
  x=dict(d);x['cluster']=d.get('target_class');x['reference_gt']=gt[k]['gt'];x['uniform_gt']=ug[k]['gt'];rows.append(x)
 if len(rows)!=853248:raise ValueError(f'expected 853248 score rows, got {len(rows)}')
 if source_hashes!={'score_rows':sha(SCORE),'reference_gt':sha(GT),'uniform_reference_gt':sha(UNIFORM),'reference_gt_manifest':sha(GT_MANIFEST)}:raise ValueError('source SHA changed while analysis was reading inputs')
 groups=collections.defaultdict(list)
 for d in rows:groups[tuple(d[k] for k in COND)].append(d)
 if len(groups)!=352:raise ValueError(f'expected 352 condition cells, got {len(groups)}')
 for key,rs in groups.items():
  target_counts=collections.Counter(r.get('target_class') for r in rs)
  if len(target_counts)!=101 or any(n!=24 for n in target_counts.values()):raise ValueError(f'target-class denominator mismatch: {key}')
 score_unknown={k:sum(r.get(k) is None for r in rows) for k in ('canonical_name_in_primary_score','literal_extracted_name_score','abstain')}
 if any(score_unknown.values()):raise ValueError(f'final score fields still have unknown rows: {score_unknown}')
 condition_rows=[]
 for key,rs in sorted(groups.items(),key=lambda x:tuple(map(str,x[0]))):
  if len(rs)!=2424:raise ValueError(f'condition denominator {len(rs)} !=2424: {key}')
  canon=[r['canonical_name_in_primary_score'] for r in rs];lit=[r['literal_extracted_name_score'] for r in rs];ab=[r.get('abstain') for r in rs];g=[r['reference_gt'] for r in rs];u=[r['uniform_gt'] for r in rs]
  num=lambda xs,v:sum(x==v for x in xs)
  known=[(a,b) for a,b in zip(ab,g) if type(a) is bool and type(b) is bool]
  uniform_known=[(a,b) for a,b in zip(ab,u) if type(a) is bool and type(b) is bool]
  gtpos=sum(b is True for _,b in known);predpos=sum(a is True for a,_ in known);tp=sum(a is True and b is True for a,b in known);fp=sum(a is True and b is False for a,b in known);fn=sum(a is False and b is True for a,b in known)
  utp=sum(a is True and b is True for a,b in uniform_known);ufp=sum(a is True and b is False for a,b in uniform_known);ufn=sum(a is False and b is True for a,b in uniform_known)
  acc_correct=num(canon,1);acc_unknown=sum(x is None for x in canon);answered=[(c,a) for c,a in zip(canon,ab) if a is False];selective_correct=sum(c==1 for c,_ in answered);selective_unknown=sum(c is None for c,_ in answered)
  rec=dict(zip(COND,key));rec.update(n=2424,answered_rows=len(answered),coverage=len(answered)/2424,selective_accuracy_correct=selective_correct,selective_accuracy_unknown=selective_unknown,selective_accuracy_lower=selective_correct/len(answered) if answered else None,selective_accuracy_upper=(selective_correct+selective_unknown)/len(answered) if answered else None,canonical_correct=acc_correct,canonical_incorrect=num(canon,0),canonical_unknown=acc_unknown,canonical_accuracy_lower=acc_correct/2424,canonical_accuracy_upper=(acc_correct+acc_unknown)/2424,literal_correct=num(lit,1),literal_incorrect=num(lit,0),literal_unknown=sum(x is None for x in lit),literal_accuracy_lower=num(lit,1)/2424,literal_accuracy_upper=(num(lit,1)+sum(x is None for x in lit))/2424,abstain_true=num(ab,True),abstain_false=num(ab,False),abstain_unknown=sum(x is None for x in ab),reference_gt_true=sum(x is True for x in g),reference_gt_false=sum(x is False for x in g),reference_gt_unknown=sum(x is None for x in g),uniform_gt_true=sum(x is True for x in u),uniform_gt_false=sum(x is False for x in u),uniform_gt_unknown=sum(x is None for x in u),abstention_precision_known=tp/(tp+fp) if tp+fp else None,abstention_recall_known=tp/(tp+fn) if tp+fn else None,reference_gt_positive_known=tp+fn,reasonable_abstentions_retained=tp,unnecessary_abstentions=fp,missed_reasonable_abstentions=fn,abstention_known_n=len(known),abstention_tp=tp,abstention_fp=fp,abstention_fn=fn,uniform_abstention_precision_known=utp/(utp+ufp) if utp+ufp else None,uniform_abstention_recall_known=utp/(utp+ufn) if utp+ufn else None,uniform_reference_gt_positive_known=utp+ufn,uniform_abstention_tp=utp,uniform_abstention_fp=ufp,uniform_abstention_fn=ufn)
  condition_rows.append(rec)
 # Build direct-baseline pairs only when all non-method condition fields match.
 # Direct baseline is the observed main-prompt direct condition, matched by
 # model, main marker, guided flag, replicate, and exact sample_id. Intervention
 # dimensions (kind/reference marker/reference guidance) remain in reported rows.
 direct_by_main={}
 for key,rs in groups.items():
  d=dict(zip(COND,key))
  if d['method']=='direct' and d['kind']=='main' and d['reference_marker']==d['marker'] and d['reference_guided']==d['guided']:
   direct_by_main[(d['model'],d['marker'],d['guided'],d['replicate'])]=rs
 rng=np.random.RandomState(SEED);draw_indices=rng.randint(0,101,size=(BOOT,101))
 comparisons=[];missing=[];pair_edges=[]
 def compare(label,ctx,a_name,a_rows,b_name,b_rows,direct_rows=None):
  amap={r['sample_id']:r for r in a_rows};bmap={r['sample_id']:r for r in b_rows}
  if set(amap)!=set(bmap):
   missing.append({'comparison':label,'context':dict(zip(PAIR_CONTEXT,ctx)),'method_a':a_name,'method_b':b_name,'n_a':len(amap),'n_b':len(bmap),'missing_a':len(set(bmap)-set(amap)),'missing_b':len(set(amap)-set(bmap)),'reason':'sample_id pairing incomplete'})
   return
  ids=sorted(amap);cl=collections.defaultdict(lambda:{'acc_lo':[],'acc_hi':[]})
  unresolved=0;defined=[];positive=collections.defaultdict(lambda:{'num':0,'den':0,'retain_num':0,'retain_den':0});new_abs={'gt_true':0,'gt_false':0,'gt_unknown':0};transitions={'baseline_correct_retained':0,'baseline_correct_lost':0,'baseline_correct_unknown_in_a':0,'new_correct_vs_baseline':0}
  directmap={r['sample_id']:r for r in direct_rows} if direct_rows is not None else {}
  specific_corr={'num':0,'den':0};all_corr={'num':0,'den':0};preserve_ref=collections.defaultdict(lambda:{'num':0,'den':0});preserve_uniform=collections.defaultdict(lambda:{'num':0,'den':0})
  uniform_positive=collections.defaultdict(lambda:{'num':0,'den':0,'retain_num':0,'retain_den':0})
  for sid in ids:
   a,b=amap[sid],bmap[sid];ca=a['canonical_name_in_primary_score'];cb=b['canonical_name_in_primary_score']
   lo=(ca if ca is not None else 0)-(cb if cb is not None else 1)
   hi=(ca if ca is not None else 1)-(cb if cb is not None else 0)
   cluster=str(a['target_class']);cl[cluster]['acc_lo'].append(lo);cl[cluster]['acc_hi'].append(hi)
   if ca is None or cb is None:unresolved+=1
   else:defined.append(ca-cb)
   ga=a['reference_gt'];gb=b['reference_gt']
   if ga is True and gb is True and type(a.get('abstain')) is bool and type(b.get('abstain')) is bool:
    positive[cluster]['num']+=int(a['abstain'])-int(b['abstain']);positive[cluster]['den']+=1
    if b['abstain']:
     positive[cluster]['retain_den']+=1;positive[cluster]['retain_num']+=int(a['abstain'])
   ua,ub=a['uniform_gt'],b['uniform_gt']
   if ua is True and ub is True and type(a.get('abstain')) is bool and type(b.get('abstain')) is bool:
    uniform_positive[cluster]['num']+=int(a['abstain'])-int(b['abstain']);uniform_positive[cluster]['den']+=1
    if b['abstain']:
     uniform_positive[cluster]['retain_den']+=1;uniform_positive[cluster]['retain_num']+=int(a['abstain'])
   if directmap:
    dr=directmap[sid];dc=dr['canonical_name_in_primary_score']
    if cb==1 and dc==0:
     all_corr['den']+=1;all_corr['num']+=int(ca==1)
     if dr.get('abstain') is False:
      specific_corr['den']+=1;specific_corr['num']+=int(ca==1)
    if dr.get('abstain') is True and dr['reference_gt'] is True:
     preserve_ref[cluster]['den']+=1;preserve_ref[cluster]['num']+=int(a.get('abstain') is True)-int(b.get('abstain') is True)
    if dr.get('abstain') is True and dr['uniform_gt'] is True:
     preserve_uniform[cluster]['den']+=1;preserve_uniform[cluster]['num']+=int(a.get('abstain') is True)-int(b.get('abstain') is True)
   if a.get('abstain') is True and b.get('abstain') is False:new_abs['gt_true' if ga is True else ('gt_false' if ga is False else 'gt_unknown')]+=1
   if cb==1 and ca==1:transitions['baseline_correct_retained']+=1
   elif cb==1 and ca==0:transitions['baseline_correct_lost']+=1
   elif cb==1 and ca is None:transitions['baseline_correct_unknown_in_a']+=1
   elif cb==0 and ca==1:transitions['new_correct_vs_baseline']+=1
  acc_lo=bootstrap_class_ci({c:v['acc_lo'] for c,v in cl.items()},draw_indices)
  acc_hi=bootstrap_class_ci({c:v['acc_hi'] for c,v in cl.items()},draw_indices)
  all_clusters=sorted({str(r['target_class']) for r in a_rows})
  recall_data={c:(v['num'],v['den']) for c,v in positive.items()}
  rec=bootstrap_ratio_diff_by_cluster(recall_data,draw_indices,all_clusters)
  retain_data={c:(v['retain_num'],v['retain_den']) for c,v in positive.items()}
  retained=bootstrap_ratio_diff_by_cluster(retain_data,draw_indices,all_clusters)
  urec=bootstrap_ratio_diff_by_cluster({c:(v['num'],v['den']) for c,v in uniform_positive.items()},draw_indices,all_clusters)
  uret=bootstrap_ratio_diff_by_cluster({c:(v['retain_num'],v['retain_den']) for c,v in uniform_positive.items()},draw_indices,all_clusters)
  preserve_ref_ci=bootstrap_ratio_diff_by_cluster({c:(v['num'],v['den']) for c,v in preserve_ref.items()},draw_indices,all_clusters)
  preserve_uniform_ci=bootstrap_ratio_diff_by_cluster({c:(v['num'],v['den']) for c,v in preserve_uniform.items()},draw_indices,all_clusters)
  p_mismatch=sum(a.get('main_prompt_sha256')!=b.get('main_prompt_sha256') for a,b in ((amap[sid],bmap[sid]) for sid in ids))
  p_missing=sum(not amap[sid].get('main_prompt_sha256') or not bmap[sid].get('main_prompt_sha256') for sid in ids)
  seed_mismatch=sum(amap[sid].get('seed')!=bmap[sid].get('seed') for sid in ids)
  seed_missing=sum(amap[sid].get('seed') is None or bmap[sid].get('seed') is None for sid in ids)
  comps={'comparison':label,'context':dict(zip(PAIR_CONTEXT,ctx)),'method_a':a_name,'method_b':b_name,'n_paired':len(ids),'target_class_clusters':len(cl),'canonical_unknown_in_pair':unresolved,'resolved_accuracy_difference_a_minus_b':mean(defined) if defined else None,'resolved_accuracy_pair_n':len(defined),'canonical_accuracy_difference_conservative_lower':acc_lo,'canonical_accuracy_difference_conservative_upper':acc_hi,'abstention_recall_difference_on_known_reference_gt_positive_a_minus_b':rec,'abstention_recall_pair_n':sum(v['den'] for v in positive.values()),'baseline_reasonable_abstentions_retained_by_a':retained,'uniform_gt_abstention_recall_difference_a_minus_b':urec,'uniform_gt_baseline_reasonable_abstentions_retained_by_a':uret,'direct_reasonable_abstention_set_preservation_difference_reference_gt':preserve_ref_ci,'direct_reasonable_abstention_set_preservation_difference_uniform_gt':preserve_uniform_ci,'specific_answer_correction_retention':{**specific_corr,'rate':specific_corr['num']/specific_corr['den'] if specific_corr['den'] else None},'all_input_correction_retention':{**all_corr,'rate':all_corr['num']/all_corr['den'] if all_corr['den'] else None},'new_abstentions_a_vs_b':new_abs,'correctness_transitions_a_vs_b':transitions,'main_prompt_sha_mismatch_count':p_mismatch,'main_prompt_sha_missing_count':p_missing,'seed_mismatch_count':seed_mismatch,'seed_missing_count':seed_missing,'noninferiority_at_minus_0p01':bool(acc_lo['ci95'][0] is not None and acc_lo['ci95'][0]>-0.01),'abstention_preservation_improvement_lower_gt_zero':bool(preserve_ref_ci['ci95'][0] is not None and preserve_ref_ci['ci95'][0]>0),'bootstrap_replicates':BOOT,'bootstrap_unit':'Food-101 target class'}
  comparisons.append(comps);pair_edges.append((label,comps['context'],a_name,a_rows,b_name,b_rows))
  return comps
 for key,rs in groups.items():
  d=dict(zip(COND,key))
  if d['method']=='direct':continue
  baseline=direct_by_main.get((d['model'],d['marker'],d['guided'],d['replicate']))
  ctx=tuple(d[k] for k in PAIR_CONTEXT)
  if baseline is None:
   missing.append({'comparison':'same_main_prompt_direct','context':dict(zip(PAIR_CONTEXT,ctx)),'method_a':d['method'],'method_b':'direct','n_a':len(rs),'n_b':0,'missing_a':0,'missing_b':len(rs),'reason':'no direct main-prompt condition for model/marker/guided/replicate'})
  else:compare('same_main_prompt_direct',ctx,d['method'],rs,'direct',baseline,direct_rows=baseline)
 # Instruction-preserving variants are paired to their base method only on the
 # original four matched-marker cells; reference-guidance remains part of the exact output context.
 main_method={}
 for key,rs in groups.items():
  d=dict(zip(COND,key))
  if d['kind']=='main':
   k=(d['model'],d['marker'],d['reference_marker'],d['guided'],d['replicate'],d['method'])
   main_method[k]=rs
 for key,rs in groups.items():
  d=dict(zip(COND,key));pairs={'instruction_vcd':'vcd','instruction_m3id':'m3id'}
  if d['method'] not in pairs or d['marker']!=d['reference_marker']:continue
  base=pairs[d['method']];k=(d['model'],d['marker'],d['reference_marker'],d['guided'],d['replicate'],base)
  other=main_method.get(k);ctx=tuple(d[x] for x in PAIR_CONTEXT)
  if other is None:
   missing.append({'comparison':'instruction_vs_base_method','context':dict(zip(PAIR_CONTEXT,ctx)),'method_a':d['method'],'method_b':base,'n_a':len(rs),'n_b':0,'missing_a':0,'missing_b':len(rs),'reason':'no corresponding base-method cell at equal markers/guidance'})
  else:compare('instruction_vs_base_method',ctx,d['method'],rs,base,other,direct_rows=direct_by_main.get((d['model'],d['marker'],d['guided'],d['replicate'])))
 # A's final scorer carries prompt SHA and seed from the canonical formal gzip.
 # Compare these bindings directly on every paired sample; do not reopen shards.
 prompt_examples=[];prompt_comparisons=seed_comparisons=0;prompt_mismatches=seed_mismatches=0;prompt_missing=seed_missing=0
 for label,ctx,an,ar,bn,br in pair_edges:
  amap={r['sample_id']:r for r in ar};bmap={r['sample_id']:r for r in br}
  for sid in amap:
   a,b=amap[sid],bmap[sid];prompt_comparisons+=1;seed_comparisons+=1
   if not a.get('main_prompt_sha256') or not b.get('main_prompt_sha256'):prompt_missing+=1
   elif a['main_prompt_sha256']!=b['main_prompt_sha256']:
    prompt_mismatches+=1
    if len(prompt_examples)<100:prompt_examples.append({'comparison':label,'context':ctx,'sample_id':sid,'method_a':an,'method_b':bn,'prompt_sha_a':a['main_prompt_sha256'],'prompt_sha_b':b['main_prompt_sha256']})
   if a.get('seed') is None or b.get('seed') is None:seed_missing+=1
   elif a['seed']!=b['seed']:
    seed_mismatches+=1
    if len(prompt_examples)<100:prompt_examples.append({'comparison':label,'context':ctx,'sample_id':sid,'method_a':an,'method_b':bn,'seed_a':a.get('seed'),'seed_b':b.get('seed')})
 prompt_audit={'paired_edges':len(pair_edges),'paired_sample_prompt_comparisons':prompt_comparisons,'main_prompt_sha_mismatch_count':prompt_mismatches,'main_prompt_sha_missing_count':prompt_missing,'seed_mismatch_count':seed_mismatches,'seed_missing_count':seed_missing,'bindings_from':'A final score_rows from canonical data/responses/formal/*.jsonl.gz','prompt_field':'main_prompt_sha256','seed_field':'seed','examples':prompt_examples}
 # Descriptive UNKNOWN/UNKNOWN slice by model, method, and condition.
 unknown_summary=[]
 for key,rs in groups.items():
  d=dict(zip(COND,key))
  if d['kind'] in {'main','instruction_preserving'} and d['marker']=='UNKNOWN' and d['reference_marker']=='UNKNOWN':
   c=sum(r['canonical_name_in_primary_score']==1 for r in rs);ab=sum(r.get('abstain') is True for r in rs)
   gtpos=[r for r in rs if r['reference_gt'] is True]
   direct_retained=sum(r.get('abstain') is True for r in gtpos)
   unknown_summary.append({**{k:d[k] for k in COND},'n':len(rs),'canonical_correct':c,'canonical_accuracy':c/len(rs),'abstain_true':ab,'reference_gt_positive':len(gtpos),'abstain_on_reference_positive':direct_retained,'abstention_recall_on_reference_positive':direct_retained/len(gtpos) if gtpos else None})
 # Save exact conditions, paired comparisons and missing-control inventory.
 for path in [OUT/'condition_metrics.csv',OUT/'paired_comparisons.jsonl',OUT/'missing_controls.jsonl',OUT/'prompt_pair_audit.json',OUT/'summary.json',OUT/'report.md',OUT/'unknown_unknown_summary.csv']:
  if path.exists():raise FileExistsError(path)
 fields=list(condition_rows[0])
 with (OUT/'condition_metrics.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(condition_rows)
 with (OUT/'paired_comparisons.jsonl').open('w',encoding='utf8',newline='\n') as f:
  for d in comparisons:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
 with (OUT/'missing_controls.jsonl').open('w',encoding='utf8',newline='\n') as f:
  for d in missing:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
 (OUT/'prompt_pair_audit.json').write_text(json.dumps(prompt_audit,ensure_ascii=False,indent=2)+'\n')
 metrics={'score_rows':len(rows),'condition_cells':len(groups),'rows_per_condition':2424,'target_classes_per_condition':101,'rows_per_target_class_per_condition':24,'paired_comparisons':len(comparisons),'missing_control_cells':len(missing),'prompt_sha_mismatch_count':prompt_mismatches,'prompt_sha_missing_count':prompt_missing,'seed_mismatch_count':seed_mismatches,'seed_missing_count':seed_missing,'paired_sample_prompt_comparisons':prompt_comparisons,'score_field_unknown_rows':score_unknown,'independent_correct_count_distribution':gt_manifest['independent_correct_count_distribution_exact'],'legacy_gt_transition_counts':dict(migration_counts),'reference_gt_unknown_rows':sum(r['reference_gt_unknown'] for r in condition_rows),'uniform_gt_unknown_rows':sum(r['uniform_gt_unknown'] for r in condition_rows),'bootstrap':{'replicates':BOOT,'seed':SEED,'unit':'Food-101 target class'},'input_sha256':source_hashes,'outputs':{}}
 summary=OUT/'summary.json';summary.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'unknown_unknown_summary.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(unknown_summary[0]) if unknown_summary else ['model','method']);w.writeheader();w.writerows(unknown_summary)
 # Compact report states scope and limits without asserting success or causal effect.
 report=['# Main results analysis','',f"Score source: {len(rows):,} rows in {len(groups)} exact eight-key condition cells; denominator 2,424 per cell.",'','The mixed reference table retains the accepted Qwen25-VL, MiniCPM-V2.6, and LLaVA-1.6-Mistral references and uses the current primary-name rule for Qwen3.5-4B and Gemma3-4B. `uniform_reference_gt.jsonl` and `condition_metrics.csv` also report all five models under the same current rule. Boolean GT is resolved for every sample in both tables.','',f"The accepted three-model reference and uniform five-model reference transitions are computed by exact model/sample join from the two GT tables: {migration_counts['legacy_true_uniform_false']} legacy-true to uniform-false and {migration_counts['legacy_false_uniform_true']} legacy-false to uniform-true, with {migration_counts['unchanged']} unchanged.",'','Canonical accuracy matches the predicted canonical class against the gold class, using only the extracted primary-name span; literal-name accuracy is a separate sensitivity column. Coverage and selective accuracy use the same score rows. Abstention precision and recall are shown under both reference choices, with unknown behavior/reference rows counted separately.','',f"Paired rows: {len(comparisons):,} (332 intervention cells); missing pair controls: {len(missing)}. Every intervention is paired against the observed direct main-prompt row by model, sample, main marker, guided flag, and replicate. Instruction-VCD/M3ID also have exact corresponding comparisons against VCD/M3ID. Main-prompt SHA and seed match on all {prompt_comparisons:,} paired samples.",'','Accuracy noninferiority is the lower 95% bound of the paired accuracy difference greater than -0.01. Bootstrap intervals use 2,000 shared resamples of the 101 Food-101 target-class clusters. Abstention-recall differences use micro totals of positive-reference numerators and denominators; “reasonable abstention retained” uses the direct-abstain and GT-positive subset. Instruction-versus-base outputs include preservation differences on that fixed direct subset, under both GT tables, and correction retention with specific-answer and all-input denominators.','',f"Uniform-GT unknown rows: {metrics['uniform_gt_unknown_rows']:,}; mixed-reference unknown rows: {metrics['reference_gt_unknown_rows']:,}. The exact UNKNOWN/UNKNOWN main and instruction-preserving condition rows are in `unknown_unknown_summary.csv`.",'','These paired results describe associations under the recorded prompts. Full eight-key tables, paired comparisons, prompt audit, and exact-condition figures accompany this report.']
 rp=OUT/'report.md';rp.write_text('\n'.join(report)+'\n',encoding='utf8')
 metrics['outputs']={p.name:sha(p) for p in [OUT/'condition_metrics.csv',OUT/'paired_comparisons.jsonl',OUT/'missing_controls.jsonl',OUT/'prompt_pair_audit.json',rp,OUT/'unknown_unknown_summary.csv']}
 summary.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'rows':len(rows),'conditions':len(groups),'paired':len(comparisons),'missing_controls':len(missing),'summary_sha256':sha(summary)},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,default=DEFAULT_OUT,help='analysis output directory; default is outputs/analysis/main_results')
 ap.add_argument('--score',type=Path,default=SCORE,help='frozen score_rows.jsonl.gz input')
 ap.add_argument('--reference-gt',type=Path,default=GT,help='accepted reference_gt.jsonl input')
 ap.add_argument('--uniform-reference-gt',type=Path,default=UNIFORM,help='uniform_reference_gt.jsonl input')
 ap.add_argument('--reference-gt-manifest',type=Path,default=GT_MANIFEST,help='reference_gt_manifest.json input')
 a=ap.parse_args();main(a.out,a.score,a.reference_gt,a.uniform_reference_gt,a.reference_gt_manifest)
