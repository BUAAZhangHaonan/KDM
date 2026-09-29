"""Full rule-based main score. Target-blind primary-name inference; raw data stays immutable."""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/annotations/main_results'
COND=['model','method','kind','marker','reference_marker','guided','reference_guided','replicate']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def qah(q,a):return hashlib.sha256(json.dumps([q,a],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def fmt(s):
 s=str(s).strip()
 for _ in range(4):
  old=s;s=s.rstrip(' \t\r\n.,;:!?')
  for l,r in [('**','**'),('__','__'),('*','*'),('`','`'),('"','"')]:
   if s.startswith(l) and s.endswith(r) and len(s)>len(l)+len(r):s=s[len(l):-len(r)].strip();break
  if old==s:break
 return s.casefold()
def role(r):
 r=str(r or '').casefold().strip().replace('-','_').replace(' ','_')
 main={'main','main_dish','main_food','main_or_explicit_competing_candidate','primary','primary_dish','dish','food','food_item','main_category','main_candidate','main_dish_or_base','base_food','explicit_food_head','dish_name','named_meal_category','meal_or_main_description','main_dish_with_container','main_dish_with_serving_form','generic_food_category','generic_dessert_category','generic_category','broader_category','ambiguous_food_name','repeated_answer','specified_item','literal_heading','composite_food_phrase','meal','qualified_restaurant_identification','unclear_fragment','unfinished_name_fragment','food_fragment'}
 if r in main or r.startswith('main_'):return 'main'
 if r in {'coequal','co_equal','coequal_dish','coequal_main','food_set','meal_or_food_set'}:return 'coequal'
 if r=='uncertain_additional_food':return 'competing'
 if ('candidate' in r or 'competing' in r or 'alternative' in r or r.startswith('food_item_')):return 'competing'
 if r in {'side','side_dish','side_or_accompaniment','side_food','side_or_accompanying_food','side_or_accompanying_dish','side_or_ingredient','side_or_topping','side_or_garnish','accompaniment','accompaniment_or_side','accompaniment_or_descriptive_food','accompanying_food','accompanying_dish','accompanying_dessert','dip_or_accompaniment','sauce_accompaniment','sauce_or_accompaniment','dressing_or_accompaniment','additional_plate_food','background_food','beverage','meal_plate'}:return 'side'
 if any(t in r for t in ('side','accompan','topping','garnish','sauce','dip','condiment','ingredient','component','filling','frosting','glaze','serving_vessel','serving_form','container','beverage')):return 'side'
 if r in {'example_food','mentioned_context_not_endorsed'}:return 'description'
 if r=='serving_bun':return 'side'
 if any(t in r for t in ('description','same_dish','translation','alias','flavor','specific_clarification','serving_detail','serving_or_flavor','brand')):return 'description'
 if r in {'food_item_a','food_item_b','food_item_c','food_item_d','food_item_e'}:return 'competing'
 return 'other'
def names_of(d):
 for k in ('endorsed_primary_names','literal_extracted_names','food_name_spans','endorsed_name_spans'):
  v=d.get(k)
  if isinstance(v,list):
   z=[]
   for x in v:
    if isinstance(x,str):z.append(x)
    elif isinstance(x,dict):
     t=x.get('name',x.get('span',x.get('text')))
     if isinstance(t,str):z.append(t)
   return z
 x=d.get('literal_extracted_name');return [x] if isinstance(x,str) else []
def relations(d):
 for k in ('name_relations','roles_relations_raw','relations'):
  if isinstance(d.get(k),list):return d[k]
 return []
def classes_for_name(name,patterns):return sorted({c for c,p in patterns if p.search(fmt(name))})
def compile_classes(classes):
 out=[]
 for c in classes:
  for phrase in {fmt(c),fmt(c.replace('_',' '))}:
   if phrase:out.append((c,re.compile(r'(?<![\w])'+re.escape(phrase)+r'(?![\w])',re.I)))
 return out
def extract(d,patterns):
 ns=names_of(d); by={fmt(n):n for n in ns if isinstance(n,str) and n.strip()}; rel=[]
 for r in relations(d):
  if isinstance(r,dict):
   v=r.get('name',r.get('span',r.get('text'))); rr=role(r.get('role'))
   if isinstance(v,str):rel.append((fmt(v),rr))
 relevant=[x for x in rel if x[1] in {'main','coequal','competing'}]
 if relevant:
  chosen=[(by[k],rr) for k,rr in relevant if k in by]
  names=chosen; has_roles=True
 elif rel:names=[];has_roles=True
 else:names=[(n,'unspecified') for n in ns if isinstance(n,str) and n.strip()];has_roles=False
 cand=[{'name':n,'role':rr,'classes':classes_for_name(n,patterns)} for n,rr in names]
 cats=sorted({c for x in cand for c in x['classes']})
 multi=any(x['role'] in {'coequal','competing'} for x in cand) and (len({fmt(x['name']) for x in cand})>1 or len(cats)>1)
 main=[x for x in cand if x['role'] in {'main','coequal','competing'}]
 if len({fmt(x['name']) for x in main})>1 and len({tuple(x['classes']) for x in main})>1:multi=True
 amb=d.get('name_scope_ambiguous');amb=amb if type(amb) is bool else None
 override=d.get('canonical_override')
 if isinstance(override,str) and override not in {'explicit_outside_101','multiple_primary'} and override not in cats:cats=sorted(set(cats+[override]))
 return {'abstain':d.get('abstain') if type(d.get('abstain')) is bool else None,'ambiguous':amb,'names':cand,'literal':sorted({x['name'] for x in cand},key=str.casefold),'classes':cats,'multi':multi,'unresolved':not cand or (amb is True and len(cats)!=1 and not multi)}
def target_absent(text,target):
 t=fmt(text); return not any(re.search(r'(?<![\w])'+re.escape(x)+r'(?![\w])',t) for x in {fmt(target),fmt(target.replace('_',' '))})
def score_variant(answer,target,d,patterns):
 forms={fmt(target),fmt(target.replace('_',' '))}
 if fmt(answer) in forms:return 1,1,'whole_response_exact_class'
 x=extract(d,patterns)
 if x['abstain'] is True:return 0,0,'primary_abstention'
 if x['multi']:return 0,0,'multiple_competing_or_coequal_primary_classes'
 if target_absent(answer,target):return 0,0,'target_absent_from_complete_response'
 if x['unresolved']:return None,None,'primary_scope_or_name_unresolved'
 if len(x['classes'])>1:return 0,0,'multiple_primary_class_candidates_single_label'
 if not x['classes']:
  if x['names'] and x['ambiguous'] is False:return 0,0,'reviewed_primary_name_outside_101'
  return None,None,'no_canonical_class_in_primary_name'
 canon=int(x['classes'][0]==target)
 lit=(int(fmt(x['literal'][0]) in forms) if len({fmt(n) for n in x['literal']})==1 else None)
 return canon,lit,'unique_canonical_class_in_reviewed_primary'
def parse_target_blind_primary(answer,patterns):
    """Conservative lead-clause parser, target-blind and auditable by exact span."""
    text=str(answer);lines=[x.strip() for x in text.splitlines() if x.strip()]
    if not lines:return {'status':'unresolved','primary_name':None,'canonical_candidates':[],'matched_spans':[],'rule':'empty_response','explicit':False}
    first=lines[0]
    # Use only the first sentence of the first nonempty line; later garnishes or
    # explanation sentences cannot displace the identified first dish.
    first=re.split(r'(?<=[.!?])\s+',first,maxsplit=1)[0].strip()
    # Explicit correction/contrast: a negative candidate is excluded and only
    # the affirmative clause after the contrast is considered.
    correction=re.search(r"\b(?:not|is not|isn't)\b[^;.!?]{0,160}[;,]\s*(?:but\s+)?(?:it\s+is|this\s+is|rather\s+)?(?:a|an|the)?\s*([^.!?]+)",first,re.I)
    if correction:phrase=correction.group(1).strip();negated=False
    elif re.search(r"\b(?:not|is not|isn't)\s+[^,;.!?]+",first,re.I):
        return {'status':'unresolved','primary_name':None,'canonical_candidates':[],'matched_spans':[],'rule':'negative_only_no_affirmative_food_name','explicit':False}
    else:phrase=first;negated=False
    cues=list(re.finditer(r"\b(?:appears\s+to\s+be|looks\s+like|seems\s+to\s+be|is\s+likely\s+to\s+be|is|shows|depicts|features|contains)\s+(?:to\s+be\s+)?",phrase,re.I))
    if cues:phrase=phrase[cues[0].end():].strip()
    cutters=[r"\bserved\s+with\b",r"\baccompanied\s+by\b",r"\bwith\s+a\s+side\s+of\b",r"\bwith\b",r"\btopped\s+with\b",r"\bgarnished\s+with\b",r"\balongside\b",r"\bwhich\s+(?:includes|contains|features)\b",r"\bthat\s+(?:includes|contains|features)\b",r"\btoppings?\s+(?:include|including|such\s+as)\b"]
    cut=[]
    for pat in cutters:
        z=re.search(pat,phrase,re.I)
        if z:cut.append(z.start())
    if cut:phrase=phrase[:min(cut)].strip(' \t,;:-')
    phrase=re.sub(r"^(?:a|an|the)\s+(?:(?:plate|bowl|piece|serving|dish|slice|portion)\s+of\s+)+",'',phrase,flags=re.I).strip(' \t,;:-')
    phrase=re.sub(r"^(?:food|dish|item)\s+(?:called|named)\s+",'',phrase,flags=re.I).strip()
    found=classes_for_name(phrase,patterns);spans=[]
    for cls in found:
        for c in {fmt(cls),fmt(cls.replace('_',' '))}:
            pat=re.compile(r"(?<![\w])"+re.escape(c)+r"(?![\w])",re.I);z=pat.search(phrase)
            if z:
                base=text.find(phrase);spans.append({'text':phrase[z.start():z.end()],'class':cls,'start':base+z.start(),'end':base+z.end()});break
    explicit=bool(cues or correction or re.match(r"^(?:based on|the (?:dish|food|image)|this (?:is|appears)|it (?:is|appears)|answer:)",first,re.I))
    if len(found)==1:return {'status':'unique_class','primary_name':phrase,'canonical_candidates':found,'matched_spans':spans,'rule':'leading_affirmative_class_single','explicit':explicit}
    if len(found)>1:return {'status':'multiple_classes','primary_name':phrase,'canonical_candidates':found,'matched_spans':spans,'rule':'leading_clause_multiple_class_candidates','explicit':explicit}
    if explicit and phrase:return {'status':'explicit_outside_101','primary_name':phrase,'canonical_candidates':[],'matched_spans':[],'rule':'leading_identification_phrase_outside_101','explicit':True}
    return {'status':'unresolved','primary_name':None,'canonical_candidates':[],'matched_spans':[],'rule':'no_conservative_leading_name_parse','explicit':False}

def load_stable():
 manifest=json.loads((ROOT/'data/manifest.json').read_text());classes=manifest['canonical_classes'];assert len(classes)==101 and len(set(classes))==101
 sm=json.loads((OUT/'source_manifest.json').read_text());assert sha(ROOT/sm['source_manifest'])==sm['source_manifest_sha256'];p=ROOT/sm['reviewed_answers'];assert sha(p)==sm['reviewed_answers_sha256'];assert sha(ROOT/sm['automatic_behavior'])==sm['automatic_behavior_sha256'];reviews={}
 for r in rows(p):reviews[(r['question'],r['answer'])]=r['variants']
 return manifest,classes,reviews,sm

def configured_marker(text,row):
 t=fmt(text)
 # Actual condition marker is the only condition-specific auto marker. Known
 # standalone refusal phrases are handled only as whole responses.
 vals={fmt(row.get(k,'')) for k in ('marker','reference_marker') if row.get(k)}
 return bool(t and t in vals)
def run_score(output_dir=None):
 run_out=Path(output_dir).resolve() if output_dir else OUT
 manifest,classes,reviews,sm=load_stable();patterns=compile_classes(classes)
 auto_path=ROOT/sm['automatic_behavior'];assert sha(auto_path)==sm['automatic_behavior_sha256']
 auto=collections.defaultdict(dict)
 with gzip.open(auto_path,'rt',encoding='utf8') as f:
  for line in f:
   d=json.loads(line);auto[d['model']][d['key']]=d
 sources=manifest['sources'];by_model=collections.defaultdict(list)
 for s in sources:by_model[s['model']].append(s)
 counts=collections.defaultdict(collections.Counter);seen=set();total=0;qa_cache={};unresolved=collections.defaultdict(lambda:{'memberships':[]})
 run_out.mkdir(parents=True,exist_ok=True);dest=run_out/'score_rows.jsonl.gz';partial=run_out/'score_rows.jsonl.gz.partial'
 if dest.exists() or partial.exists():raise FileExistsError(f'preserving existing output: {dest}')
 raw_hashes={}
 with gzip.open(partial,'wt',encoding='utf8',newline='\n',compresslevel=6) as out:
  for model,files in by_model.items():
   for entry in files:
    p=ROOT/entry['path'];h=hashlib.sha256();nfile=0
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rb') as f:
     for lineno,raw in enumerate(f,1):
      h.update(raw);nfile+=1;row=json.loads(raw);identity=(model,row['key'])
      if identity in seen:raise RuntimeError(f'duplicate raw identity: {identity}')
      seen.add(identity);total+=1
      if row['model']!=model:raise RuntimeError('path/model mismatch')
      sample=row['sample'];q=sample['question'];answer=row['text'];target=sample['class'];qkey=(q,answer)
      if qkey not in qa_cache:
       vs=reviews.get(qkey,[]); states=[extract(d,patterns) for d in vs]
       qa_cache[qkey]={'variants':vs,'states':states,'classes':sorted({c for x in states for c in x['classes']}),'names':sorted({n for x in states for n in x['literal']},key=str.casefold)}
      cache=qa_cache[qkey];vs=cache['variants'];state_scores=[]
      label=auto[model].get(row['key']);rawsha=hashlib.sha256(raw).hexdigest();abstain=None;behavior=None;auto_reason=None
      if label and label['source_row_sha256']!=rawsha:raise RuntimeError(f'auto behavior raw SHA mismatch: {model}/{row["key"]}')
      # Binary priority: root exact-QA/overlays, fresh follow-up, unanimous
      # existing review, frozen automatic label (raw-line SHA bound), condition marker.
      rootvs=[d for d in vs if d.get('_root_overlay') or d.get('_source')=='root_independent_gap_review']
      followvs=[d for d in vs if str(d.get('_source','')).startswith('followup_')]
      selected_bin=rootvs or followvs or vs
      vals={d.get('abstain') for d in selected_bin if type(d.get('abstain')) is bool}
      if len(vals)==1:abstain=next(iter(vals));auto_reason='root_or_reviewed_QA_binary'
      elif len(vals)>1:abstain=None;auto_reason='review_binary_conflict'
      elif label:
       lab=label.get('behavior_label')
       if type(label.get('abstain')) is bool:abstain=label['abstain'];behavior='abstain' if abstain else 'answer';auto_reason=label.get('source') or 'frozen_primary_review_binary'
       elif lab:abstain=(lab=='abstain');behavior='abstain' if abstain else 'answer';auto_reason=label.get('source') or 'frozen_primary_review_binary'
      if abstain is None and auto_reason!='review_binary_conflict' and configured_marker(answer,row):abstain=True;behavior='abstain';auto_reason='exact_condition_marker_response'
      # Apply the chosen binary decision to all name variants, preventing a
      # lower-priority older abstain field from reversing a root correction.
      effective_vs=[]
      for v in vs:
       z=dict(v);z['abstain']=abstain if abstain is not None else None;effective_vs.append(z)
       if z.get('canonical_override')=='explicit_outside_101':state_scores.append((0,0,'explicit_primary_outside_101'));continue
       if z.get('canonical_override')=='multiple_primary':state_scores.append((0,0,'multiple_competing_or_coequal_primary_classes'));continue
       if isinstance(z.get('canonical_override'),str) and z['canonical_override']:
        c=z['canonical_override'];full=z.get('literal_full_name');lit=(int(fmt(full) in {fmt(target),fmt(target.replace('_',' '))}) if isinstance(full,str) else 0);ovlit=z.get('literal_score_override');lit=ovlit if type(ovlit) is int else lit;state_scores.append((int(c==target),lit,'root_exact_QA_primary_class'));continue
       a,b,r=score_variant(answer,target,z,patterns);state_scores.append((a,b,r))
      vs=effective_vs
      exact=fmt(answer) in {fmt(target),fmt(target.replace('_',' '))}
      if abstain is True:canon=lit=0;reason=auto_reason or 'explicit_abstention'
      elif rootvs and state_scores:
       cvs={x[0] for x in state_scores};lvs={x[1] for x in state_scores}
       canon=next(iter(cvs)) if len(cvs)==1 else None;lit=next(iter(lvs)) if len(lvs)==1 else None
       reasons={x[2] for x in state_scores};reason=next(iter(reasons)) if len(reasons)==1 else 'root_review_conflict'
       behavior=behavior or ('answer' if abstain is False else None)
      elif exact:canon=lit=1;reason='whole_response_exact_class';abstain=False;behavior='answer'
      elif target_absent(answer,target):canon=lit=0;reason='target_absent_from_complete_response';behavior=behavior or 'answer'
      elif state_scores:
       cvs={x[0] for x in state_scores};lvs={x[1] for x in state_scores}
       canon=next(iter(cvs)) if len(cvs)==1 else None;lit=next(iter(lvs)) if len(lvs)==1 else None
       reasons={x[2] for x in state_scores};reason=next(iter(reasons)) if len(reasons)==1 else 'cross_review_disagreement'
       if len(cvs)>1 or len(lvs)>1:reason='cross_review_disagreement'
       behavior=behavior or ('answer' if abstain is False else None)
      else:canon=lit=None;reason='no_exact_QA_primary_review'
      if behavior is None and abstain is not None:behavior='abstain' if abstain else 'answer'
      rec={k:row[k] for k in COND};rec.update(key=row['key'],sample_id=sample['id'],qa_key=qah(q,answer),target_class=target,source_path=entry['path'],source_line=lineno,raw_line_sha256=rawsha,
       literal_extracted_names=cache['names'],literal_extracted_name=cache['names'][0] if len(cache['names'])==1 else None,
       canonical_class_candidates=cache['classes'],canonical_name_in_primary=cache['classes'][0] if len(cache['classes'])==1 else None,
       literal_extracted_name_score=lit,canonical_name_in_primary_score=canon,abstain=abstain,behavior=behavior,score_reason=reason,
       review_sources=sorted({d.get('_source',d.get('source','unknown')) for d in vs}),review_statuses=sorted({d.get('_status','existing_review') for d in vs}),multiple_primary_classes=(len(cache['classes'])>1 or any(d.get('canonical_override')=='multiple_primary' for d in vs)),main_prompt_sha256=hashlib.sha256(row.get('prompt','').encode()).hexdigest(),reference_prompt_sha256=(hashlib.sha256(row['reference_prompt'].encode()).hexdigest() if isinstance(row.get('reference_prompt'),str) else None),neutral_prompt_sha256=(hashlib.sha256(row['neutral_prompt'].encode()).hexdigest() if isinstance(row.get('neutral_prompt'),str) else None),config_sha256=hashlib.sha256(json.dumps(row.get('config'),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),seed=row.get('seed'))
      out.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n')
      c=counts[tuple(row[k] for k in COND)];c['n']+=1
      for typ,val in [('canonical',canon),('literal',lit)]:c[typ+'_unknown' if val is None else typ+'_'+str(val)]+=1
      c['abstain_unknown' if abstain is None else 'abstain_'+str(int(abstain))]+=1;c['reviewed_rows']+=bool(vs);c['multiple_primary_class_rows']+=(len(cache['classes'])>1 or any(d.get('canonical_override')=='multiple_primary' for d in vs))
      if canon is None or abstain is None:
       g=unresolved[qkey];g.update(qa_key=qah(q,answer),question=q,answer=answer,canonical_candidates=cache['classes'],literal_names=cache['names']);g['memberships'].append({'model':model,'key':row['key'],'sample_id':sample['id'],'target_class':target,'canonical_unknown':canon is None,'abstain_unknown':abstain is None,'score_reason':reason,'condition':{k:row[k] for k in COND}})
    if nfile!=entry['rows']:raise RuntimeError(f'source line count {entry["path"]}: {nfile}/{entry["rows"]}')
    if h.hexdigest()!=entry['decompressed_sha256']:raise RuntimeError(f'decompressed source SHA mismatch: {entry["path"]}')
    if sha(p)!=entry['sha256']:raise RuntimeError(f'compressed source file SHA mismatch: {entry["path"]}')
    raw_hashes[entry['path']]=h.hexdigest()
 partial.replace(dest)
 if total!=853248 or len(seen)!=853248 or len(counts)!=352:raise RuntimeError(f'coverage: {total}/{len(seen)}/{len(counts)}')
 fields=COND+['n','canonical_correct','canonical_incorrect','canonical_unknown','canonical_lower','canonical_upper','literal_correct','literal_incorrect','literal_unknown','literal_lower','literal_upper','abstain_true','abstain_false','abstain_unknown','reviewed_rows','multiple_primary_class_rows'];cp=run_out/'condition_counts.csv'
 if cp.exists():raise FileExistsError(cp)
 with cp.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for cond,c in sorted(counts.items()):
   assert c['n']==2424 and c['canonical_0']+c['canonical_1']+c['canonical_unknown']==2424 and c['literal_0']+c['literal_1']+c['literal_unknown']==2424 and c['abstain_0']+c['abstain_1']+c['abstain_unknown']==2424
   r=dict(zip(COND,cond));r.update(n=2424,canonical_correct=c['canonical_1'],canonical_incorrect=c['canonical_0'],canonical_unknown=c['canonical_unknown'],canonical_lower=c['canonical_1']/2424,canonical_upper=(c['canonical_1']+c['canonical_unknown'])/2424,literal_correct=c['literal_1'],literal_incorrect=c['literal_0'],literal_unknown=c['literal_unknown'],literal_lower=c['literal_1']/2424,literal_upper=(c['literal_1']+c['literal_unknown'])/2424,abstain_true=c['abstain_1'],abstain_false=c['abstain_0'],abstain_unknown=c['abstain_unknown'],reviewed_rows=c['reviewed_rows'],multiple_primary_class_rows=c['multiple_primary_class_rows']);w.writerow(r)
 gp=run_out/'unresolved_groups.jsonl'
 with gp.open('w',encoding='utf8',newline='\n') as f:
  for (q,a),g in sorted(unresolved.items(),key=lambda x:qah(*x[0])):f.write(json.dumps(g,ensure_ascii=False,separators=(',',':'))+'\n')
 receipt={'status':'full_rule_score_pending_root_review','rows':total,'conditions':len(counts),'rows_per_condition':2424,'unique_model_key_identities':len(seen),'unique_review_QA':len(reviews),'unique_scored_QA':len(qa_cache),'unresolved_unique_QA':len(unresolved),'canonical_classes':len(classes),'raw_source_sha256':raw_hashes,'rule_parser_counts':json.loads((OUT/'rule_candidate_counts.json').read_text()) if (OUT/'rule_candidate_counts.json').exists() else {},'outputs':{},'policy':{'primary':'canonical_name_in_primary_score','sensitivity':'literal_extracted_name_score','extraction':'exact-QA endorsed names intersect explicit main roles; side/description excluded; target-blind 101-class phrase matching at word boundaries','multiple':'explicit coequal/competing distinct main classes score 0 for single-label task; abstain stays false','modifiers':'full reviewed name retained; a unique embedded canonical class is scored in primary column','markers':'frozen behavior labels bound to raw-line SHA; otherwise exact complete response equal to actual condition marker or known standalone marker only','unknown':'missing/ambiguous scope remains null, never converted to incorrect by default','zero':'target canonical phrase absent from complete response is a necessary-condition strict zero'},'limits':['A100 and B100 follow-up names are integrated but pending full root cohort quality review.','C204 contributes only root-reviewed abstention=false.','Root independent exact-QA classifications are preserved with higher priority.','No global quality acceptance is implied by complete file coverage.']}
 for p in [dest,cp,gp,OUT/'reviewed_answers.jsonl',OUT/'source_manifest.json',OUT/'automatic_behavior.jsonl.gz',ROOT/'data/manifest.json']:
  try:name=str(p.relative_to(ROOT))
  except ValueError:name=str(p)
  receipt['outputs'][name]=sha(p)
 rp=run_out/'receipt.json';rp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'rows':total,'conditions':len(counts),'unique_review_QA':len(reviews),'unique_scored_QA':len(qa_cache),'unresolved_unique_QA':len(unresolved),'outputs':receipt['outputs']},ensure_ascii=False,indent=2))
def self_test():
 classes=['fried_rice','shrimp_and_grits','fish_and_chips','ice_cream','chocolate_cake','steak','french_fries','cup_cakes','fish_tacos']
 pat=compile_classes(classes)
 def d(names,rels,amb=False,ab=False):return {'endorsed_primary_names':names,'name_relations':rels,'name_scope_ambiguous':amb,'abstain':ab}
 assert classes_for_name('shrimp and grits',pat)==['shrimp_and_grits']
 assert target_absent('fish taco','fish_tacos')
 assert target_absent('cupcakes','cup_cakes')
 assert target_absent('fried-rice','fried_rice')
 assert score_variant('not fried rice; it is risotto','fried_rice',d(['risotto'],[{'name':'risotto','role':'main'}]),pat)[0]==0
 assert score_variant('chocolate cake with ice cream','chocolate_cake',d(['chocolate cake'],[{'name':'chocolate cake','role':'main'},{'name':'ice cream','role':'side'}]),pat)[0]==1
 assert score_variant('steak with french fries','steak',d(['steak','french fries'],[{'name':'steak','role':'main'},{'name':'french fries','role':'side'}]),pat)[0]==1
 assert score_variant('steak and french fries','steak',d(['steak','french fries'],[{'name':'steak','role':'coequal'},{'name':'french fries','role':'coequal'}],True),pat)[0]==0
 assert score_variant('shrimp and grits','shrimp_and_grits',d(['shrimp and grits'],[{'name':'shrimp and grits','role':'main'}]),pat)[0]==1
 assert score_variant('fish taco','fish_tacos',d(['fish taco'],[{'name':'fish taco','role':'main'}]),pat)[0]==0
 assert score_variant('fried-rice','fried_rice',d(['fried-rice'],[{'name':'fried-rice','role':'main'}]),pat)[0]==0
 assert configured_marker(' UNKNOWN ',{'marker':'UNKNOWN','reference_marker':'UNCLEAR'})
 assert not configured_marker('if unsure answer UNKNOWN',{'marker':'UNKNOWN','reference_marker':'UNCLEAR'})
 assert score_variant('echo says fried rice; answer is risotto','fried_rice',d(['risotto'],[{'name':'risotto','role':'main'}]),pat)[0]==0
 print('12 primary score rules passed')
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--run',action='store_true');g.add_argument('--test',action='store_true');ap.add_argument('--output-dir',default=None,help='output directory; default is outputs/annotations/main_results');a=ap.parse_args()
 if a.run:run_score(a.output_dir)
 else:self_test()
