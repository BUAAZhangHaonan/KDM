"""Prepare only the 9047 user-authorized unresolved rows; blind exact deduplication."""
import ast,hashlib,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(j):return hashlib.sha256(json.dumps(j,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,j):
 with p.open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2)
def jsonl(p,rows):
 with p.open('x',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 names={'llava16_mistral_unknown_main_20260923_v1':6880,'qwen25vl_unknown_controls_20260923_v1':2167}
 mapping=[];groups={};sources=[];seen=set()
 for name,want in names.items():
  directory=ROOT/'outputs/annotations/acceleration_v4'/name; summary=json.loads((directory/'summary.json').read_text())
  labels=[json.loads(x) for x in (directory/'labels.jsonl').open()]
  selected=[x for x in labels if x['screening_label']=='needs_confirmation']
  assert len(selected)==summary['unresolved']==want
  byline={r['source_line']:r for r in selected};source_paths={r['source_path'] for r in selected};assert len(source_paths)==1
  rawpath=ROOT/next(iter(source_paths));found=0
  sourceproof=summary['sources'][0];assert len(summary['sources'])==1
  with rawpath.open('rb') as f:assert hashlib.sha256(f.read(sourceproof['source_prefix_bytes'])).hexdigest()==sourceproof['source_prefix_sha256']
  with rawpath.open('rb') as f:
   for line_no,line in enumerate(f,1):
    if line_no>max(byline):break
    if line_no not in byline:continue
    label=byline[line_no];row=json.loads(line)
    assert line.endswith(b'\n') and hashlib.sha256(line).hexdigest()==label['source_row_sha256']
    assert row['key']==label['key'] and row['identity']==label['source_identity'] and row['sample']['id']==label['sample_id']
    pair=(row['model'],row['key']);assert pair not in seen;seen.add(pair)
    content={'question':row['sample']['question'],'answer':row['text']};gid=digest(content)
    if gid in groups:assert groups[gid]==content
    groups[gid]=content
    mapping.append({'source_label':label,'source_label_file':str((directory/'labels.jsonl').relative_to(ROOT)),'group_sha256':gid});found+=1
  assert found==want
  sources.append({'screening_directory':str(directory.relative_to(ROOT)),'labels_sha256':sha(directory/'labels.jsonl'),'summary_sha256':sha(directory/'summary.json'),'unresolved':want,'source_prefix':sourceproof})
 assert len(mapping)==9047
 queue=[{'id':i,'group_sha256':g,**v} for i,(g,v) in enumerate(groups.items(),1)]
 byhash={r['group_sha256']:r['id'] for r in queue}
 for r in mapping:r['id']=byhash[r['group_sha256']]
 jsonl(OUT/'blind_queue.jsonl',queue);jsonl(OUT/'source_mapping.jsonl',mapping)
 tree=ast.parse((ROOT/'workflows/deepseek_annotation_v2/annotate.py').read_text())
 rubric=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
 (OUT/'RUBRIC.txt').write_text(rubric)
 write(OUT/'manifest.json',{'schema':'luna_semantic_9047_queue_v1','model':'gpt-6-luna','reasoning_effort':'medium','subagent':'/root/luna_semantic_9047','source_rows':len(mapping),'exact_question_answer_groups':len(queue),'sources':sources,'queue_sha256':sha(OUT/'blind_queue.jsonl'),'mapping_sha256':sha(OUT/'source_mapping.jsonl'),'rubric_sha256':sha(OUT/'RUBRIC.txt'),'prepare_sha256':sha(Path(__file__)),'human_reviewed':False,'final_gt':False,'separate_api_calls':0})
 (OUT/'batches').mkdir()
 for start in range(0,len(queue),100):jsonl(OUT/'batches'/f'input_{start//100+1:03d}.jsonl',queue[start:start+100])
 print(json.dumps({'source_rows':len(mapping),'unique_blind_groups':len(queue),'batches':(len(queue)+99)//100,'out':str(OUT)}))
if __name__=='__main__':main()
