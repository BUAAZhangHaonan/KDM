"""Prepare only the user-authorized 108 census gaps plus two formal fragments."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.open()]
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def jsonl(p,rs):
 with p.open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
 parent=ROOT/'outputs/annotations/deepseek_v2/census/merged_after_retry151_v1';queue=ROOT/'outputs/annotations/deepseek_v2/census/queue.jsonl'
 errors=read(parent/'errors.jsonl');keys={r['key'] for r in errors};assert len(keys)==len(errors)==108
 source=[r for r in read(queue) if r['key'] in keys];assert len(source)==108
 previous={r['key'] for r in read(parent/'labels.jsonl')};assert len(previous)==293236 and not previous&keys
 old=ROOT/'outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1';formal=[r for r in read(old/'validated_v3/semantic_labels.jsonl') if r['status']=='unresolved'];assert len(formal)==2 and {r['text'] for r in formal}=={"DON'T",'The Celti'}
 blinded=[];mapping=[]
 for n,r in enumerate(source+formal,1):
  q={'id':n,'question':r.get('question','What specific food is shown in this image?'),'answer':r['text']}
  q['group_sha256']=hashlib.sha256(json.dumps({'question':q['question'],'answer':q['answer']},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  blinded.append(q);mapping.append({'id':n,'kind':'census' if n<=108 else 'formal_fragment','source':r})
 OUT.mkdir(parents=True,exist_ok=False);jsonl(OUT/'blind_input.jsonl',blinded);jsonl(OUT/'source_mapping.jsonl',mapping)
 (OUT/'RUBRIC.txt').write_bytes((old/'RUBRIC.txt').read_bytes())
 write(OUT/'manifest.json',{'expected_census':108,'formal_fragments':2,'input_sha256':sha(OUT/'blind_input.jsonl'),'mapping_sha256':sha(OUT/'source_mapping.jsonl'),'rubric_sha256':sha(OUT/'RUBRIC.txt'),'parent_labels_sha256':sha(parent/'labels.jsonl'),'parent_identity_sha256':sha(parent/'identity.json'),'errors_sha256':sha(parent/'errors.jsonl'),'model':'gpt-6-luna','reasoning_effort':'medium','human_reviewed':False,'final_gt':False,'api_requests':0})
 print(OUT)
if __name__=='__main__':main()
