import argparse,hashlib,json
from pathlib import Path
from transformers import AutoProcessor
from PIL import Image
from kdm.models.remote import MiniCPMModel,PhiVisionModel
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
spec=json.loads((ROOT/a.spec).read_text());model_path=Path(spec['kwargs']['model_path'])
cls=MiniCPMModel if spec['key'].startswith('minicpm') else PhiVisionModel
engine=cls.__new__(cls);engine.model_path=model_path;engine.device='cpu'
kw={'num_crops':4} if cls is PhiVisionModel else {}
engine.proc=AutoProcessor.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True,**kw)
rows=[]
for line in (ROOT/'data/current/interface16.jsonl').read_text().splitlines():
 sample=json.loads(line);im=Image.open(sample['image_path']).convert('RGB')
 inputs=engine.build(im,'What food is shown? Answer briefly.');ids=inputs['input_ids'][0]
 if cls is MiniCPMModel:
  bounds=inputs['image_bound'][0].tolist();positions=[i for start,end in bounds for i in range(start,end)]
 else:
  positions=((ids<0)&(ids>-1000000000)).nonzero().flatten().tolist();bounds=[]
  for i in positions:
   if not bounds or bounds[-1][1]!=i:bounds.append([i,i+1])
   else:bounds[-1][1]+=1
 gaps=[i for i in range(min(positions),max(positions)+1) if i not in set(positions)] if positions else []
 rows.append({'id':sample['id'],'sequence_length':len(ids),'visual_count':len(positions),
              'bounds_half_open':bounds,'min_max_gap_count':len(gaps),
              'gap_token_ids':[int(ids[i]) for i in gaps],
              'gap_text':engine.proc.tokenizer.decode([int(ids[i]) for i in gaps]) if gaps else '',
              'visual_positions':positions,'image_bound_present':'image_bound' in inputs,
              'negative_token_ids':sorted(set(int(ids[i]) for i in positions)) if cls is PhiVisionModel else []})
record={'spec':spec,'scope':'processor-only native visual embedding position mapping; no model weights/GPU forward',
        'mapping_source':'image_bound half-open union' if cls is MiniCPMModel else 'Phi3ImageEmbedding native negative input id predicate',
        'rows':rows,'remote_source_sha256':hashlib.sha256((ROOT/'src/kdm/models/remote.py').read_bytes()).hexdigest()}
(ROOT/a.output).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'key':spec['key'],'visual_counts':sorted(set(r['visual_count'] for r in rows)),
                  'number_of_intervals':sorted(set(len(r['bounds_half_open']) for r in rows)),
                  'gap_counts':sorted(set(r['min_max_gap_count'] for r in rows))}))
