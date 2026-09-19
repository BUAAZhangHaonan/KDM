"""Full-manifest native shape/token preflight, CPU only; no model loading."""
import argparse,hashlib,inspect,json
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
from kdm.models.backbone import FamilyModel
from kdm.models.remote import MiniCPMModel,PhiVisionModel
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('key');a=p.parse_args()
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
spec=json.loads((ROOT/'configs/runtime'/f'{a.key}.json').read_text())
config=json.loads((Path(spec['kwargs']['model_path'])/'config.json').read_text())
processor_kwargs={'num_crops':4} if a.key=='phi35' else {}
proc=AutoProcessor.from_pretrained(spec['kwargs']['model_path'],trust_remote_code=True,local_files_only=True,**processor_kwargs)
headers_path=ROOT/'outputs/records/sid_visual_image_headers.json';headers=json.loads(headers_path.read_text())
assert len(headers['rows'])==9167 and not headers['errors']
assert headers['manifest_sha256']==hashlib.sha256((ROOT/'data/current/all.jsonl').read_bytes()).hexdigest()
count_by_size={};errors=[]
for row in headers['rows']:
 size=tuple(row[-2:])
 if size in count_by_size:continue
 try:
  if a.key.startswith('minicpm'):
   placeholder=proc.image_processor.get_slice_image_placeholder(image_size=(size[1],size[0]))
   count_by_size[size]=placeholder.count(proc.image_processor.unk_token)
  elif a.key=='phi35':
   count_by_size[size]=int(proc.calc_num_image_tokens_from_image_size(size[1],size[0]))
  else:
   result=proc._get_num_multimodal_tokens(image_sizes=[size])
   counts=result.num_image_tokens if hasattr(result,'num_image_tokens') else result['num_image_tokens']
   count_by_size[size]=int(counts[0])
 except Exception as e:
  count_by_size[size]=None;errors.append({'height':size[0],'width':size[1],'error':repr(e)})
bygroup=defaultdict(list);bad=[];size_examples={}
for id,dataset,split,h,w in headers['rows']:
 count=count_by_size[(h,w)];size_examples.setdefault((h,w),id)
 if count is None:continue
 bygroup[(dataset,split)].append(count)
 if count<100:bad.append({'id':id,'dataset':dataset,'split':split,'height':h,'width':w,'visual_tokens':count})
sources={}
for obj in [type(proc),type(proc.image_processor)]:
 path=Path(inspect.getfile(obj));sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
record={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':spec['environment_python']+' verification/check_full_visual_counts.py '+a.key,'key':a.key,'spec':spec,'scope':'full-manifest image-header enumeration with installed native processor count formula, checked against actual boundary processor outputs; not model forward','manifest_sha256':headers['manifest_sha256'],'header_record':str(headers_path.relative_to(ROOT)),'header_record_sha256':hashlib.sha256(headers_path.read_bytes()).hexdigest(),'processor_class':type(proc).__name__,'image_processor_class':type(proc.image_processor).__name__,'image_processor_config':proc.image_processor.to_dict(),'installed_source_sha256':sources,'total_images':len(headers['rows']),'unique_dimensions':len(count_by_size),'shape_errors':errors,'groups':[{'dataset':d,'split':s,'n':len(v),'minimum':min(v),'maximum':max(v),'below100':sum(n<100 for n in v)} for (d,s),v in sorted(bygroup.items())],'below100_count':len(bad),'below100':bad,'shape_formula_counts':[{'height':h,'width':w,'visual_tokens':v} for (h,w),v in sorted(count_by_size.items())],'boundary_checks':[],'boundary_checks_passed':False,'all_images_at_least100':False}
out=ROOT/'outputs/verification'/f'{a.key}_sid_full_visual_count.json'
if out.exists():
 raise FileExistsError(f'Refusing to overwrite existing visual-count evidence: {out}')
out.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({'key':a.key,'stage':'exact_native_formula_enumerated','below100':len(bad),'groups':record['groups'],'shape_errors':len(errors)},ensure_ascii=False),flush=True)
if errors:raise RuntimeError('Native formula rejected an actual image shape; no global pass')
# Deterministic real-image boundary checks; no synthetic resizing or parameter overrides.
sizes=list(count_by_size)
selected=[]
def add(size):
 if size not in selected:selected.append(size)
add(min(sizes,key=lambda x:count_by_size[x]));add(max(sizes,key=lambda x:count_by_size[x]))
for pool,choose in [([s for s in sizes if count_by_size[s]<100],max),([s for s in sizes if count_by_size[s]>=100],min)]:
 if pool:add(choose(pool,key=lambda x:count_by_size[x]))
for key in [lambda x:x[0],lambda x:x[1],lambda x:max(x)/min(x),lambda x:x[0]*x[1]]:
 add(min(sizes,key=key));add(max(sizes,key=key))
manifest={x['id']:x for x in map(json.loads,(ROOT/'data/current/all.jsonl').read_text().splitlines())}
engine_class=MiniCPMModel if a.key.startswith('minicpm') else PhiVisionModel if a.key=='phi35' else FamilyModel
engine=engine_class.__new__(engine_class);engine.proc=proc;engine.device='cpu';engine.mt=config['model_type'];engine.model_path=Path(spec['kwargs']['model_path'])
exif_path=ROOT/'outputs/records/sid_image_exif_orientation_check.json'
exif=json.loads(exif_path.read_text())
record['exif_orientation_record']={'path':str(exif_path.relative_to(ROOT)),
    'sha256':hashlib.sha256(exif_path.read_bytes()).hexdigest(),'counts':exif['orientations']}
if a.key in ('qwen25vl','qwen3vl','glm46v'):
 symmetric=[]
 for h,w in sizes:
  result=proc._get_num_multimodal_tokens(image_sizes=[(w,h)])
  counts=result.num_image_tokens if hasattr(result,'num_image_tokens') else result['num_image_tokens']
  symmetric.append(int(counts[0])==count_by_size[(h,w)])
 record['count_invariant_under_exif_90_degree_swap']=all(symmetric)
 assert all(symmetric)
checks_to_run=[((h,w),size_examples[(h,w)]) for h,w in selected]
if exif['nontrivial']:
 extra_id=exif['nontrivial'][0]['id']
 extra=next(row for row in headers['rows'] if row[0]==extra_id)
 if extra_id not in [id for size,id in checks_to_run]:checks_to_run.append((tuple(extra[-2:]),extra_id))
for (h,w),id in checks_to_run:
 sample=manifest[id]
 with Image.open(sample['image_path']) as image:inputs=engine.build(image.convert('RGB'),'What food is shown? Answer briefly.')
 ids=inputs['input_ids'][0]
 if a.key.startswith('minicpm'):
  bounds=inputs['image_bound'][0]
  positions=torch.cat([torch.arange(int(start),int(end)) for start,end in bounds])
  grid=None;grid_count=int((bounds[:,1]-bounds[:,0]).sum())
 elif a.key=='phi35':
  positions=((ids<0)&(ids>-int(1e9))).nonzero().flatten()
  grid=None;grid_count=int(positions.numel())
 else:
  positions=(ids==proc.image_token_id).nonzero().flatten()
  grid=inputs['image_grid_thw'][0].tolist();grid_count=grid[0]*grid[1]*grid[2]//proc.image_processor.merge_size**2
 actual=int(positions.numel())
 expected=count_by_size[(h,w)]
 check={'id':id,'height':h,'width':w,'formula_tokens':expected,'native_visual_position_count':actual,'image_grid_thw':grid,'grid_count':grid_count,'visual_positions_contiguous':bool(actual and int(positions[-1]-positions[0]+1)==actual),'equal':actual==expected==grid_count}
 record['boundary_checks'].append(check)
 del inputs
 if not check['equal']:
  out.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n');raise RuntimeError('Native boundary count mismatch; no extrapolated pass')
record['boundary_checks_passed']=all(x['equal'] for x in record['boundary_checks'])
record['all_images_at_least100']=not bad and not errors and record['boundary_checks_passed']
record['gpu_initialized']=torch.cuda.is_initialized();assert not record['gpu_initialized']
out.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({'key':a.key,'boundary_checks':len(record['boundary_checks']),'boundary_checks_passed':record['boundary_checks_passed'],'all_images_at_least100':record['all_images_at_least100'],'below100':len(bad)},ensure_ascii=False))
