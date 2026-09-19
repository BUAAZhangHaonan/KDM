"""Maximum real VizWiz input resource receipt; no decoding, scoring or fallback."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--stage',choices=['layer','instruction_vcd','cda'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=ROOT/a.output
 if out.exists():raise FileExistsError(out)
 spec=json.loads((ROOT/a.spec).read_text());key=spec['key'];countpath=ROOT/f'outputs/verification/{key}_sid_full_visual_count.json';counts=json.loads(countpath.read_text())
 headers=json.loads((ROOT/counts['header_record']).read_text())
 manifest=ROOT/'data/current/all.jsonl'
 if sha(manifest)!=counts['manifest_sha256']:raise ValueError('manifest count identity changed')
 samples=[json.loads(l) for l in manifest.read_text().splitlines()]
 maximum=max(g['maximum'] for g in counts['groups'] if g['dataset']=='vizwiz')
 selected=next(c for c in counts['boundary_checks'] if c['formula_tokens']==maximum and c['id'].startswith('vizwiz:'))
 sample=next(s for s in samples if s['id']==selected['id'])
 record={'scope':'Maximum-count actual VizWiz image; resource-only fixed prefixes, no generated answers or scientific scores; independent process per actual session group','passed':False,'spec':spec,'stage':a.stage,'script_sha256':sha(__file__),'command':sys.argv,'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'sample':{k:sample[k] for k in ['id','dataset','split','image_path']},'image_size':[selected['width'],selected['height']],'expected_visual_tokens':maximum,'manifest_sha256':sha(manifest),'count_record':str(countpath.relative_to(ROOT)),'count_record_sha256':sha(countpath),'runtime_adapter_sha256':{n:sha(ROOT/'src/kdm/models'/n) for n in ['hf.py','backbone.py']},'visits':[],'branch_input_shapes':{},'phase':'imports'}
 start=time.monotonic();torch=None
 try:
  import torch,numpy as np
  from PIL import Image
  from kdm.pipeline import make_backend,sessions
  from kdm.decoding import DecodeConfig
  from kdm.prompts import task_prompt
  from kdm.io import stable_seed
  record['phase']='model_load';backend=make_backend(spec,'cuda:0')
  record['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
  record['model_load_peak_bytes']=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
  record['phase']='session_build'
  with Image.open(sample['image_path']) as im:image=im.convert('RGB')
  method={'layer':'dola','instruction_vcd':'instruction_vcd','cda':'cda_visual'}[a.stage]
  task={'sample':sample,'marker':'UNKNOWN','reference_marker':'UNKNOWN','guided':True,'reference_guided':False}
  seed=stable_seed(sample['id'],key,0)
  main,reference,neutral,_,_=sessions(backend,image,task,DecodeConfig(method=method),seed)
  if a.stage=='layer':branches=[('main_need_layers',main)]
  elif a.stage=='instruction_vcd':branches=[('main',main),('noise_reference',reference),('neutral',neutral)]
  else:
   plain=task_prompt(sample['question'],guided=False);null=task_prompt('[N/A]',guided=False)
   prior=backend.session(image,plain,reference='text_only');context=backend.session(image,plain)
   null_prior=backend.session(image,null,reference='text_only');null_context=backend.session(Image.new('RGB',image.size,(127,127,127)),null)
   branches=[('prior_text',prior),('context_image',context),('abstention_image',main),('null_prior_text',null_prior),('null_context_image',null_context)]
  record['branch_order']=[n for n,s in branches];record['seed']=seed
  for name,session in branches:
   record['branch_input_shapes'][name]={k:list(v.shape) for k,v in session.inputs.items() if hasattr(v,'shape')}
   if not session.text_only:
    grid=session.inputs['image_grid_thw'];n=int(grid.prod().item())//(backend.em.proc.image_processor.merge_size**2)
    record['branch_input_shapes'][name]['visual_token_count']=n
    if n!=maximum:raise AssertionError(f'actual visual count {n} != maximum {maximum}')
  tokens=backend.encode(' food dish plate')
  if len(tokens)<2:raise ValueError('fixed neutral prefix needs two token IDs')
  record['fixed_prefix_token_ids']=tokens[:2]
  for length in (0,1,2):
   for name,session in branches:
    record['phase']=f'forward/{length}/{name}';print(record['phase'],flush=True)
    step=session.next(tuple(tokens[:length]));torch.cuda.synchronize()
    row={'prefix_length':length,'branch':name,'logit_shape':list(step.logits.shape),'logits_finite':bool(np.isfinite(step.logits).all()),'raw_layers':len(step.early_raw),'normalized_layers':len(step.early_normalized),'all_layer_logits_finite':all(bool(np.isfinite(v).all()) for d in (step.early_raw,step.early_normalized) for v in d.values()),'layer_shapes':sorted({tuple(v.shape) for d in (step.early_raw,step.early_normalized) for v in d.values()})}
    record['visits'].append(row)
    if not row['logits_finite'] or not row['all_layer_logits_finite']:raise AssertionError('nonfinite forward')
    del step
  record['passed']=True;record['phase']='complete'
 except BaseException as e:
  record['error_type']=type(e).__name__;record['error']=str(e);record['traceback']=traceback.format_exc();print(record['traceback'],flush=True)
 finally:
  record['elapsed_seconds']=time.monotonic()-start
  if torch is not None and torch.cuda.is_initialized():
   record['peak_allocated_bytes']=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
   record['peak_reserved_bytes']=[torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]
  with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps({k:record.get(k) for k in ['stage','passed','phase','error_type','peak_allocated_bytes','elapsed_seconds']}),flush=True)
 return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
