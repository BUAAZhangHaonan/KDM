#!/usr/bin/env python3
"""Native independent-prompt fixed16 export with explicit relocated execution identity."""
import argparse,copy,json,math,os,sys,time,traceback
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows/food_closed_v3'),str(ROOT/'workflows/acceleration_v4')]
import admission
from kdm.io import atomic_json,file_hash,read_jsonl,within
from kdm.protocol import validate_freeze
from kdm.pipeline import make_backend
from kdm.execution import validate_host,resolve_image_path
from kdm.prompts import task_prompt
from vllm_closed_v3 import PATHS,summarize
from PIL import Image

def template(backend,image,prompt):
 em=backend.em
 if em.mt=='minicpmv':
  return em.proc.tokenizer.apply_chat_template([{'role':'user','content':'(<image>./</image>)\n'+prompt}],tokenize=False,add_generation_prompt=True)
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
 kw={'tokenize':False,'add_generation_prompt':True}
 if em.mt in ('qwen3_5','qwen3_vl','glm4v'):kw['enable_thinking']=False
 return em.proc.apply_chat_template(messages,**kw)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=PATHS,required=True);ap.add_argument('--out',required=True)
 args=ap.parse_args();out=within(ROOT,args.out);out.mkdir(parents=True,exist_ok=False)
 status={'model':args.model,'pid':os.getpid(),'status':'loading','completed':0,'expected':16,'started_unix':time.time()}
 atomic_json(out/'progress.json',status)
 try:
  cards=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
  if cards!=['0']:raise ValueError('This exporter is authorized on physical6403GPU0 only')
  host,host_info=validate_host(ROOT,cards)
  if host!='6403':raise ValueError('Native export relocated host mismatch')
  if Path(os.readlink('/proc/self/fd/20'))!=ROOT/'outputs/locks/gpu_0.lock':raise ValueError('Missing inherited approved GPU lock')
  specpath=ROOT/'configs/runtime'/f'{args.model}.json';spec=json.loads(specpath.read_text())
  observed={k:version(k) for k in spec['versions']}
  if observed!=spec['versions']:raise ValueError('Exact source package versions required: '+str({k:(spec['versions'][k],v) for k,v in observed.items() if spec['versions'][k]!=v}))
  expected_env='native311' if args.model in ('qwen35_4b','llava16_mistral') else 'mprisk-tf553'
  if Path(sys.executable)!=ROOT/'.environments'/expected_env/'bin/python':raise ValueError('Unexpected relocated Python executable')
  checkpoint=Path(PATHS[args.model]);checks={}
  for item in spec['weights']:
   checks[item['filename']]=(checkpoint/item['filename']).stat().st_size==item['size_bytes']
  checks['config.json']=file_hash(checkpoint/'config.json')==spec['model_config_sha256']
  checks.update({n:file_hash(checkpoint/n)==h for n,h in spec['processor']['files'].items()})
  if not all(checks.values()):raise ValueError('Checkpoint/processor identity differs: '+str(checks))
  for name,digest in spec.get('adapter_source_sha256',{}).items():
   if file_hash(ROOT/'src/kdm/models'/name)!=digest:raise ValueError('Native adapter changed: '+name)
  freeze=validate_freeze(ROOT)
  runtime=copy.deepcopy(spec);runtime['kwargs']['model_path']=str(checkpoint);runtime['environment_python']=sys.executable
  samples=list(read_jsonl(ROOT/'data/current/interface16.jsonl'))
  if len(samples)!=16 or len({s['id'] for s in samples})!=16:raise ValueError('Exact fixed16 required')
  backend=make_backend(runtime,'cuda:0')
  result={'schema':'kdm_native_independent_reference_v1','model':args.model,'backend':spec,'backend_spec_sha256':file_hash(specpath),
   'source_blobs':freeze['source_blobs'],'manifest_sha256':file_hash(ROOT/'data/current/all.jsonl'),'interface16_sha256':file_hash(ROOT/'data/current/interface16.jsonl'),
   'workflow_sha256':file_hash(__file__),'eos_token_ids':sorted(backend.eos),
   'execution':{'host':host,'hostname':host_info['hostname'],'physical_gpus':cards,'gpu_uuids':{c:host_info['gpu_uuids'][c] for c in cards},
    'environment_python':sys.executable,'versions':observed,'checkpoint_path':str(checkpoint),'checkpoint_identity_checks':checks,
    'precision':'bfloat16','scope':'relocated exact package versions, original checkpoint and adapter; real GPU reference measured here'},
   'checkpoint_generation_config':backend.model.generation_config.to_dict(),'rows':[]}
  status['status']='running';atomic_json(out/'progress.json',status)
  for sample in samples:
   started=time.perf_counter();prompt=task_prompt(sample['question'],guided=False,attempt=True)
   with Image.open(resolve_image_path(sample['image_path'],ROOT)) as image_source:image=image_source.convert('RGB')
   text=template(backend,image,prompt);inputs=backend.em.build(image,prompt)
   summary=summarize(inputs);expanded=inputs['input_ids'][0].tolist()
   backend.torch.cuda.synchronize();start_native=time.perf_counter()
   with backend.torch.inference_mode():
    if hasattr(backend.em,'native_generate'):
     native=backend.em.native_generate(inputs,max_new_tokens=32)
    else:
     generated=backend.model.generate(**inputs,do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=32,eos_token_id=sorted(backend.eos),pad_token_id=backend.tokenizer.pad_token_id or backend.tokenizer.eos_token_id)
     native=generated[0,inputs['input_ids'].shape[-1]:].tolist()
   backend.torch.cuda.synchronize();native_wall=time.perf_counter()-start_native
   if not 1<=len(native)<=32 or any(t in backend.eos for t in native[:-1]) or (native[-1] not in backend.eos and len(native)!=32):raise ValueError('Native EOS/budget invalid')
   session=backend.session(image,prompt);prefix=[];selected=[];argmax=[]
   for token in native:
    logits=session.next(prefix).logits
    z=backend.torch.from_numpy(logits).double();lp=float(backend.torch.log_softmax(z,dim=-1)[token])
    if not math.isfinite(lp):raise ValueError('Nonfinite selected raw log probability')
    argmax.append(int(logits.argmax()));selected.append(lp);prefix.append(token)
   row={'sample':sample,'prompt':prompt,'unexpanded_prompt_text':text,'unexpanded_prompt_ids':backend.encode(text),'expanded_prompt_ids':expanded,
    'processor_summary':summary,'greedy_tokens':native,'greedy_selected_log_probabilities':selected,'text':backend.decode(native),
    'terminated':native[-1] in backend.eos,'native_greedy_wall_s':native_wall,'wall_s':time.perf_counter()-started,
    'native_backend_matched_prefix_argmax':argmax,'native_backend_greedy_equal':argmax==native,
    'logprob_source':'original native backend full-vocabulary logits at every native-generated prefix; FP64 log_softmax',
    'image_sha256':file_hash(resolve_image_path(sample['image_path'],ROOT))}
   result['rows'].append(row);atomic_json(out/'partial.json',result)
   status.update(completed=len(result['rows']),last_sample_id=sample['id'],updated_unix=time.time());atomic_json(out/'progress.json',status)
   print(json.dumps({'model':args.model,'completed':status['completed'],'equal':argmax==native,'native_wall_s':native_wall}),flush=True)
   if argmax!=native:raise ValueError('Actual native generation and frozen decoder greedy differ')
   del session,inputs;backend.torch.cuda.empty_cache()
  result.update(passed=True,completed=16,peak_allocated_bytes=backend.torch.cuda.max_memory_allocated(),finished_unix=time.time())
  atomic_json(out/'reference.json',result);atomic_json(out/'complete.json',{'complete':True,'reference_sha256':file_hash(out/'reference.json'),'rows':16})
  status.update(status='complete',finished_unix=time.time());atomic_json(out/'progress.json',status)
 except BaseException as exc:
  atomic_json(out/'error.json',{'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'completed':status['completed']})
  status.update(status='failed',error=str(exc));atomic_json(out/'progress.json',status);raise
if __name__=='__main__':main()
