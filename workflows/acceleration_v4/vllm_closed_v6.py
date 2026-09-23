#!/usr/bin/env python3
"""Batched prefix-tree scoring using raw next-token probabilities and native labels."""
import argparse,json,os,sys,time,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from kdm.io import Ledger,atomic_json,file_hash,stable_hash,read_jsonl
from kdm.execution import resolve_image_path
from kdm.prompts import closed_prompt
PATHS={'qwen25vl':'/home/team/lvshuyang/Models/Qwen2.5-VL-7B-Instruct',
'qwen35_4b':'/home/team/lvshuyang/Models/Qwen3.5-4B',
'gemma3_4b':'/home/team/lvshuyang/Models/gemma-3-4b-it',
'minicpm26':'/home/team/lvshuyang/Models/MiniCPM-V-2_6',
'llava16_mistral':str(ROOT/'cache/models/llava-v1.6-mistral-7b-hf')}
def summarize(v):
 import torch
 if torch.is_tensor(v):
  x=v.detach().cpu().contiguous()
  return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
 if isinstance(v,dict):return {str(k):summarize(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [summarize(x) for x in v]
 return v if v is None or isinstance(v,(str,int,float,bool)) else {'type':type(v).__name__}
def core_summary(v):
 if isinstance(v,dict):
  if 'sha256' in v:return {k:v[k] for k in ('shape','dtype','sha256')}
  return {k:core_summary(x) for k,x in v.items()}
 if isinstance(v,list):return [core_summary(x) for x in v]
 return v
class Scorer:
 def __init__(self,model):
  from transformers import AutoProcessor,AutoConfig
  self.model=model;self.path=PATHS[model];self.config=AutoConfig.from_pretrained(self.path,trust_remote_code=True)
  self.proc=AutoProcessor.from_pretrained(self.path,trust_remote_code=True)
  self.mt=self.config.model_type
  if self.mt=='minicpmv':
   tok=self.proc.tokenizer
   strings={'im_start':'<image>','im_end':'</image>','ref_start':'<ref>','ref_end':'</ref>','box_start':'<box>','box_end':'</box>','quad_start':'<quad>','quad_end':'</quad>','slice_start':'<slice>','slice_end':'</slice>','im_id_start':'<image_id>','im_id_end':'</image_id>'}
   for key,value in strings.items():
    assert getattr(tok,key)==value, 'MiniCPM special token changed: '+key
   ids={'im_start_id':'<image>','im_end_id':'</image>','slice_start_id':'<slice>','slice_end_id':'</slice>','im_id_start_id':'<image_id>','im_id_end_id':'</image_id>','newline_id':'\n'}
   for key,value in ids.items():
    assert getattr(tok,key)==int(tok.convert_tokens_to_ids(value)), 'MiniCPM token id changed: '+key
   for key in ('bos','eos','unk'):
    assert getattr(tok,key+'_id')==int(getattr(tok,key+'_token_id'))
  self.names=sorted(json.loads((ROOT/'configs/kdm/food_aliases.json').read_text()))
  self.tokens=[self.proc.tokenizer.encode(n.replace('_',' '),add_special_tokens=False) for n in self.names]
  self.nodes={}
  for seq in self.tokens:
   for j,t in enumerate(seq):self.nodes.setdefault(tuple(seq[:j]),set()).add(t)
  assert all(len(v)<=128 for v in self.nodes.values())
 def prepare(self,sample):
  im=Image.open(resolve_image_path(sample['image_path'],ROOT)).convert('RGB')
  prompt=closed_prompt(sample['question'],self.names)
  if self.mt=='minicpmv':
   text=self.proc.tokenizer.apply_chat_template([{'role':'user','content':'(<image>./</image>)\n'+prompt}],tokenize=False,add_generation_prompt=True)
   inp=self.proc(text=[text],images=[[im]],return_tensors='pt')
   mask=inp['attention_mask'];pos=mask.long().cumsum(-1)-1;inp['position_ids']=pos.masked_fill(mask==0,0)
  else:
   msgs=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
   kw={'add_generation_prompt':True}
   if self.mt in ('qwen3_5','qwen3_vl','glm4v'):kw['enable_thinking']=False
   text=self.proc.apply_chat_template(msgs,tokenize=False,**kw)
   inp=self.proc.apply_chat_template(msgs,tokenize=True,return_dict=True,return_tensors='pt',**kw)
  return im,prompt,self.proc.tokenizer.encode(text,add_special_tokens=False),inp
 def load(self):
  from vllm import LLM
  self.llm=LLM(model=self.path,dtype='bfloat16',trust_remote_code=True,tensor_parallel_size=1,
   gpu_memory_utilization=0.80,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=8192,
   enable_prefix_caching=True,enforce_eager=True,limit_mm_per_prompt={'image':1,'video':0},
   logprobs_mode='raw_logprobs',max_logprobs=128,seed=0,disable_log_stats=False,
   skip_mm_profiling=True,attention_config=({'backend':'TRITON_ATTN'} if self.mt=='gemma3' else {}),
   worker_cls=('workflows.acceleration_v4.gemma_audit_worker.GemmaAuditWorker' if self.mt=='gemma3' else 'auto'))
 def score(self,sample):
  from vllm import SamplingParams
  t=time.perf_counter();im,prompt,base,inp=self.prepare(sample);expanded=inp['input_ids'][0].tolist()
  if self.mt=='llava_next':
   bos=self.proc.tokenizer.bos_token_id
   if expanded and expanded[0]==bos and base and base[0]!=bos:base=[bos]+base
  values={};cached=0;requested=0
  for depth in range(max(map(len,self.tokens))):
   prefixes=[p for p in self.nodes if len(p)==depth]
   requests=[{'prompt_token_ids':base+list(p),'multi_modal_data':{'image':im},'multi_modal_uuids':{'image':sample['id']}} for p in prefixes]
   params=[SamplingParams(max_tokens=1,temperature=0,logprob_token_ids=sorted(self.nodes[p]),ignore_eos=True) for p in prefixes]
   for prefix,res in zip(prefixes,self.llm.generate(requests,params,use_tqdm=False)):
    assert res.prompt_token_ids==expanded+list(prefix),'Engine prompt expansion changed'
    cached+=getattr(res,'num_cached_tokens',0) or 0;requested+=len(res.prompt_token_ids)
    lp=res.outputs[0].logprobs[0]
    for tok in self.nodes[prefix]:
     value=float(lp[tok].logprob)
     if not math.isfinite(value):raise ValueError('Nonfinite class score')
     values[(prefix,tok)]=value
  scores=[]
  for name,tokens in zip(self.names,self.tokens):
   ls=[values[(tuple(tokens[:j]),tok)] for j,tok in enumerate(tokens)]
   scores.append({'label':name,'sum_logp':sum(ls),'mean_logp':float(np.mean(ls)),'n_tokens':len(tokens)})
  gold=next(r['mean_logp'] for r in scores if r['label']==sample['class'])
  return {'status':'ok','model':self.model,'sample':sample,'prompt':prompt,'target':sample['class'],
   'candidate_scores':scores,'gold_rank':1+sum(r['mean_logp']>gold for r in scores),'ranking_rule':'mean_log_probability',
   'wall_s':time.perf_counter()-t,'num_cached_tokens':cached,'requested_prompt_tokens':requested}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=PATHS);p.add_argument('--reference',required=True);p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true')
 a=p.parse_args();run=ROOT/'outputs/records/acceleration_v4'/a.run/a.model
 run.mkdir(parents=True,exist_ok=False)
 try:
  ref=json.loads((ROOT/a.reference).read_text());rows=ref['rows'];sc=Scorer(a.model)
  # The original declared model weights/config/processor must be unchanged.
  spec=ref.get('backend',json.loads((ROOT/f'configs/runtime/{a.model}.json').read_text()))
  for item in spec['weights']:
   if (Path(sc.path)/item['filename']).stat().st_size!=item['size_bytes']:raise ValueError('Weight size mismatch')
  if file_hash(Path(sc.path)/'config.json')!=spec['model_config_sha256']:raise ValueError('Model config mismatch')
  for name,digest in spec['processor']['files'].items():
   if file_hash(Path(sc.path)/name)!=digest:raise ValueError('Processor file changed: '+name)
  for d in rows:
   _,_,base,inp=sc.prepare(d['reference']['sample'])
   assert base==d['unexpanded_prompt_ids'],'Template/tokenizer changed'
   assert inp['input_ids'][0].tolist()==d['expanded_prompt_ids'],'Expanded processor input changed'
   tokens=d['candidates_tokens'];expected=[tokens[n] for n in sc.names] if isinstance(tokens,dict) else tokens
   assert sc.tokens==expected,'Class tokenization changed'
   for k,v in d['processor_summary'].items():
    assert k in inp and summarize(inp[k])==core_summary(v),'Processor tensor changed: '+k
  atomic_json(run/'preflight.json',{'input_parity':True,'rows':len(rows),'reference_sha256':file_hash(ROOT/a.reference)})
  if a.model=='gemma3_4b':os.environ['KDM_ATTN_AUDIT_PATH']=str(run/'attention_mask_audit.json')
  sc.load();checks=[]
  for d in rows:
   r=sc.score(d['reference']['sample']);old=d['reference'];x=np.array([s['mean_logp'] for s in old['candidate_scores']]);y=np.array([s['mean_logp'] for s in r['candidate_scores']])
   checks.append({'sample_id':r['sample']['id'],'old_wall_s':old['wall_s'],'wall_s':r['wall_s'],'top1_equal':bool(np.argmax(x)==np.argmax(y)),
    'gold_rank':r['gold_rank'],'old_gold_rank':old['gold_rank'],'max_mean_logp_error':float(np.max(np.abs(x-y))),'mean_abs_error':float(np.mean(np.abs(x-y))),
    'num_cached_tokens':r['num_cached_tokens'],'requested_prompt_tokens':r['requested_prompt_tokens'],'new':r})
  eligible=all(x['top1_equal'] and ((x['gold_rank']==1)==(x['old_gold_rank']==1)) for x in checks)
  proof={'input_parity':True,'finite_101_class_scores':True,'top1_parity_on_reference':eligible,'numerically_identical':False,
   'scope':'small engineering reference only; vLLM results form a separate complete backend cohort, not mixed with old scores','checks':checks,
   'speedup':sum(x['old_wall_s'] for x in checks)/sum(x['wall_s'] for x in checks)}
  if a.model=='gemma3_4b':
   mask=json.loads((run/'attention_mask_audit.json').read_text())
   assert mask['backend']=='TRITON_ATTN' and mask['nonempty_real_ranges'] and mask['num_actual_tokens']>0
   assert mask['module_sha256']==file_hash(ROOT/'workflows/acceleration_v4/gemma_audit_worker.py')
   proof['attention_metadata_audit_sha256']=file_hash(run/'attention_mask_audit.json')
  atomic_json(run/'benchmark.json',proof)
  if not eligible:raise ValueError('Reference top1 differs; stop for discrepancy review, no production launch')
  if not a.execute:print(json.dumps({'model':a.model,'benchmark_complete':True,'speedup':proof['speedup']}),flush=True);return
  import vllm,torch,transformers
  identity={'schema':'kdm_vllm_tree_closed_v4','model':a.model,'backend_spec_sha256':ref['backend_spec_sha256'],'checkpoint_path':sc.path,
   'source_blobs':(ref['source_blobs'] if 'source_blobs' in ref else ref['source_identity']['definition']['source_blobs']),'native_reference_sha256':file_hash(ROOT/a.reference),'benchmark_sha256':file_hash(run/'benchmark.json'),
   'implementation_sha256':file_hash(Path(__file__)),'manifest_sha256':file_hash(ROOT/'data/current/all.jsonl'),
   'versions':{'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__},'physical_gpus':os.environ['CUDA_VISIBLE_DEVICES'],
   'dtype':'bfloat16','ranking':'mean original token log probabilities, no EOS','precision_note':'separate vLLM numerical cohort; no old-row reuse'}
  output=ROOT/'outputs/raw/acceleration_v4'/a.run/a.model/'closed.jsonl'
  if output.exists():raise ValueError('Fresh output required')
  ledger=Ledger(output,identity);food=[s for s in read_jsonl(ROOT/'data/current/all.jsonl') if s['dataset']=='food101']
  assert len(food)==4848
  progress={'model':a.model,'pid':os.getpid(),'status':'running','expected':4848,'closed':0,'output':str(output.relative_to(ROOT))}
  validated={x['new']['sample']['id']:x['new'] for x in checks}
  for sample in food:
   r=validated.pop(sample['id'],None)
   if r is None:r=sc.score(sample)
   ledger.add(stable_hash([a.model,sample['id'],'closed']),r)
   progress.update(closed=len(ledger.keys),last_sample_id=sample['id'],last_wall_s=r['wall_s'],updated_unix=time.time())
   atomic_json(run/'progress.json',progress)
  seen=set()
  for r in read_jsonl(output):
   if r['sample']['id'] in seen or r['identity']!=ledger.identity or len(r['candidate_scores'])!=101:raise ValueError('Final coverage invalid')
   seen.add(r['sample']['id'])
  assert seen=={s['id'] for s in food}
  atomic_json(run/'complete.json',{'complete':True,'closed':4848,'identity':identity,'output_sha256':file_hash(output)})
  progress['status']='complete';atomic_json(run/'progress.json',progress)
 except BaseException as e:
  import traceback
  atomic_json(run/'error.json',{'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
