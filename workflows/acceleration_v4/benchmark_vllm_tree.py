#!/usr/bin/env python3
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from kdm.execution import resolve_image_path
from transformers import AutoProcessor
from vllm import LLM,SamplingParams
def main():
 data=json.loads((ROOT/'outputs/records/acceleration_v4/qwen3vl_reference16.json').read_text())
 model_path='/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct'
 proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
 # Verify new processor inputs and exact candidate tokenization before GPU execution.
 for d in data:
  row=d['reference'];im=Image.open(resolve_image_path(row['sample']['image_path'],ROOT)).convert('RGB')
  msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':row['prompt']}]}]
  inp=proc.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
  assert inp['input_ids'][0].tolist()==d['expanded_prompt_ids'],'Prompt token mismatch'
  assert hashlib.sha256(inp['pixel_values'].numpy().tobytes()).hexdigest()==d['pixel_sha256'],'Processor pixels changed'
  assert [proc.tokenizer.encode(x['label'].replace('_',' '),add_special_tokens=False) for x in row['candidate_scores']]==d['candidates_tokens']
 print('PROCESSOR_PARITY_PASSED',flush=True)
 start=time.perf_counter()
 llm=LLM(model=model_path,dtype='bfloat16',trust_remote_code=True,tensor_parallel_size=1,
  gpu_memory_utilization=0.80,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=8192,
  enable_prefix_caching=True,enforce_eager=True,limit_mm_per_prompt={'image':1,'video':0},
  logprobs_mode='raw_logprobs',max_logprobs=128,seed=0,disable_log_stats=False)
 results=[]
 for idx,d in enumerate(data):
  start_row=time.perf_counter();row=d['reference'];sample=row['sample']
  im=Image.open(resolve_image_path(sample['image_path'],ROOT)).convert('RGB')
  tokens=d['candidates_tokens'];nodes={}
  for seq in tokens:
   for j,t in enumerate(seq):nodes.setdefault(tuple(seq[:j]),set()).add(t)
  measured={};cached=0;prompt_total=0
  for depth in range(max(map(len,tokens))):
   prefixes=[x for x in nodes if len(x)==depth]
   prompts=[{'prompt_token_ids':d['unexpanded_prompt_ids']+list(prefix),'multi_modal_data':{'image':im},'multi_modal_uuids':{'image':sample['id']}} for prefix in prefixes]
   params=[SamplingParams(max_tokens=1,temperature=0,logprob_token_ids=sorted(nodes[p]),ignore_eos=True) for p in prefixes]
   outputs=llm.generate(prompts,params,use_tqdm=False)
   for prefix,res in zip(prefixes,outputs):
    expected=d['expanded_prompt_ids']+list(prefix)
    assert res.prompt_token_ids==expected,'Engine prompt expansion mismatch'
    cache=getattr(res,'num_cached_tokens',0);cached+=cache or 0;prompt_total+=len(expected)
    probs=res.outputs[0].logprobs[0]
    for tok in nodes[prefix]:
     assert tok in probs,f'Missing requested logprob {tok}'
     measured[(prefix,tok)]=float(probs[tok].logprob)
  scores=[]
  for old,seq in zip(row['candidate_scores'],tokens):
   ls=[measured[(tuple(seq[:j]),t)] for j,t in enumerate(seq)]
   scores.append({'label':old['label'],'n_tokens':len(seq),'sum_logp':sum(ls),'mean_logp':float(np.mean(ls))})
  old=np.array([x['mean_logp'] for x in row['candidate_scores']]);new=np.array([x['mean_logp'] for x in scores])
  gold=[x['label'] for x in scores].index(sample['class'])
  rank=1+int(np.sum(new>new[gold]))
  result={'sample_id':sample['id'],'wall_s':time.perf_counter()-start_row,'old_wall_s':row['wall_s'],
   'max_mean_logp_error':float(np.max(np.abs(new-old))),'mean_abs_error':float(np.mean(np.abs(new-old))),
   'top1_equal':int(np.argmax(old))==int(np.argmax(new)),'gold_rank':rank,'old_gold_rank':row['gold_rank'],
   'num_cached_tokens':cached,'requested_prompt_tokens':prompt_total,'prefix_nodes':len(nodes),'candidate_scores':scores}
  results.append(result)
  out=ROOT/'outputs/records/acceleration_v4/qwen3vl_vllm_tree_benchmark.json'
  out.write_text(json.dumps({'model':'qwen3vl','rows':results,'processor_parity':True,'full_16_complete':len(results)==16,'load_and_run_s':time.perf_counter()-start},indent=2))
  print(json.dumps({k:v for k,v in result.items() if k!='candidate_scores'}),flush=True)
 if len(results)==16:
  print('BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
