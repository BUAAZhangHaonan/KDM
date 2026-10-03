"""Real independent16 engine audit; never admits a failed check or starts production."""
import argparse,json,math,os,sys,time,traceback
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import independent as impl
from kdm.io import atomic_json,file_hash

def cast_tree(x,dtype):
 import torch
 if torch.is_tensor(x):return x.to(dtype=dtype) if x.is_floating_point() else x
 if isinstance(x,dict):return {k:cast_tree(v,dtype) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cast_tree(v,dtype) for v in x]
 return x

def run(a):
 from vllm import SamplingParams
 from vllm.multimodal.processing import ProcessorInputs,TimingContext
 impl.worker_admission()
 dest=ROOT/a.out;dest.mkdir(parents=True,exist_ok=False)
 native=json.loads((ROOT/a.native_reference).read_text())
 assert native['schema']=='kdm_native_independent_reference_v1' and native['model']==a.model
 sc=impl.Generator(a.model);sc.audit_path=dest/'gemma_attention_ranges.json'
 report={'schema':'kdm_vllm_independent_reference_check_v1','model':a.model,'passed':False,
  'native_reference_sha256':file_hash(ROOT/a.native_reference),'implementation_sha256':file_hash(Path(impl.__file__)),
  'closed_preparer_sha256':file_hash(Path(impl.closed.__file__)),
  'versions':{p:version(p) for p in ('vllm','torch','transformers')},'rows':[],
  'eos_token_ids':native['eos_token_ids'],'sampling_cohort':'vllm_separate_from_numpy',
  'numerical_difference_review':'pending','pid':os.getpid(),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']}
 try:
  sc.load();proc=sc.llm.llm_engine.input_processor.renderer.mm_processor
  cfg=sc.llm.llm_engine.model_config
  fixed={x['id'] for x in impl.read_jsonl(ROOT/'data/current/interface16.jsonl')}
  assert len(native['rows'])==16 and {x['sample']['id'] for x in native['rows']}==fixed
  for row in native['rows']:
   s=row['sample'];image,prompt,base,inp=sc.prepare(s)
   assert prompt==row['prompt'] and base==row['unexpanded_prompt_ids']
   assert inp['input_ids'][0].tolist()==row['expanded_prompt_ids']
   assert all(impl.closed.summarize(inp[k])==impl.closed.core_summary(v) for k,v in row['processor_summary'].items())
   base=impl.engine_base_ids(a.model,base,row['expanded_prompt_ids'],sc.proc.tokenizer.bos_token_id)
   processed=proc.apply(ProcessorInputs(prompt=base,mm_data_items=proc.info.parse_mm_data({'image':image}),hf_processor_mm_kwargs={},tokenization_kwargs={}),TimingContext(enabled=False))
   actual=impl.closed.summarize(processed['mm_kwargs'].get_data(device='cpu',pin_memory=False))
   expected=impl.closed.summarize(cast_tree(dict(inp),cfg.dtype))
   required={'qwen2_5_vl':{'pixel_values','image_grid_thw'},'qwen3_5':{'pixel_values','image_grid_thw'},'llava_next':{'pixel_values','image_sizes'},'gemma3':{'pixel_values'},'minicpmv':{'pixel_values','tgt_sizes'}}[sc.mt]
   assert required.issubset(actual) and required.issubset(expected),('missing_vision_fields',required,set(actual),set(expected))
   if sc.mt=='minicpmv':
    import torch
    raw_mm=processed['mm_kwargs'].get_data(device='cpu',pin_memory=False)
    x=raw_mm['tgt_sizes'];y=inp['tgt_sizes']
    if isinstance(y,list):y=torch.stack(y)
    assert torch.equal(x.cpu().reshape(-1,2),y.cpu().reshape(-1,2)), 'MiniCPM target sizes differ'
    actual['tgt_sizes']=impl.closed.summarize(x.cpu().reshape(-1,2));expected['tgt_sizes']=impl.closed.summarize(y.cpu().reshape(-1,2))
   parity={k:impl.closed.core_summary(actual[k])==impl.closed.core_summary(expected[k]) for k in required}
   assert parity and all(parity.values()),('engine_processor',parity)
   assert processed['prompt_token_ids']==row['expanded_prompt_ids']
   req={'prompt_token_ids':base,'multi_modal_data':{'image':image},'multi_modal_uuids':{'image':s['id']}}
   p=impl.parameters(a.model,s['id'],0,native['eos_token_ids']);p['temperature']=0.0
   attempts=sc.generate_ten(s,native['eos_token_ids'])
   t=time.perf_counter();result=sc.llm.generate([req],SamplingParams(**p),use_tqdm=False)[0];greedy_wall=time.perf_counter()-t
   c=result.outputs[0];tokens=list(c.token_ids);lp=[float(v[t].logprob) for t,v in zip(tokens,c.logprobs)]
   old=row['greedy_selected_log_probabilities'];common=0
   for x,y in zip(tokens,row['greedy_tokens']):
    if x!=y:break
    common+=1
   item={'sample_id':s['id'],'engine_greedy_tokens':tokens,'native_greedy_tokens':row['greedy_tokens'],
    'engine_greedy_selected_log_probabilities':lp,'greedy_tokens_equal':tokens==row['greedy_tokens'],
    'max_abs_selected_logprob_error':max((abs(x-y) for x,y in zip(lp,old)),default=float('inf')) if tokens==row['greedy_tokens'] else None,
    'matched_prefix_tokens':common,'matched_prefix_max_abs_logprob_error':max((abs(lp[i]-old[i]) for i in range(common)),default=None),
    'engine_expanded_prompt_ids_equal':result.prompt_token_ids==row['expanded_prompt_ids'],
    'engine_processor_tensor_parity':parity,'actual_engine_processor_tensors':actual,
    'ten_distinct_stable_seeds':len({r['seed'] for r in attempts})==10,
    'native_eos_handling':all((r['tokens'][-1] in native['eos_token_ids'])==r['terminated'] and (r['terminated'] or len(r['tokens'])==32) for r in attempts),
    'finite_selected_logprobs':all(math.isfinite(x) for r in attempts for x in r['selected_log_probabilities']) and all(math.isfinite(x) for x in lp),
    'num_cached_tokens':sum(r['num_cached_tokens'] for r in attempts),'ten_request_wall_s':attempts[0]['wall_s'],
    'greedy_wall_s':greedy_wall,'unique_generated_sequences':len({tuple(r['tokens']) for r in attempts}),
    'distribution_scope':'raw selected-token log probabilities; not full-vocabulary equivalence'}
   report['rows'].append(item)
   with (dest/'attempts.jsonl').open('a') as f:
    for r in attempts:f.write(json.dumps(r)+'\n')
   atomic_json(dest/'partial.json',report)
   print(json.dumps({'sample':s['id'],'completed':len(report['rows']),'greedy_equal':item['greedy_tokens_equal'],'max_logprob_difference':item['max_abs_selected_logprob_error'],'ten_wall_s':item['ten_request_wall_s'],'cached':item['num_cached_tokens']}),flush=True)
  if a.model=='gemma3_4b':report['gemma_attention_audit']={**sc.attention_audit,'module_sha256':file_hash(ROOT/'workflows/acceleration_v4/gemma_audit_worker.py')}
  report['checks_passed']=all(r['greedy_tokens_equal'] and r['engine_expanded_prompt_ids_equal'] and r['ten_distinct_stable_seeds'] and r['native_eos_handling'] and r['finite_selected_logprobs'] and r['num_cached_tokens']>0 for r in report['rows'])
  report['ten_answer_seconds']=sum(r['ten_request_wall_s'] for r in report['rows'])
  report['answers_per_second']=160/report['ten_answer_seconds']
  report['throughput_scope']='inference_only; excludes prepare and disk writes; production elapsed progress required for ETA'
  atomic_json(dest/'check.json',report)
  print(json.dumps({'checks_passed':report['checks_passed'],'answers_per_second':report['answers_per_second'],'review_required':True}),flush=True)
 except BaseException as e:
  atomic_json(dest/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'retry_performed':False});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--native-reference',required=True);p.add_argument('--out',required=True);run(p.parse_args())
