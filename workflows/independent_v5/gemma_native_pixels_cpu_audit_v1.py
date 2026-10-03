"""CPU-only actual vLLM multimodal processor audit for the native pixel bridge."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
os.environ['CUDA_VISIBLE_DEVICES']=''
from kdm.io import atomic_json,file_hash
import independent as impl
import gemma_native_pixels_v1 as bridge_module

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out
 if out.exists():raise ValueError('Fresh CPU receipt required')
 out.parent.mkdir(parents=True,exist_ok=True)
 report={'schema':'kdm_gemma_native_pixels_cpu_audit_v1','cpu_only':True,'gpu_inference_performed':False,'script_sha256':file_hash(__file__),'rows':[]}
 bridge=None
 try:
  started=time.perf_counter();bridge=bridge_module.install();report['native_bridge_identity']=bridge.identity
  from vllm.config import ModelConfig
  from vllm.multimodal import MULTIMODAL_REGISTRY
  from vllm.multimodal.processing import ProcessorInputs,TimingContext
  cfg=ModelConfig(model=str(bridge_module.CHECKPOINT),dtype='bfloat16',trust_remote_code=True,max_model_len=32768,limit_mm_per_prompt={'image':1,'video':0})
  processor=MULTIMODAL_REGISTRY.create_processor(cfg,cache=None);sc=impl.Generator('gemma3_4b')
  ref_path=ROOT/'outputs/records/independent_v5/native16_v2/gemma3_4b/reference.json';ref=json.loads(ref_path.read_text());report['native_reference_sha256']=file_hash(ref_path)
  for row in ref['rows']:
   t=time.perf_counter();before=bridge.calls;image,prompt,base,inp=sc.prepare(row['sample'])
   external={k:impl.closed.summarize(inp[k])==impl.closed.core_summary(v) for k,v in row['processor_summary'].items()}
   assert prompt==row['prompt'] and base==row['unexpanded_prompt_ids'] and inp['input_ids'][0].tolist()==row['expanded_prompt_ids']
   assert all(external.values()),external
   after_external=bridge.calls
   result=processor.apply(ProcessorInputs(prompt=base,mm_data_items=processor.info.parse_mm_data({'image':image}),hf_processor_mm_kwargs={},tokenization_kwargs={}),TimingContext(enabled=False))
   actual=result['mm_kwargs'].get_data(device='cpu',pin_memory=False)
   expected=inp['pixel_values'].to(dtype=cfg.dtype)
   engine=impl.closed.summarize(actual['pixel_values'])==impl.closed.summarize(expected)
   assert engine and result['prompt_token_ids']==row['expanded_prompt_ids']
   assert after_external>before and bridge.calls>after_external
   report['rows'].append({'sample_id':row['sample']['id'],'external_native_tensor_parity':external,'actual_engine_native_pixel_parity':engine,'expanded_tokens_equal':True,'native_worker_calls_external':after_external-before,'native_worker_calls_engine':bridge.calls-after_external,'wall_s':time.perf_counter()-t,'actual_engine_pixels':impl.closed.summarize(actual['pixel_values'])})
  report.update(passed=True,count=len(report['rows']),wall_s=time.perf_counter()-started,native_worker_calls=bridge.calls,worker_pid=bridge.process.pid)
 except BaseException as e:
  report.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());raise
 finally:
  if bridge is not None:bridge.close()
  atomic_json(out,report)
 print(json.dumps({'passed':report['passed'],'rows':len(report['rows']),'calls':report.get('native_worker_calls'),'wall_s':report.get('wall_s')}))
if __name__=='__main__':main()
