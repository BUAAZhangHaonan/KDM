#!/usr/bin/env python3
"""MiniCPM-V-4_5 separate vLLM cohort; native CPU input evidence plus actual engine audit."""
import argparse,hashlib,json,os,sys,time,traceback
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kdm.io import atomic_json,file_hash,read_jsonl
from workflows.remaining11_minicpm45_v1 import closed_scorer as core
core.PATHS['minicpm45']='/home/team/lvshuyang/Models/MiniCPM-V-4_5'
BASE=ROOT/'outputs/records/remaining11_minicpm45_v1/engine'
REF=ROOT/'outputs/records/remaining11_minicpm45_v1/native_closed16_reference.json'
MODEL='minicpm45'
def export():
 from importlib.metadata import version
 specpath=ROOT/'configs/runtime/minicpm45.json';spec=json.loads(specpath.read_text())
 actual={k:version(k) for k in spec['versions']}
 assert actual==spec['versions'],(actual,spec['versions'])
 src=ROOT/'outputs/raw/probes_v2/food_closed_20260923_v3/minicpm45/closed.jsonl'
 identity=json.loads(src.with_name('closed.identity.json').read_text())
 assert identity['definition']['backend_spec_sha256']==file_hash(specpath)
 sc=core.Scorer(MODEL)
 rows=[]
 for r in list(read_jsonl(src))[:16]:
  assert r['identity']==identity['identity'] and len(r['candidate_scores'])==101
  assert [s['label'] for s in r['candidate_scores']]==sc.names
  assert [s['n_tokens'] for s in r['candidate_scores']]==list(map(len,sc.tokens))
  _,prompt,base,inp=sc.prepare(r['sample']);assert prompt==r['prompt']
  rows.append({'reference':r,'unexpanded_prompt_ids':base,'expanded_prompt_ids':inp['input_ids'][0].tolist(),'processor_summary':core.summarize(dict(inp)),'candidates_tokens':sc.tokens})
 assert len(rows)==16
 atomic_json(REF,{'model':MODEL,'backend':spec,'backend_spec_sha256':file_hash(specpath),'source_blobs':identity['definition']['source_blobs'],'source_raw_sha256':file_hash(src),'source_identity_sha256':file_hash(src.with_name('closed.identity.json')),'source_identity':identity,'native_cpu_export_versions':actual,'rows':rows,'scope':'native processor inputs for 16 previously measured real TF closed scores; no new native inference'})
 print('NATIVE_CPU_REFERENCE_COMPLETE',flush=True)
def engine_audit():
 from vllm.config import ModelConfig
 from vllm.multimodal import MULTIMODAL_REGISTRY
 from vllm.multimodal.processing import ProcessorInputs,TimingContext
 from PIL import Image
 import torch
 ref=json.loads(REF.read_text());sc=core.Scorer(MODEL)
 cfg=ModelConfig(model=sc.path,dtype='bfloat16',trust_remote_code=True,max_model_len=40960,limit_mm_per_prompt={'image':1,'video':0})
 proc=MULTIMODAL_REGISTRY.create_processor(cfg,cache=None);checks=[]
 for d in ref['rows']:
  im,_,base,inp=sc.prepare(d['reference']['sample'])
  assert base==d['unexpanded_prompt_ids'] and inp['input_ids'][0].tolist()==d['expanded_prompt_ids']
  expected=d['processor_summary']
  assert all(core.summarize(inp[k])==core.core_summary(v) for k,v in expected.items())
  assert len(d['expanded_prompt_ids'])+max(map(len,sc.tokens))<=40960, 'Native context40960 exceeded, no truncation'
  items=proc.info.parse_mm_data({'image':im})
  engine_base=base
  bos=sc.proc.tokenizer.bos_token_id
  if d['expanded_prompt_ids'][0]==bos and base[0]!=bos: engine_base=[bos]+base
  out=proc.apply(ProcessorInputs(prompt=engine_base,mm_data_items=items,hf_processor_mm_kwargs={},tokenization_kwargs={}),TimingContext(enabled=False))
  assert out['prompt_token_ids']==d['expanded_prompt_ids']
  mm=out['mm_kwargs'].get_data(device='cpu',pin_memory=False)
  comparisons={}
  for k,v in mm.items():
   assert k in inp,'Unmatched actual multimodal input '+k
   native=inp[k]
   if torch.is_tensor(native) and native.is_floating_point():native=native.to(cfg.dtype)
   comparisons[k]=core.summarize(v)==core.summarize(native)
   assert comparisons[k],'Actual engine tensor differs '+k
  assert 'pixel_values' in comparisons and comparisons
  checks.append({'sample_id':d['reference']['sample']['id'],'expanded_prompt_equal':True,'engine_native_bf16_tensors_equal':comparisons})
 atomic_json(BASE/'actual_processor_audit.json',{'passed':True,'reference_sha256':file_hash(REF),'entry_sha256':file_hash(__file__),'processor_class':type(proc).__name__,'rows':checks,'scope':'actual vLLM multimodal tensors match native after declared BF16 cast'})
 print('ACTUAL_ENGINE_PROCESSOR_PARITY_PASSED',flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');a=p.parse_args()
 BASE.mkdir(parents=True,exist_ok=True)
 if a.export:raise RuntimeError('Use4028 export_minicpm45_v1.py for original native environment')
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
  assert Path(os.readlink('/proc/self/fd/20'))==ROOT/'outputs/locks/gpu_0.lock'
  engine_audit()
  sys.argv=[str(__file__),'--model',MODEL,'--reference',str(REF.relative_to(ROOT)),'--run','remaining11_minicpm45_closed_20260924_v1','--execute']
  core.main()
 except BaseException as e:
  atomic_json(BASE/'entry_error.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
if __name__=='__main__':main()
