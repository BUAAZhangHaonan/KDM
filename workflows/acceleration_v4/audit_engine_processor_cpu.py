#!/usr/bin/env python3
"""CPU-only audit of vLLM's actual multimodal processor versus native reference."""
import argparse,copy,hashlib,json,os,sys,traceback
from pathlib import Path
from dataclasses import asdict,is_dataclass
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''
from kdm.io import atomic_json,file_hash
from kdm.execution import resolve_image_path
from PIL import Image
def summary(x):
 import torch
 if torch.is_tensor(x):
  t=x.detach().cpu().contiguous()
  d={'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
  if t.numel()<=2048 and not t.is_floating_point():d['values']=t.tolist()
  return d
 if isinstance(x,dict):return {str(k):summary(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [summary(v) for v in x]
 if is_dataclass(x):return summary(asdict(x))
 return x if x is None or isinstance(x,(str,int,float,bool)) else str(x)
def core(x):
 if isinstance(x,dict):
  if 'sha256' in x:return {k:x[k] for k in ['shape','dtype','sha256']}
  return {k:core(v) for k,v in x.items()}
 if isinstance(x,list):return [core(v) for v in x]
 return x
def ranges(vals):
 out=[];start=None
 for i,v in enumerate(vals+[0]):
  if v==1 and start is None:start=i
  if v!=1 and start is not None:out.append([start,i]);start=None
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['gemma3_4b','qwen35_4b','qwen25vl','minicpm26','llava16_mistral']);a=ap.parse_args()
 from vllm.config import ModelConfig
 from vllm.multimodal import MULTIMODAL_REGISTRY
 from vllm.multimodal.processing import ProcessorInputs,TimingContext
 from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase
 paths={'gemma3_4b':'gemma-3-4b-it','qwen35_4b':'Qwen3.5-4B','qwen25vl':'Qwen2.5-VL-7B-Instruct','minicpm26':'MiniCPM-V-2_6'}
 out=ROOT/'outputs/records/acceleration_v4/cpu_processor_audit_v1';out.mkdir(parents=True,exist_ok=True)
 for model in a.models:
  dest=out/(model+'.json')
  if dest.exists():raise RuntimeError('fresh evidence required '+str(dest))
  evidence={'model':model,'cpu_only':True,'script_sha256':file_hash(Path(__file__)),'rows':[]}
  try:
   refpath=ROOT/'outputs/records/acceleration_v4/native_reference'/(model+'.json');ref=json.loads(refpath.read_text())
   path=Path('/home/team/lvshuyang/Models')/paths[model] if model in paths else ROOT/'cache/models/llava-v1.6-mistral-7b-hf'
   cfg=ModelConfig(model=str(path),dtype='bfloat16',trust_remote_code=True,max_model_len=32768,limit_mm_per_prompt={'image':1,'video':0})
   evidence.update(reference_sha256=file_hash(refpath),model_config_is_mm_prefix_lm=cfg.is_mm_prefix_lm,model_type=cfg.hf_config.model_type)
   if model=='gemma3_4b':
    sub=copy.deepcopy(cfg);sub.hf_config=copy.deepcopy(cfg.hf_config.text_config);sub.hf_config.architectures=['Gemma3ForCausalLM'];sub.model_arch_config=sub.get_model_arch_config()
    evidence['submodel_arch_is_mm_prefix_lm']=sub.is_mm_prefix_lm
   proc=MULTIMODAL_REGISTRY.create_processor(cfg,cache=None)
   evidence['vllm_processor_class']=type(proc).__name__
   for row in ref['rows']:
    s=row['reference']['sample']
    with Image.open(resolve_image_path(s['image_path'],ROOT)) as im:image=im.convert('RGB')
    items=proc.info.parse_mm_data({'image':image})
    processed=proc.apply(ProcessorInputs(prompt=row['unexpanded_prompt_ids'],mm_data_items=items,hf_processor_mm_kwargs={},tokenization_kwargs={}),TimingContext(enabled=False))
    mm=processed['mm_kwargs'].get_data(device='cpu',pin_memory=False)
    actual=summary(mm);expected=row['processor_summary']
    common={k:core(actual[k])==core(expected[k]) for k in actual if k in expected}
    native_types=expected.get('token_type_ids',{}).get('values',[[]])[0]
    evidence['rows'].append({'sample_id':s['id'],'expanded_prompt_ids_equal':processed['prompt_token_ids']==row['expanded_prompt_ids'],'common_tensor_parity':common,'actual_processor_tensors':actual,'mm_placeholders':summary(processed['mm_placeholders']),'native_image_token_ranges':ranges(native_types) if native_types else None})
   evidence['status']='complete'
  except Exception as e:evidence.update(status='failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
  atomic_json(dest,evidence);print(json.dumps({'model':model,'status':evidence['status'],'error':evidence.get('error'),'rows':len(evidence['rows'])}),flush=True)
if __name__=='__main__':main()
