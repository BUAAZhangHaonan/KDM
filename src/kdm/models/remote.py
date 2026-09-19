"""Thin image adapters derived from pinned mprisk wrappers and checkpoint code.

mprisk source: cc6c0d82a77a958fd20c58e35efdc18c1ce0c036,
models/minicpm_v.py and models/phi3_vision.py. No video task is introduced.
Native data dictionaries, image slices and image_bound mappings are retained.
"""
from pathlib import Path
import torch

def move_inputs(value,device):
    if isinstance(value,dict):return {k:move_inputs(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move_inputs(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move_inputs(v,device) for v in value)
    return value.to(device) if hasattr(value,'to') else value

class MiniCPMModel:
    def __init__(self,model_path,device,attn_implementation=None,dtype='bfloat16',**kwargs):
        from transformers import AutoModel,AutoProcessor,AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        self.device=device;self.model_path=Path(model_path)
        self.cfg=AutoConfig.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        self.mt=self.cfg.model_type
        remote_class=get_class_from_dynamic_module('modeling_minicpmv.MiniCPMV',str(model_path),local_files_only=True)
        if not hasattr(remote_class,'all_tied_weights_keys'):remote_class.all_tied_weights_keys={}
        self.proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        self.model=AutoModel.from_pretrained(model_path,trust_remote_code=True,dtype=getattr(torch,dtype),
                    attn_implementation=attn_implementation or 'sdpa',device_map={'':device},local_files_only=True).eval()
        self._configure_tokenizer()
        self.eos={int(self.proc.tokenizer.convert_tokens_to_ids(t)) for t in self.model.terminators}
        self.lm_head=self.model.llm.get_output_embeddings();self.norm=self.model.llm.model.norm
        self.lcd_ready=True

    def build(self,image,text):
        self._configure_tokenizer()
        body='(<image>./</image>)\n'+text if image is not None else text
        kw={'tokenize':False,'add_generation_prompt':True}
        if '4_5' in self.model_path.name or '4-5' in self.model_path.name:kw['enable_thinking']=False
        prompt=self.proc.tokenizer.apply_chat_template([{'role':'user','content':body}],**kw)
        if image is not None:
            inputs=self.proc(text=[prompt],images=[[image]],return_tensors='pt')
        else:
            inputs=dict(self.proc.tokenizer([prompt],return_tensors='pt',padding=True))
            inputs.update(pixel_values=[[]],tgt_sizes=[],image_bound=[torch.empty((0,2),dtype=torch.long)])
        mask=inputs.get('attention_mask')
        if mask is None:raise ValueError('MiniCPM processor did not return attention_mask')
        if inputs.get('position_ids') is None:
            positions=mask.long().cumsum(-1)-1;inputs['position_ids']=positions.masked_fill(mask==0,0)
        return move_inputs(dict(inputs),self.device)

    def build_text(self,text):return self.build(None,text)

    def _prefill(self,inputs,ohs=False):
        keys=('input_ids','pixel_values','tgt_sizes','image_bound','position_ids','temporal_ids')
        data={key:inputs[key] for key in keys if key in inputs}
        if 'position_ids' not in data:raise ValueError('MiniCPM requires position_ids')
        return self.model(data=data,attention_mask=inputs.get('attention_mask'),use_cache=True,
                          output_hidden_states=ohs,return_dict=True)

    def _step(self,token,cache,ohs=False):
        return self.model.llm(input_ids=torch.tensor([[token]],device=self.device),past_key_values=cache,
                              use_cache=True,output_hidden_states=ohs,return_dict=True)

    def native_generate(self,inputs,max_new_tokens=32):
        keys=('input_ids','pixel_values','tgt_sizes','image_bound','temporal_ids','attention_mask')
        native={k:v for k,v in inputs.items() if k in keys}
        return self.model.generate(**native,tokenizer=self.proc.tokenizer,do_sample=False,
                                   max_new_tokens=max_new_tokens,decode_text=False)[0].tolist()

    def _configure_tokenizer(self) -> None:
        tokenizer = self.proc.tokenizer
        special_tokens = {
            "im_start": "<image>",
            "im_end": "</image>",
            "ref_start": "<ref>",
            "ref_end": "</ref>",
            "box_start": "<box>",
            "box_end": "</box>",
            "quad_start": "<quad>",
            "quad_end": "</quad>",
            "slice_start": "<slice>",
            "slice_end": "</slice>",
            "im_id_start": "<image_id>",
            "im_id_end": "</image_id>",
        }
        for name, token in special_tokens.items():
            setattr(tokenizer, name, token)
        token_ids = {
            "im_start_id": "<image>",
            "im_end_id": "</image>",
            "slice_start_id": "<slice>",
            "slice_end_id": "</slice>",
            "im_id_start_id": "<image_id>",
            "im_id_end_id": "</image_id>",
            "newline_id": "\n",
        }
        for name, token in token_ids.items():
            setattr(tokenizer, name, int(tokenizer.convert_tokens_to_ids(token)))
        tokenizer.bos_id = int(tokenizer.bos_token_id)
        tokenizer.eos_id = int(tokenizer.eos_token_id)
        tokenizer.unk_id = int(tokenizer.unk_token_id)

class PhiVisionModel:
    def __init__(self,model_path,device,attn_implementation=None,dtype='bfloat16',**kwargs):
        from transformers import AutoModelForCausalLM,AutoProcessor,AutoConfig
        self.device=device
        self.cfg=AutoConfig.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        self.mt=self.cfg.model_type
        self.proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,num_crops=4)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,
              torch_dtype=getattr(torch,dtype),_attn_implementation=attn_implementation or 'eager',
              device_map={'':device},local_files_only=True,low_cpu_mem_usage=True).eval()
        eos=self.cfg.eos_token_id;self.eos=set(eos if isinstance(eos,list) else [eos])
        if self.proc.tokenizer.eos_token_id is not None:self.eos.add(self.proc.tokenizer.eos_token_id)
        self.lm_head=self.model.get_output_embeddings();self.norm=self.model.model.norm
        self.lcd_ready=True

    def build(self,image,text):
        body='<|image_1|>\n'+text if image is not None else text
        prompt=f'<|user|>\n{body}<|end|>\n<|assistant|>\n'
        kw={'text':prompt,'return_tensors':'pt'}
        if image is not None:kw['images']=[image]
        return move_inputs(dict(self.proc(**kw)),self.device)

    def build_text(self,text):return self.build(None,text)
