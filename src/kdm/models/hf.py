"""Incremental multimodal runner using model construction extracted from KDM.

The installer creates kdm.models.backbone from the pinned, already available
repository code. Custom model factories can implement the same session API.
"""
from __future__ import annotations
import math
import numpy as np
from ..decoding import Step

def vcd_noise(x,seed,generator=None):
    """Official VCD schedule and native tensor noise dtype, with a private RNG."""
    import torch
    if generator is None:
        generator=torch.Generator(device=x.device);generator.manual_seed(seed)
    beta=torch.sigmoid(torch.linspace(-6,6,1000))*(.005-.00001)+.00001
    abar=torch.cumprod(1-beta,0)
    noise=torch.empty_like(x).normal_(generator=generator)
    return abar.sqrt()[500]*x+(1-abar).sqrt()[500]*noise

def vcd_processed_noise(value,seed):
    """Preserve MiniCPM's nested slice structure with one per-session RNG stream."""
    import torch
    tensors=[]
    def collect(v):
        if isinstance(v,(list,tuple)):
            for item in v:collect(item)
        elif torch.is_tensor(v):tensors.append(v)
        else:raise TypeError('Processed image slices must be tensors')
    collect(value)
    if not tensors:raise ValueError('VCD requires processed image tensors')
    devices={x.device for x in tensors}
    if len(devices)!=1:raise ValueError('Processed image slices span multiple devices')
    generator=torch.Generator(device=tensors[0].device);generator.manual_seed(seed)
    def perturb(v):
        if isinstance(v,list):return [perturb(x) for x in v]
        if isinstance(v,tuple):return tuple(perturb(x) for x in v)
        return vcd_noise(v,seed,generator)
    return perturb(value)

class HFBackend:
    def __init__(self,model_path,device='cuda:0',attn_implementation=None,device_map=None,max_memory=None,dtype='bfloat16'):
        import torch
        if isinstance(device_map,str) and device_map in {'auto','balanced','balanced_low_0','sequential'}:
            raise ValueError('Provide an explicit GPU device map; automatic placement may offload weights')
        if isinstance(device_map,dict) and any(str(v) in {'cpu','disk'} for v in device_map.values()):
            raise ValueError('CPU/disk weight offload is prohibited')
        if max_memory is not None:max_memory={int(k) if str(k).isdigit() else k:v for k,v in max_memory.items()}
        from .backbone import get_engine
        self.em=get_engine(model_path,device,attn_implementation=attn_implementation,device_map=device_map,max_memory=max_memory,dtype=dtype)
        self.model=self.em.model;self.tokenizer=self.em.proc.tokenizer
        self.device=device;self.eos=set(self.em.eos);self.torch=torch
        placement=getattr(self.model,'hf_device_map',{})
        if any(str(v) in {'cpu','disk'} for v in placement.values()):raise RuntimeError('Unexpected model CPU/disk placement')
        if not self.eos: raise RuntimeError("No EOS token configured")
    def encode(self,text): return self.tokenizer.encode(text,add_special_tokens=False)
    def decode(self,tokens): return self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    def session(self,image,prompt,reference='clean',seed=0,need_layers=False):
        if reference=='text_only': inputs=self._text_inputs(prompt)
        else:
            if image is None: raise ValueError("Image is required")
            inputs=self.em.build(image,prompt)
        if reference=='sid':
            from .sid import SIDSession
            return SIDSession(self,inputs)
        if reference=='noise':
            if 'pixel_values' not in inputs: raise RuntimeError("No processed pixel_values for VCD")
            inputs={k:(v.clone() if hasattr(v,'clone') else v) for k,v in inputs.items()}
            inputs['pixel_values']=vcd_processed_noise(inputs['pixel_values'],seed)
        if reference not in {'clean','noise','text_only'}:
            raise RuntimeError(f"Reference needs an explicit adapter: {reference}")
        return HFSession(self,inputs,reference=='text_only',need_layers)
    def _text_inputs(self,text):
        t=self.torch
        if hasattr(self.em,'build_text'):
            return self.em.build_text(text)
        else:
            msg=[{'role':'user','content':[{'type':'text','text':text}]}]
            kw=dict(add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            if getattr(self.em,'mt','') in {'qwen3_5','qwen3_vl','glm4v'}: kw['enable_thinking']=False
            enc=self.em.proc.apply_chat_template(msg,**kw)
        return {k:v.to(self.device) for k,v in enc.items() if hasattr(v,'to')}
    def _forward(self,inputs=None,token=None,cache=None,ohs=False,text_only=False):
        t=self.torch
        with t.inference_mode():
            if getattr(self.em,'mt','')=='minicpmv':
                return self.em._prefill(inputs,ohs=ohs) if inputs is not None else self.em._step(token,cache,ohs=ohs)
            if getattr(self.em,'mt','')=='internvl_chat':
                if inputs is not None:
                    if text_only:
                        return self.model.language_model(**inputs,use_cache=True,output_hidden_states=ohs)
                    return self.em._prefill(inputs,ohs=ohs)
                return self.em._step(token,cache,ohs=ohs)
            if inputs is not None: return self.model(**inputs,use_cache=True,output_hidden_states=ohs)
            return self.model(input_ids=t.tensor([[token]],device=self.device),past_key_values=cache,
                              use_cache=True,output_hidden_states=ohs)

class HFSession:
    def __init__(self,backend,inputs,text_only,need_layers):
        self.b=backend;self.inputs=inputs;self.text_only=text_only;self.ohs=need_layers
        self.prefix=None;self.output=None;self.rope_state={}
    def _call(self, **kwargs):
        # Qwen implementations store rotary-position offsets on shared modules.
        # Restore each branch's offsets before an incremental forward.
        nodes = [(name, module) for name, module in self.b.model.named_modules()
                 if hasattr(module, 'rope_deltas')]
        for name, module in nodes:
            value = None if kwargs.get('inputs') is not None else self.rope_state.get(name)
            module.rope_deltas = value.clone() if hasattr(value, 'clone') else value
        result = self.b._forward(**kwargs)
        self.rope_state = {name: (module.rope_deltas.clone()
                          if hasattr(module.rope_deltas, 'clone') else module.rope_deltas)
                          for name, module in self.b.model.named_modules()
                          if hasattr(module, 'rope_deltas')}
        return result
    def next(self,prefix):
        prefix=tuple(prefix)
        extends = self.prefix is not None and len(prefix)==len(self.prefix)+1 and prefix[:-1]==self.prefix
        if self.prefix is None or (prefix != self.prefix and not extends):
            self.output=self._call(inputs=self.inputs,ohs=self.ohs,text_only=self.text_only)
            self.prefix=()
            for tok in prefix:
                self.output=self._call(token=tok,cache=self.output.past_key_values,ohs=self.ohs)
                self.prefix+= (tok,)
        elif prefix!=self.prefix:
            self.output=self._call(token=prefix[-1],cache=self.output.past_key_values,ohs=self.ohs)
            self.prefix=prefix
        z=self.output.logits[0,-1].float().detach().cpu().numpy()
        raw,normed={},{}
        if self.ohs:
            t=self.b.torch;hidden=self.output.hidden_states
            L=len(hidden)-1;head=self.b.em.lm_head;norm=self.b.em.norm
            if head is None or norm is None: raise RuntimeError("Layer decoder modules missing")
            with t.inference_mode():
                for idx in range(L): raw[idx]=head(hidden[idx][:,-1]).float()[0].cpu().numpy()
                for idx in range(math.ceil(.625*L),math.floor(.875*L)+1):
                    normed[idx]=head(norm(hidden[idx][:,-1])).float()[0].cpu().numpy()
        return Step(z,raw,normed)
