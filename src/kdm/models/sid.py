"""SID least-attended visual-token reference for compatible eager-attention models.

The visual span uses (start, length). Existing causal masks are preserved.
Reference selection is computed from the clean session at the same prefix.
"""
from __future__ import annotations
from .hf import HFSession


def keep_visual_indices(attention,start,length,k=100):
    if length<=0 or start<0 or start+length>attention.shape[-1]:raise ValueError('Invalid visual span')
    row=attention.mean(dim=0)[-1,start:start+length]
    return row.topk(min(k,length),largest=False).indices.sort().values


def restricted_attention_mask(torch,original,hidden,kv_length,start,length,keep):
    qlen=hidden.shape[1];device=hidden.device;dtype=hidden.dtype
    blocked=torch.zeros(kv_length,dtype=torch.bool,device=device)
    blocked[start:start+length]=True;blocked[start+keep.to(device)]=False
    if original is not None:
        if original.ndim!=4 or original.shape[-1]!=kv_length:raise ValueError('SID requires explicit 4D causal mask')
        return original.masked_fill(blocked[None,None,None,:],float('-inf'))
    past=kv_length-qlen
    if past<0:raise ValueError('Invalid attention lengths')
    q=torch.arange(qlen,device=device)[:,None]+past
    key=torch.arange(kv_length,device=device)[None,:]
    valid=key<=q
    return torch.zeros((1,1,qlen,kv_length),device=device,dtype=dtype).masked_fill(
           (~valid)[None,None,:,:]|blocked[None,None,None,:],float('-inf'))


class SIDControl:
    def __init__(self,backend,inputs):
        self.b=backend;self.active=False;self.attention=None;self.hooks=[]
        model=backend.model;layers=None
        for path in ('language_model.model.layers','model.language_model.layers','model.layers'):
            obj=model
            for name in path.split('.'):obj=getattr(obj,name,None)
            if obj is not None:layers=list(obj);break
        if layers is None:raise RuntimeError('SID transformer layers are unavailable')
        tok=getattr(backend.em,'img_ctx_id',None)
        if tok is None:tok=getattr(model.config,'image_token_id',None)
        if tok is None:raise RuntimeError('SID needs an explicit image token id')
        positions=(inputs['input_ids'][0]==tok).nonzero(as_tuple=True)[0]
        if not len(positions):raise RuntimeError('SID found no visual tokens')
        self.start=int(positions.min());self.length=int(positions.max())-self.start+1
        if self.length!=len(positions):raise RuntimeError('SID visual span is not contiguous')
        self.attn_module=layers[1].self_attn;self.original_forward=self.attn_module.forward
        def capture(*args,**kwargs):
            cfg=self.attn_module.config;previous=cfg._attn_implementation
            cfg._attn_implementation='eager'
            try:output=self.original_forward(*args,**kwargs)
            finally:cfg._attn_implementation=previous
            if not self.active:
                if len(output)<2 or output[1] is None:raise RuntimeError('SID needs layer-1 attention weights')
                self.attention=output[1].detach()[0]
            return output
        self.attn_module.forward=capture
        for idx,layer in enumerate(layers):
            if idx>=2:self.hooks.append(layer.register_forward_pre_hook(self._hook,with_kwargs=True))
    def _hook(self,module,args,kwargs):
        if not self.active:return args,kwargs
        if self.attention is None:raise RuntimeError('Run matched clean session before SID reference')
        hidden=args[0] if args else kwargs['hidden_states']
        keep=keep_visual_indices(self.attention,self.start,self.length)
        kwargs['attention_mask']=restricted_attention_mask(self.b.torch,kwargs.get('attention_mask'),hidden,
            self.attention.shape[-1],self.start,self.length,keep)
        return args,kwargs
    def close(self):
        self.attn_module.forward=self.original_forward
        for hook in self.hooks:hook.remove()
        self.hooks=[]


class SIDSession(HFSession):
    def __init__(self,backend,inputs):
        if getattr(backend,'sid_control',None) is not None:backend.sid_control.close()
        backend.sid_control=SIDControl(backend,inputs)
        self.control=backend.sid_control
        super().__init__(backend,inputs,False,False)
    def next(self,prefix):
        self.control.active=True
        try:return super().next(prefix)
        finally:self.control.active=False
