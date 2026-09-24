"""Byte-only metadata digest supporting scalar tensors; never mutates inputs."""
import hashlib
def summarize(v):
 import torch
 if torch.is_tensor(v):
  x=v.detach().cpu().contiguous()
  return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
 if isinstance(v,dict):return {str(k):summarize(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [summarize(x) for x in v]
 return v if v is None or isinstance(v,(str,int,float,bool)) else {'type':type(v).__name__}
