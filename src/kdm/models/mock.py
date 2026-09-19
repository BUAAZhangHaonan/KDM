"""Deterministic CPU model for contract tests, never scientific data."""
import numpy as np
from ..decoding import Step

class MockSession:
    def __init__(self,table): self.table=table;self.calls=[]
    def next(self,prefix):
        prefix=tuple(prefix);self.calls.append(prefix)
        probs=self.table.get(prefix,self.table.get('default',[.01,.01,.01,.97]))
        z=np.log(np.asarray(probs,float))
        return Step(z,{0:z*.2,1:z*.7},{0:z*.3,1:z*.8})

class MockBackend:
    def __init__(self,device="cpu"):
        self.device=device
    eos={3}
    def encode(self,text):
        return {'UNKNOWN':[0],'UNCLEAR':[1],'UNSURE':[0,1],'I cannot identify it':[1,0],'milk':[2]}.get(text,[2])
    def decode(self,tokens):
        return ' '.join({0:'UNKNOWN',1:'UNCLEAR',2:'milk'}[i] for i in tokens if i!=3)
    def session(self,image,prompt,reference='clean',seed=0,need_layers=False):
        root=[.45,.05,.4,.1] if reference=='clean' else [.7,.02,.18,.1]
        return MockSession({():root,'default':[.01,.01,.01,.97]})
