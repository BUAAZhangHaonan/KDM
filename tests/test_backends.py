from types import SimpleNamespace
import numpy as np,pytest
import torch
from kdm.models.hf import HFSession
from kdm.models.sid import restricted_attention_mask,keep_visual_indices

class RopeModel(torch.nn.Module):
    def __init__(self):super().__init__();self.rope_deltas=None
class FakeBackend:
    def __init__(self):self.model=RopeModel();self.torch=torch;self.em=SimpleNamespace()
    def _forward(self,inputs=None,token=None,cache=None,ohs=False,text_only=False):
        if inputs is not None:self.model.rope_deltas=torch.tensor(inputs['offset']);length=0
        else:
            assert self.model.rope_deltas.item()==cache['offset']
            length=cache['length']+1
        value=int(self.model.rope_deltas.item())
        return SimpleNamespace(logits=torch.tensor([[[value,1.,2.,3.]]]),past_key_values={'offset':value,'length':length})

def test_separate_rope_state_and_prefix_reset():
    b=FakeBackend();a=HFSession(b,{'offset':7},False,False);r=HFSession(b,{'offset':13},False,False)
    assert a.next(()).logits[0]==7
    assert r.next(()).logits[0]==13
    assert a.next((0,)).logits[0]==7
    assert r.next((1,)).logits[0]==13
    assert a.next((1,2)).logits[0]==7
    assert a.next(()).logits[0]==7

def test_sid_span_and_causal_mask():
    att=torch.tensor([[[.2,.4,.1,.3,.5,.6]]])
    idx=keep_visual_indices(att,1,3,k=1);assert idx.tolist()==[1]
    h=torch.zeros(1,6,8);mask=restricted_attention_mask(torch,None,h,6,1,3,idx)
    assert torch.isneginf(mask[0,0,0,1:]).all()
    assert torch.isneginf(mask[0,0,5,1]) and mask[0,0,5,2]==0
    original=torch.zeros(1,1,1,6);original[:,:,:,0]=-float('inf')
    new=restricted_attention_mask(torch,original,torch.zeros(1,1,8),6,1,3,idx)
    assert torch.isneginf(new[0,0,0,0])
    with pytest.raises(ValueError):keep_visual_indices(att,4,3)
