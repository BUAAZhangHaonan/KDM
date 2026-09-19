"""CPU regression with real randomly initialized Llama attention and native KV cache."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kdm.models.hf import HFBackend, HFSession
from kdm.models.sid import SIDSession, keep_visual_indices, restricted_attention_mask


def checker():
    path=Path(__file__).resolve().parents[1]/'verification/sid_reference_check.py'
    spec=importlib.util.spec_from_file_location('sid_reference_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend():
    torch.set_num_threads(1)
    torch.manual_seed(719)
    config=LlamaConfig(vocab_size=160,hidden_size=32,intermediate_size=64,
                       num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                       max_position_embeddings=512,attention_dropout=0.)
    config.image_token_id=159
    config._attn_implementation='eager'
    result=HFBackend.__new__(HFBackend)
    result.model=LlamaForCausalLM(config).eval()
    result.torch=torch;result.device='cpu';result.em=SimpleNamespace(mt='fixture')
    return result


def inputs(extra=0):
    ids=torch.tensor([[1]+[159]*104+[3,4,5]+[7]*extra])
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}


def test_official_source_real_forward_interleaving_and_nonmonotonic(backend):
    module=checker()
    checks,evidence=module.compare(backend,inputs(),inputs(3),[(),(9,),(9,10),(9,10),(11,),(),(9,10,11)])
    assert all(checks.values()),evidence
    assert any(row['actual'] and row['actual'][0]['query_length']>1 for row in evidence)
    assert any(row['actual'] and row['actual'][0]['query_length']==1 for row in evidence)
    assert all(len(trace['selected'])==100 for row in evidence for trace in row['actual'])
    assert all(trace['layer'] in (2,3) for row in evidence for trace in row['actual'])
    assert not getattr(backend,'_sid_active_control',None)
    assert all(not layer._forward_pre_hooks for layer in backend.model.model.layers)


def test_clean_forward_not_captured_and_exception_restores_hooks(backend):
    original=backend.model.model.layers[1].self_attn.forward
    clean=HFSession(backend,inputs(),False,False).next(()).logits.copy()
    one=SIDSession(backend,inputs());two=SIDSession(backend,inputs(3))
    assert backend.model.model.layers[1].self_attn.forward==original
    sid_logits=one.next(()).logits
    assert not np.array_equal(clean,sid_logits), 'Reference pruning must have a measurable numerical effect'
    assert np.array_equal(clean,HFSession(backend,inputs(),False,False).next(()).logits)
    def explode(event):raise RuntimeError('deliberate audit failure')
    two.control.audit=explode
    with pytest.raises(RuntimeError,match='deliberate audit failure'):two.next(())
    assert backend.model.model.layers[1].self_attn.forward==original
    assert not getattr(backend,'_sid_active_control',None)
    assert all(not layer._forward_pre_hooks for layer in backend.model.model.layers)
    assert np.array_equal(clean,HFSession(backend,inputs(),False,False).next(()).logits)


def test_query_rows_keep_causality_padding_and_selected_visuals():
    hidden=torch.zeros(1,4,8)
    original=torch.zeros(1,1,4,108)
    query=torch.arange(4)[:,None]+104
    original.masked_fill_((torch.arange(108)[None,:]>query)[None,None],float('-inf'))
    original[:,:,:,103]=float('-inf')
    keep=torch.arange(100)
    actual=restricted_attention_mask(torch,original,hidden,108,1,104,keep)
    assert torch.equal(actual[:,:,:,106].isneginf(),torch.tensor([[[True,True,False,False]]]))
    assert actual[:,:,:,101:105].isneginf().all()
    assert actual[:,:,:,0].eq(0).all() and actual[:,:,:,1:101].eq(0).all()
    assert original[:,:,:,101].eq(0).all()
    with pytest.raises(ValueError,match='floating'):
        restricted_attention_mask(torch,original.bool(),hidden,108,1,104,keep)


def test_rank_never_clamped_and_visual_span_must_be_expanded(backend):
    with pytest.raises(ValueError,match='at least 100'):
        keep_visual_indices(torch.ones(2,4,36),1,32)
    with pytest.raises(RuntimeError,match='placeholder/Q-former'):
        SIDSession(backend,{'input_ids':torch.tensor([[1,159,2]])})
    bad=inputs();bad['input_ids'][0,51]=3
    with pytest.raises(RuntimeError,match='not contiguous'):SIDSession(backend,bad)


def test_phi_visual_positions_use_native_negative_predicate(backend):
    backend.model.config.model_type='phi3_v'
    ids=inputs()['input_ids'].clone();ids[ids==159]=-1
    session=SIDSession(backend,{'input_ids':ids})
    assert session.control.start==1 and session.control.length==104
    bad=ids.clone();bad[0,50]=-1000000000
    with pytest.raises(RuntimeError,match='not contiguous'):
        SIDSession(backend,{'input_ids':bad})


def test_downstream_flash_dispatch_scoped_and_restored_even_on_error(backend):
    control=SIDSession(backend,inputs()).control
    cfg=backend.model.config;cfg._attn_implementation='flash_attention_2'
    before=[]
    for layer in backend.model.model.layers[2:]:
        def observe(**kwargs):return cfg._attn_implementation
        layer.self_attn.forward=observe
        before.append(observe)
    with pytest.raises(RuntimeError,match='scope exit'):
        with control.forward_scope(108,108):
            for layer in backend.model.model.layers[2:]:
                assert layer.self_attn.forward()=='eager'
                assert cfg._attn_implementation=='flash_attention_2'
            raise RuntimeError('scope exit')
    assert [layer.self_attn.forward for layer in backend.model.model.layers[2:]]==before
    assert cfg._attn_implementation=='flash_attention_2'
    assert all(not layer._forward_pre_hooks for layer in backend.model.model.layers)
