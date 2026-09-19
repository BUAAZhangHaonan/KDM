import pytest
from kdm.models.backbone import InternVLModel
from kdm.models.hf import HFBackend
from kdm.models.internvl_dual import InternVLDualBackend,InternVLDualEngine,validate_dual_placement


def placement():
    result={'vision_model':0,'mlp1':0,'language_model.model.embed_tokens':0,
            'language_model.model.rotary_emb':0,'language_model.model.norm':1,'language_model.lm_head':1}
    result.update({f'language_model.model.layers.{i}':int(i>=18) for i in range(36)})
    return result


def test_dual_only_changes_model_construction():
    for name in ('build','build_text','_query','_prefill','_step'):
        assert getattr(InternVLDualEngine,name) is getattr(InternVLModel,name)
    for name in ('session','_forward','encode','decode','_text_inputs'):
        assert getattr(InternVLDualBackend,name) is getattr(HFBackend,name)


def test_approved_explicit_layout_has_no_missing_or_duplicate_layers():
    device_map=placement()
    assert validate_dual_placement(device_map,{'0':'22GiB','1':'22GiB'},'cuda:0','bfloat16')=={0:'22GiB',1:'22GiB'}
    layers=[device_map[f'language_model.model.layers.{i}'] for i in range(36)]
    assert layers.count(0)==layers.count(1)==18
    assert device_map['vision_model']==device_map['mlp1']==device_map['language_model.model.embed_tokens']==0


@pytest.mark.parametrize('wrong',['auto',{'':'cpu'},{'':'disk'},{}])
def test_reject_offload_or_incomplete_maps_before_loading(wrong):
    with pytest.raises(ValueError):validate_dual_placement(wrong,{0:'22GiB',1:'22GiB'},'cuda:0','bfloat16')


def test_reject_dtype_input_device_or_memory_changes():
    for device,dtype,memory in [('cuda:1','bfloat16',{0:'22GiB',1:'22GiB'}),
                                ('cuda:0','float16',{0:'22GiB',1:'22GiB'}),
                                ('cuda:0','bfloat16',{0:'22GiB'})]:
        with pytest.raises(ValueError):validate_dual_placement(placement(),memory,device,dtype)
