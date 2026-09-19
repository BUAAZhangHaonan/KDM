import importlib.util,json
from pathlib import Path
import pytest
from kdm.models.hf import HFBackend

def discovery():
    path=Path(__file__).resolve().parents[1]/'scripts/discover_models.py'
    spec=importlib.util.spec_from_file_location('kdm_discover',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_config_without_weight_shards_is_incomplete(tmp_path):
    d=tmp_path/'checkpoint';d.mkdir();(d/'config.json').write_text('{}')
    (d/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'part.safetensors'}}))
    rows=discovery().discover([{'key':'sample','directory_names':['checkpoint']}],[tmp_path])
    assert rows[0]['resolution']=='incomplete'
    assert 'part.safetensors' in rows[0]['checkpoint_error']

def test_discovery_does_not_choose_ambiguous_model(tmp_path):
    roots=[tmp_path/'a',tmp_path/'b']
    for root in roots:
        d=root/'checkpoint';d.mkdir(parents=True);(d/'config.json').write_text('{}');(d/'model.safetensors').write_bytes(b'weights')
    row=discovery().discover([{'directory_names':['checkpoint']}],roots)[0]
    assert row['resolution']=='ambiguous' and row['resolved_path'] is None

def test_decode_retains_whitespace():
    class Tokenizer:
        def decode(self,tokens,**kwargs):
            assert kwargs=={'skip_special_tokens':True,'clean_up_tokenization_spaces':False}
            return ' answer '
    backend=HFBackend.__new__(HFBackend);backend.tokenizer=Tokenizer()
    assert backend.decode([1,2])==' answer '

@pytest.mark.parametrize('dtype_name',['float32','bfloat16'])
def test_noise_matches_official_native_dtype(dtype_name):
    import torch
    from kdm.models.hf import vcd_noise
    path=Path(__file__).resolve().parents[1]/'reference_repos/vcd/vcd_utils/vcd_add_noise.py'
    spec=importlib.util.spec_from_file_location('official_vcd_noise',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=torch.arange(60,dtype=torch.float32).reshape(1,3,4,5).to(getattr(torch,dtype_name))/13
    torch.manual_seed(1729);expected=module.add_diffusion_noise(x,500)
    state=torch.random.get_rng_state().clone();actual=vcd_noise(x,1729)
    assert torch.equal(actual,expected)
    assert torch.equal(state,torch.random.get_rng_state())

@pytest.mark.parametrize('placement',['auto','balanced',{'model':'cpu'},{'model':'disk'}])
def test_offload_is_rejected_before_model_construction(placement):
    with pytest.raises(ValueError):HFBackend('/nonexistent',device_map=placement)

def test_minicpm_forward_preserves_original_data_mapping():
    import torch
    from kdm.models.remote import MiniCPMModel
    engine=MiniCPMModel.__new__(MiniCPMModel)
    image_bound=[torch.tensor([[4,8],[10,14]])]
    pixels=[[torch.zeros((3,4,4)),torch.ones((3,4,4))]]
    inputs={'input_ids':torch.tensor([[1,2,3]]),'pixel_values':pixels,
            'image_bound':image_bound,'tgt_sizes':[torch.tensor([[2,2]])],
            'position_ids':torch.tensor([[0,1,2]]),'attention_mask':torch.ones(1,3),
            'temporal_ids':[[]]}
    def model(**kw):
        assert kw['data']['pixel_values'] is pixels
        assert kw['data']['image_bound'] is image_bound
        assert kw['data']['position_ids'] is inputs['position_ids']
        assert kw['data']['temporal_ids'] is inputs['temporal_ids']
        assert kw['attention_mask'] is inputs['attention_mask']
        assert kw['use_cache'] is True
        return 'forward-output'
    engine.model=model
    assert engine._prefill(inputs)=='forward-output'


def test_nested_minicpm_noise_preserves_slices_and_rng_stream():
    import torch
    from kdm.models.hf import vcd_processed_noise
    path=Path(__file__).resolve().parents[1]/'reference_repos/vcd/vcd_utils/vcd_add_noise.py'
    spec=importlib.util.spec_from_file_location('official_vcd_noise_nested',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    first=torch.arange(12,dtype=torch.float32).reshape(1,3,4)
    second=torch.zeros(1,4,4,dtype=torch.bfloat16)
    torch.manual_seed(47)
    expected=[module.add_diffusion_noise(first,500),module.add_diffusion_noise(second,500)]
    original=first.clone();result=vcd_processed_noise([[first,second]],47)
    assert isinstance(result,list) and len(result)==1 and len(result[0])==2
    assert all(torch.equal(a,b) for a,b in zip(result[0],expected))
    assert torch.equal(first,original)

def test_nonempty_truncated_safetensors_is_incomplete(tmp_path):
    d=tmp_path/'checkpoint';d.mkdir();(d/'config.json').write_text('{}')
    header=json.dumps({'weight':{'dtype':'F32','shape':[25],'data_offsets':[0,100]}}).encode()
    (d/'model.safetensors').write_bytes(len(header).to_bytes(8,'little')+header+b'x')
    row=discovery().discover([{'directory_names':['checkpoint']}],[tmp_path])[0]
    assert row['resolution']=='incomplete'
    assert 'expected' in row['checkpoint_error']
