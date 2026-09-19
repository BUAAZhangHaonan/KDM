"""No-weight SID structural rejection, based on actual config and installed source."""
import argparse,hashlib,json,sys,inspect
from pathlib import Path
from transformers import AutoConfig
from kdm.models.sid import SIDControl
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
a=p.parse_args();spec=json.loads((ROOT/a.spec).read_text());key=spec['key']
assert key in {'qwen35_4b','qwen35_9b','minicpm26','minicpm45','phi35'}
cfg=AutoConfig.from_pretrained(spec['kwargs']['model_path'],trust_remote_code=True,local_files_only=True)
names={'hf.py','backbone.py','sid.py'}
if key.startswith('minicpm') or key=='phi35':names.add('remote.py')
record={'passed':False,'reference_commit':'127dd412fa6b61ab1c9babf6979ec4da98002438','spec':spec,
        'stage':'before_weight_loading','environment_python':sys.executable,
        'runtime_adapter_sha256':{n:hashlib.sha256((ROOT/'src/kdm/models'/n).read_bytes()).hexdigest() for n in names},
        'gpu_compute_executed':False,'checks':{},'evidence':[]}
def evidence(path,needle):
 path=Path(path);lines=path.read_text().splitlines();hits=[]
 for n,line in enumerate(lines,1):
  if needle in line:hits.append({'line':n,'source':'\n'.join(lines[max(0,n-2):min(len(lines),n+6)])})
 assert hits,(path,needle)
 record['evidence'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'hits':hits})
if key.startswith('qwen35'):
 from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
 assert cfg.text_config.layer_types[1]=='linear_attention'
 evidence(inspect.getsourcefile(Qwen3_5DecoderLayer),'self.block_type = config.layer_types[layer_idx]')
 record['classification']='fixed_reference_structure_incompatible'
 record['error']='Fixed SID reads second decoder self_attn weights, but registered layer_types[1]=linear_attention constructs Qwen3_5GatedDeltaNet as linear_attn (no self_attn). Selecting another full-attention layer would change agg_layer=2 protocol.'
 record['layer_types']=cfg.text_config.layer_types
elif key.startswith('minicpm'):
 path=Path(spec['kwargs']['model_path'])/'modeling_minicpmv.py'
 evidence(path,'self.llm = ')
 evidence(ROOT/'src/kdm/models/sid.py',"for path in ('language_model.model.layers'")
 assert not hasattr(cfg,'image_token_id')
 record['classification']='adapter_structural_mapping_gap'
 record['error']='Actual MiniCPMV wrapper holds decoder at llm.model.layers; current SID resolver registers no llm.model.layers path and no explicit image_token_id mapping. This is an adapter/visual-span mapping gap, not proof that SID cannot be defined for MiniCPM.'
else:
 from kdm.models.remote import PhiVisionModel
 assert not hasattr(cfg,'image_token_id')
 assert 'img_ctx_id' not in inspect.getsource(PhiVisionModel)
 evidence(ROOT/'src/kdm/models/remote.py','class PhiVisionModel')
 evidence(Path(spec['kwargs']['model_path'])/'configuration_phi3_v.py','class Phi3VConfig')
 record['classification']='adapter_structural_mapping_gap'
 record['error']='Actual Phi3VConfig has no image_token_id and PhiVisionModel provides no img_ctx_id. Current SID cannot map its visual sequence positions; no claim that SID is intrinsically impossible for Phi.'
record['config_sha256']=hashlib.sha256((Path(spec['kwargs']['model_path'])/'config.json').read_bytes()).hexdigest()
path=ROOT/a.output;path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'key':key,'classification':record['classification'],'error':record['error']},ensure_ascii=False))
