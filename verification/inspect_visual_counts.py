import json,sys,inspect
from pathlib import Path
from transformers import AutoProcessor
r=Path(__file__).resolve().parents[1];s=json.loads((r/'configs/runtime'/f'{sys.argv[1]}.json').read_text())
p=AutoProcessor.from_pretrained(s['kwargs']['model_path'],trust_remote_code=True,local_files_only=True)
ip=p.image_processor
print(type(p).__name__,type(ip).__name__,inspect.getfile(type(ip)))
print(json.dumps(ip.to_dict(),default=str))
for obj,name in [(p,'_get_num_multimodal_tokens'),(ip,'get_number_of_image_patches'),(ip,'_preprocess')]:
 print('METHOD',name);print(inspect.getsource(getattr(obj,name)))
