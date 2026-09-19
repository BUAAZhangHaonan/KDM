import os, sys
os.environ['HF_HOME'] = os.path.abspath('cache/hf')
os.environ['XDG_CACHE_HOME'] = os.path.abspath('cache/xdg')
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np

model_path = '/home/g203-4028/Models/Qwen3.5-4B'
proc = AutoProcessor.from_pretrained(model_path)
print('processor ok:', type(proc).__name__)
model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda:1')
model.eval()
print('model loaded, params:', sum(p.numel() for p in model.parameters())/1e9, 'B')

img = Image.fromarray((np.random.rand(448, 448, 3) * 255).astype('uint8'))
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'What is shown in this image? Answer with one word.'}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda:1')
print('input ids shape:', inputs['input_ids'].shape)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
text = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('OUTPUT:', text)

# test logits access
with torch.no_grad():
    res = model(**inputs)
print('logits shape:', res.logits.shape)
print('TEST_OK')
