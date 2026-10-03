#!/usr/bin/env python3
"""Original OneVision CPU processor binding for the authorized fixed-dev pilot."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import torch
from transformers import AutoProcessor, AutoConfig
from kdm.io import file_hash, stable_seed
from kdm.prompts import task_prompt
from kdm.models.backbone import FamilyModel
from kdm.models.hf import HFBackend
from kdm.execution import resolve_image_path
from kdm.protocol import validate_environment
from workflows.paper_core.ip_dev_pilot import roster, tasks_for, write
from workflows.paper_core.native_audit import input_description
from PIL import Image

def main():
    spec=json.loads((ROOT/'configs/runtime/onevision.json').read_text())
    environment=validate_environment(spec)
    model_path=spec['kwargs']['model_path']
    engine=FamilyModel.__new__(FamilyModel)
    engine.device='cpu'
    engine.cfg=AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    engine.mt=engine.cfg.model_type
    engine.proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    backend=HFBackend.__new__(HFBackend)
    backend.em,backend.tokenizer,backend.torch,backend.device=engine,engine.proc.tokenizer,torch,'cpu'
    roster_path,samples=roster('dev404')
    records={}
    for marker in ('UNKNOWN','UNCLEAR','UNSURE'):
        conditions=[]
        for sample in samples[:8]:
            main=task_prompt(sample['question'],marker,True)
            plain=task_prompt(sample['question'],marker,False)
            image_path=resolve_image_path(sample['image_path'],ROOT)
            with Image.open(image_path) as image:
                clean=engine.build(image.convert('RGB'),main)
                neutral=engine.build(image.convert('RGB'),plain)
            reference=backend._text_inputs(plain)
            conditions.append(dict(sample_id=sample['id'],main_prompt=main,reference_prompt=plain,
                 neutral_prompt=plain,seed=stable_seed(sample['id'],'onevision',0),
                 clean_inputs=input_description(clean),reference_inputs=input_description(reference),
                 neutral_inputs=input_description(neutral),
                 main_prompt_token_ids=clean['input_ids'].tolist(),
                 reference_prompt_token_ids=reference['input_ids'].tolist(),
                 neutral_prompt_token_ids=neutral['input_ids'].tolist(),
                 offset_prompt_tokens=backend.encode(plain),image_source_path=sample['image_path'],
                 original_image_sha256=file_hash(image_path)))
        records[marker]=conditions
    if torch.cuda.is_initialized():raise ValueError('CPU input binding initialized CUDA')
    result=dict(schema='kdm_original_OneVision_dev8_CPU_fixture_v1',model='onevision',dataset='food101',
       split='dev',methods=['instruction_m3id'],markers=list(records),original_cuda_initialized=False,
       environment=environment,roster_sha256=file_hash(roster_path),conditions=records,
       original_processor_source_sha256=file_hash(ROOT/'src/kdm/models/backbone.py'),
       original_text_adapter_source_sha256=file_hash(ROOT/'src/kdm/models/hf.py'),
       original_operator_source_sha256=file_hash(ROOT/'workflows/supplemental/remaining4/ip_m3id_compatibility.py'),
       original_GPU_same_answer_comparison_available=False,scientific_parameters_changed=False)
    out=ROOT/'outputs/paper_core_20261002_dev_viz/ip_only_dev404_pilot_20261003_2030/onevision/original_cpu_fixture.json'
    write(out,result)
    print(json.dumps(dict(path=str(out),sha256=file_hash(out),input_routes=72,cuda_initialized=False)))
if __name__=='__main__':main()
