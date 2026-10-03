#!/usr/bin/env python3
"""Original OneVision failed-input three-branch next-logit capacity check."""
import argparse,pathlib,sys,json,os,time,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.supplemental.remaining11 import execution
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.native_audit import input_description
from kdm.execution import resolve_image_path
from kdm.pipeline import make_backend,sessions
from kdm.decoding import DecodeConfig
from kdm.io import stable_seed,file_hash
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--physical-gpus',required=True);p.add_argument('--host-registry',required=True);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
try:
 execution.REGISTRY=a.host_registry;os.environ['KDM_SUPPLEMENTAL_DATASET']='vizwiz';spec=json.loads((ROOT/'configs/runtime/onevision.json').read_text());adm=execution.validate_supplemental_runtime(ROOT,spec,'onevision',a.physical_gpus.split(','),'formal','One_failed_input_target_forward','selected_main_viz');backend=make_backend(adm['runtime_spec'],'cuda:0');loaded=time.perf_counter()-started
 _,samples=roster('viz512');sample=next(x for x in samples if x['id']=='vizwiz:VizWiz_val_00002603.jpg');seed=stable_seed(sample['id'],'onevision',0);task=dict(sample=sample,method='instruction_vcd',kind='instruction_preserving',marker='I cannot identify it',reference_marker='I cannot identify it',guided=True,reference_guided=False,replicate=0)
 import numpy as np,torch
 with Image.open(resolve_image_path(sample['image_path'],ROOT))as src:
  image=src.convert('RGB');main,ref,neutral,prompt,rprompt=sessions(backend,image,task,DecodeConfig(method='instruction_vcd'),seed);proof=[]
  for role,session in [('main',main),('registered_noisy_reference',ref),('neutral_clean_oom_lm_head_branch',neutral)]:
   step=session.next(());assert np.isfinite(step.logits).all();proof.append(dict(role=role,full_vocab_finite=True,vocab_n=len(step.logits),inputs=input_description(session.inputs)))
  assert len(proof)==3
 rec=dict(target_ready=True,model='onevision',scientific_parameters_changed=False,actual_admission=adm,runtime_spec=adm['runtime_spec'],failed_sample_id=sample['id'],failed_key='323e78ac9bbd21e95799c56d0f86fa5f5e3c198b861b87bac3abd9c10807ad5a',same_three_live_sessions=True,original_failed_neutral_lm_head_branch_actually_forwarded=True,actual_forward_proof=proof,actual_generated_scientific_rows=0,model_load_wall_s=loaded,actual_GPU_seconds_including_load=(time.perf_counter()-started)*len(a.physical_gpus.split(',')),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),entry_sha256=file_hash(pathlib.Path(__file__)))
 (out/'actual_target_ready.json').open('x').write(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k]for k in ['target_ready','failed_sample_id','same_three_live_sessions','original_failed_neutral_lm_head_branch_actually_forwarded','actual_generated_scientific_rows','actual_GPU_seconds_including_load','peak_allocated_bytes','peak_reserved_bytes']}))
except BaseException as e:
 (out/'failure.json').open('x').write(json.dumps(dict(error=str(e),traceback=traceback.format_exc(),actual_GPU_seconds_including_load=time.perf_counter()-started),indent=2)+'\n');raise
