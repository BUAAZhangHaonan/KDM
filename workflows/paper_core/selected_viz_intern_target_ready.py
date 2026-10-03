#!/usr/bin/env python3
"""One registered Viz input, four next-logit forwards; no generated science row."""
import pathlib,sys,json,time,os,traceback,datetime,argparse
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.paper_core import selected_viz_k100_intern as route
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.native_audit import input_description
from workflows.supplemental.remaining11 import execution
from kdm.pipeline import make_backend
from kdm.io import stable_seed,file_hash
from kdm.prompts import task_prompt
from kdm.execution import resolve_image_path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
try:
 route.install();execution.REGISTRY='workflows/supplemental/remaining4/dispatch_intern_k100_single_viz_registry_20261003.json';os.environ['KDM_SUPPLEMENTAL_DATASET']='vizwiz'
 spec=json.loads((ROOT/'configs/runtime/internvl35_8b.json').read_text());adm=execution.validate_supplemental_runtime(ROOT,spec,'internvl35_8b',['0'],'formal','intern_Viz_forward_readiness','selected_main_viz');backend=make_backend(adm['runtime_spec'],'cuda:0');loaded=time.perf_counter()-started
 _,samples=roster('viz512');sample=samples[-1];seed=stable_seed(sample['id'],'internvl35_8b',0);proof=[]
 import numpy as np
 with Image.open(resolve_image_path(sample['image_path'],ROOT))as source:
  image=source.convert('RGB')
  for role,marker,guided,ref in [('g','I cannot identify it',True,'clean'),('c','UNKNOWN',False,'clean'),('r','UNKNOWN',False,'noise'),('t','UNKNOWN',False,'text_only')]:
   session=backend.session(image,task_prompt(sample['question'],marker,guided),reference=ref,seed=seed);step=session.next(());assert np.isfinite(step.logits).all()and len(step.logits)>1000;proof.append({'role':role,'reference':ref,'prompt':task_prompt(sample['question'],marker,guided),'seed':seed,'inputs':input_description(session.inputs),'vocab_n':len(step.logits),'finite':True})
 rec={'target_ready':True,'model':'internvl35_8b','scientific_parameters_changed':False,'existing_single_gate':route.GATE,'existing_single_gate_sha256':file_hash(ROOT/route.GATE),'actual_runtime_spec':adm['runtime_spec'],'actual_admission':adm,'actual_forward_proof':proof,'sample_id':sample['id'],'actual_generated_scientific_rows':0,'completed_first8_not_regenerated':True,'model_load_wall_s':loaded,'actual_GPU_seconds_including_load':time.perf_counter()-started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (out/'actual_target_ready.json').open('x').write(json.dumps(rec,indent=2)+'\n');print(json.dumps({'target_ready':True,'GPU_seconds_including_load':rec['actual_GPU_seconds_including_load'],'actual_forward_count':len(proof),'actual_generated_scientific_rows':0}))
except BaseException as e:
 (out/'failure.json').open('x').write(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'actual_GPU_seconds_including_load':time.perf_counter()-started,'actual_generated_scientific_rows':0},indent=2));raise
