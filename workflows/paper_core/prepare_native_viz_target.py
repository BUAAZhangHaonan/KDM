"""Bounded CPU checkpoint/software and current GPU-capacity location check."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json,file_hash
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11.execution import checkpoint_identity
from author_baseline_pilot import load_inputs
from workflows.paper_core.dev_viz import roster
import workflows.paper_core.execution as execution
import kdm.execution as image_execution
import torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--registry',required=True)
p.add_argument('--gpu',required=True);p.add_argument('--output',required=True);a=p.parse_args()
_,spec,proof=load_inputs(a.model,'dola');execution.REGISTRY=a.registry
registry,host,details=execution.registration(ROOT)
actual=spec if a.registry=='configs/runtime/hosts.json' else execution.runtime_spec(ROOT,spec,a.model)
minimum=registry.get('minimum_free_mib',{}).get(a.model)
environment=validate_environment(actual);checkpoint=checkpoint_identity(actual)
image_execution.REGISTRY=a.registry
image_execution.read_registry(ROOT)
_,samples=roster('viz512')
first_image=image_execution.resolve_image_path(samples[0]['image_path'],ROOT)
line=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
card,uuid,free=[v.strip() for v in line.split(',')]
assert card==a.gpu and uuid==details['gpu_uuids'][card] and (minimum is None or int(free)>=minimum)
assert actual['dtype']==spec['dtype'] and not torch.cuda.is_initialized()
result=dict(target_ready=True,model=a.model,host=host,registry=a.registry,registry_sha256=file_hash(ROOT/a.registry),
    runtime_spec=actual,environment=environment,checkpoint=checkpoint,native_projection_proof=proof,
    physical_gpu=card,gpu_uuid=uuid,observed_free_mib=int(free),minimum_free_mib=minimum,
    first_registered_image=str(first_image),scientific_parameters_changed=False,CUDA_initialized=False,actual_model_generation=False,
    lock_and_current_free_memory_rechecked_by_actual_worker=True)
atomic_json(ROOT/a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_spec','environment','checkpoint','native_projection_proof')}))
