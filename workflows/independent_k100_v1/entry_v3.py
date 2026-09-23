"""K100 reviewed independent cohort; guarded entry supports vLLM spawn."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 sys.path.insert(0,str(ROOT/'workflows/independent_v5'))
 sys.path.insert(0,str(ROOT/'workflows/independent_k100_v1'))
 import admission
 import independent
 mode=sys.argv[1];model=sys.argv[sys.argv.index('--model')+1]
 resource=admission.register(model)
 independent.closed.PATHS.update(resource['model_paths'])
 independent.resolve_image_path=admission.image_resolver
 independent.closed.resolve_image_path=admission.image_resolver
 resource['entry_sha256']=admission.sha(__file__)
 resource['worker_sha256']=admission.sha(ROOT/'workflows/independent_k100_v1/worker_v3.sh')
 resource['multiprocessing_start_method']='spawn'
 independent.worker_admission=lambda:resource
 if mode not in ['verify','generate']:raise ValueError('Expected verify or generate')
 sys.argv=[sys.argv[0]]+sys.argv[2:]
 if mode=='verify':runpy.run_path(str(ROOT/'workflows/independent_v5/verify_engine.py'),run_name='__main__')
 else:
  import reviewed_entry
  reviewed_entry.main()
if __name__=='__main__':main()
