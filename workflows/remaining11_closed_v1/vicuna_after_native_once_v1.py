import os,sys,json,time,select,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/records/remaining11_closed_v1/llava16_vicuna_v1'
if __name__=='__main__':
 native=json.loads((BASE/'native_dispatch.json').read_text());pid=int(native['pid'])
 try:fd=os.pidfd_open(pid)
 except ProcessLookupError:fd=None
 if fd is not None:select.select([fd],[],[]);os.close(fd)
 state=json.loads((BASE/'native_progress.json').read_text());assert state['status']=='complete' and state['completed']==16
 reference=BASE/'native_closed16_reference.json';assert hashlib.sha256(reference.read_bytes()).hexdigest()==state['reference_sha256']
 assert not (BASE/'engine_dispatch.json').exists()
 cmd=['bash','workflows/food_closed_v3/worker.sh',str(ROOT),'0','/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python','-u','workflows/remaining11_closed_v1/vicuna_entry_v1.py']
 env=os.environ.copy();env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 p=subprocess.Popen(cmd,cwd=ROOT,stdout=open(BASE/'engine.log','xb'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
 (BASE/'engine_dispatch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'physical_gpu':0,'gpu_uuid':'GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402','started_unix':time.time(),'native_reference_sha256':state['reference_sha256'],'dtype':'float16','entry_sha256':hashlib.sha256((ROOT/'workflows/remaining11_closed_v1/vicuna_entry_v1.py').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256((ROOT/'workflows/remaining11_closed_v1/vicuna_closed_scorer_v1.py').read_bytes()).hexdigest(),'gate':'strict actual16 input processing plus top1 and gold-correctness agreement before4848 new cohort'},indent=2));print('ENGINE_DISPATCH',p.pid,flush=True)
