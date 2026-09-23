"""One approved LLaVA cohort, then one free CPU labeling pass; no retries."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,traceback
ROOT=pathlib.Path('/home/k100/projects/knowledge-deficit-mitigation')
RECORD=ROOT/'outputs/records/independent_v5/llava16_dispatch_v1.json'
RUN='panel5_independent_food_v5'
MODEL='llava16_mistral'
PYTHON=ROOT/'.environments/vllm024/bin/python'
record_dir=f'outputs/records/acceleration_v4/{RUN}/{MODEL}'
closed_dir='outputs/records/acceleration_v4/panel5_engine_v2/llava16_mistral'
cmd=['bash','workflows/independent_k100_v1/worker_v2.sh','workflows/independent_k100_v1/entry_v2.py','generate','--model',MODEL,'--closed-record',closed_dir,'--native-reference','outputs/records/independent_v5/native16_v3/llava16_mistral/reference.json','--verification','outputs/records/independent_v5/engine_verify_v2/llava16_mistral/check.json','--execute','--run',RUN]
post=[str(PYTHON),'workflows/independent_v5/postprocess.py','--record',record_dir,'--closed-record',closed_dir,'--out',f'outputs/annotations/independent_v5/{RUN}/{MODEL}']
state={'schema':'kdm_single_production_then_free_cpu_postprocess_v1','supervisor_pid':os.getpid(),'model':MODEL,'run':RUN,'host':'RTX_Pro_6000','gpu':'0','gpu_uuid':'GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea','command':cmd,'postprocess_command':post,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_retries':True,'api_requests':0,'status':'launching','driver_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
def write():
 temp=RECORD.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(RECORD)
assert not RECORD.exists(),'Fresh dispatch required'
assert not (ROOT/record_dir).exists(),'Fresh production required'
write()
try:
 with (ROOT/'outputs/records/independent_v5/llava16_production_v1.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  state.update(status='generation_running',worker_pid=p.pid);write();rc=p.wait()
 state['generation_exit_code']=rc
 if rc:raise RuntimeError(f'Generation failed with exit {rc}; no retry')
 complete=json.loads((ROOT/record_dir/'complete.json').read_text())
 assert complete['generation_complete'] is True and complete['independent']==48480,'Incomplete generation'
 state.update(status='free_postprocess_running');write()
 with (ROOT/'outputs/records/independent_v5/llava16_free_postprocess_v1.log').open('x') as log:
  q=subprocess.Popen(post,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  state['postprocess_pid']=q.pid;write();post_rc=q.wait()
 state['postprocess_exit_code']=post_rc
 if post_rc:raise RuntimeError(f'Free postprocess failed with exit {post_rc}; no retry')
 state.update(status='generation_and_free_postprocess_complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write()
except BaseException as e:
 state.update(status='failed_no_retry',error=repr(e),traceback=traceback.format_exc(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write();raise
