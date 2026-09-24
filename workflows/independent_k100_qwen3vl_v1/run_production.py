"""One reviewed Qwen3VL production launch; no automatic retry or paid API."""
import datetime,hashlib,json,os,pathlib,subprocess,traceback
ROOT=pathlib.Path(__file__).resolve().parents[2]
D=ROOT/'outputs/records/independent_k100_qwen3vl_v1'
RECORD=D/'production_dispatch.json'
RUN='extra11_independent_food_qwen3vl_v1';MODEL='qwen3vl'
record_dir=ROOT/'outputs/records/acceleration_v4'/RUN/MODEL
cmd=['bash','workflows/independent_k100_qwen3vl_v1/worker.sh','workflows/independent_k100_qwen3vl_v1/entry.py','generate','--model',MODEL,'--native-reference','outputs/records/independent_qwen3vl_native_v1/native16/reference.json','--verification','outputs/records/independent_k100_qwen3vl_v1/engine16/check.json','--execute','--run',RUN]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state={'schema':'kdm_qwen3vl_independent_production_dispatch_v1','supervisor_pid':os.getpid(),'model':MODEL,'run':RUN,'host':'RTX_Pro_6000','physical_gpu':0,'gpu_uuid':'GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea','command':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_retries':True,'api_requests':0,'status':'launching','driver_sha256':sha(pathlib.Path(__file__)),'source_files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in pathlib.Path(__file__).parent.iterdir() if p.is_file()},'final_gt':False,'closed_cohort_status':'pending_separate_verified_cohort'}
def write():
 temp=RECORD.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(RECORD)
assert not RECORD.exists() and not record_dir.exists(),'Fresh dispatch/production required'
review=D/'engine16/review.json';assert json.loads(review.read_text())['accepted_separate_cohort'] is True
state['review_sha256']=sha(review);write()
try:
 with (D/'production.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  state.update(status='generation_running',worker_pid=p.pid);write();rc=p.wait()
 state['generation_exit_code']=rc
 if rc:raise RuntimeError(f'Generation failed with exit {rc}; no retry')
 complete=json.loads((record_dir/'complete.json').read_text());progress=json.loads((record_dir/'progress.json').read_text())
 assert complete['generation_complete'] is True and complete['independent']==48480 and progress['status']=='generation_complete'
 state.update(status='generation_complete_pending_annotation_and_closed_gt',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_sha256=sha(record_dir/'complete.json'));write()
except BaseException as e:
 state.update(status='failed_no_retry',error=repr(e),traceback=traceback.format_exc(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write();raise
