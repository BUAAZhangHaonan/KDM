"""One-shot pidfd handoff from completed MiniCPM to Gemma GPU1 audit only."""
import argparse,hashlib,json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/records/independent_v5/gemma_native_pixels_handoff_v1'
ENTRY=ROOT/'workflows/independent_v5/gemma_native_pixels_entry_v1.py'
PYTHON='/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python'

def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def descendants(pid,parent):
 seen=set()
 while pid in parent and pid not in seen:
  seen.add(pid)
  if parent[pid]==228003:return True
  pid=parent[pid]
 return False

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 evidence={'schema':'kdm_one_shot_pidfd_gemma_audit_handoff_v1','watcher_pid':os.getpid(),'watched_pid':228003,'gpu':'1','production_launch_allowed':False,'started_unix':time.time(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 fds=[]
 try:
  rootfd=os.pidfd_open(228003);fds.append(rootfd)
  argv=Path('/proc/228003/cmdline').read_bytes().split(b'\0');joined=b' '.join(argv).decode()
  assert 'workflows/independent_v5/reviewed_entry.py' in joined and '--model minicpm26' in joined and '--run panel5_independent_food_v5' in joined
  parent={}
  for p in Path('/proc').glob('[0-9]*/stat'):
   try:
    data=p.read_text().rsplit(')',1)[1].split();parent[int(p.parent.name)]=int(data[1])
   except (OSError,ValueError):pass
  gpu_uuid='GPU-5c45e961-7442-eb90-9b8c-295a1cf14995'
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
  gpu_pids=[]
  for row in rows:
   uuid,pid=[v.strip() for v in row.split(',')]
   if uuid!=gpu_uuid:continue
   pid=int(pid);assert pid==228003 or descendants(pid,parent),'Unexpected GPU1 owner, do not schedule'
   gpu_pids.append(pid)
   if pid!=228003:
    try:fds.append(os.pidfd_open(pid))
    except ProcessLookupError:pass
  evidence.update(status='waiting_pidfd',watched_command=joined,initial_gpu_pids=gpu_pids,entry_sha256=hashlib.sha256(ENTRY.read_bytes()).hexdigest());write('status.json',evidence)
  poll=select.poll()
  for fd in fds:poll.register(fd,select.POLLIN)
  pending=set(fds)
  while pending:
   for fd,_ in poll.poll():
    poll.unregister(fd);pending.remove(fd)
  record=ROOT/'outputs/records/acceleration_v4/panel5_independent_food_v5/minicpm26'
  complete=json.loads((record/'complete.json').read_text());progress=json.loads((record/'progress.json').read_text())
  assert complete['generation_complete'] and complete['independent']==48480 and progress['status']=='generation_complete'
  raw=ROOT/progress['output'];h=hashlib.sha256()
  with raw.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==complete['output_sha256']
  assert evidence['entry_sha256']==hashlib.sha256(ENTRY.read_bytes()).hexdigest(),'Entry changed while waiting'
  command=['bash',str(ROOT/'workflows/food_closed_v3/worker.sh'),str(ROOT),'1',PYTHON,'-u',str(ENTRY),'verify','--model','gemma3_4b','--native-reference','outputs/records/independent_v5/native16_v2/gemma3_4b/reference.json','--out','outputs/records/independent_v5/engine_verify_gemma_pixels_v1/gemma3_4b']
  with (OUT/'audit.log').open('wb') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  evidence.update(status='audit_running',audit_pid=child.pid,command=command,predecessor_complete_sha256=hashlib.sha256((record/'complete.json').read_bytes()).hexdigest(),launched_unix=time.time());write('status.json',evidence)
  rc=child.wait();evidence.update(status='audit_exited_review_required' if rc==0 else 'audit_failed_no_retry',audit_exit_code=rc,finished_unix=time.time());write('status.json',evidence)
 except BaseException as exc:
  evidence.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),finished_unix=time.time());write('status.json',evidence);raise
 finally:
  for fd in fds:os.close(fd)

if __name__=='__main__':main()
