import os,select,json,subprocess,time,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];B=R/'outputs/records/remaining11_minicpm45_v1'
state={'pid':os.getpid(),'wait_native_pid':415207,'status':'waiting_native_pidfd','no_retry':True}
def write(): (B/'after_native.json').write_text(json.dumps(state,indent=2))
write()
try:
 try:
  fd=os.pidfd_open(415207);p=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)
 except ProcessLookupError:pass
 native=json.loads((B/'native_sequence.json').read_text());assert native['status']=='native_references_complete'
 assert json.loads((B/'native_progress.json').read_text())['completed']==16
 assert json.loads((B/'independent_native16/complete.json').read_text())['complete'] is True
 cmd=['bash','workflows/food_closed_v3/worker.sh',str(R),'0','/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python','-u','workflows/remaining11_minicpm45_v1/closed_entry.py']
 (B/'engine').mkdir(exist_ok=False)
 with (B/'engine/log.txt').open('x') as f:
  q=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT);state.update(status='candidate_engine_audit_running',worker_pid=q.pid,command=cmd);write();rc=q.wait()
 state.update(status='complete' if rc==0 else 'failed_no_retry',exit_code=rc);write()
except BaseException as e:
 state.update(status='failed_no_retry',error=str(e),traceback=traceback.format_exc());write();raise
