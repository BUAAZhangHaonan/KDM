import json,subprocess,time,traceback,os
from pathlib import Path
R=Path(__file__).resolve().parents[2];B=R/'outputs/records/remaining11_llava15_v1'
commands=[['bash', 'workflows/food_closed_v3/worker.sh', '/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation', '0', '/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation/.environments/mprisk-tf553/bin/python', '-u', 'workflows/remaining11_llava15_v1/export_closed16.py'], ['bash', 'workflows/food_closed_v3/worker.sh', '/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation', '0', '/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation/.environments/mprisk-tf553/bin/python', '-u', 'workflows/remaining11_llava15_v1/export_independent16.py', '--model', 'llava15_7b', '--out', 'outputs/records/remaining11_llava15_v1/independent_native16']]
state={'pid':os.getpid(),'commands':commands,'status':'running','no_retry':True,'dtype':'float16'}
def write(): (B/'native_sequence.json').write_text(json.dumps(state,indent=2))
write()
try:
 for i,cmd in enumerate(commands):
  with (B/('native_'+str(i)+'.log')).open('x') as f:
   p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT);state.update(worker_pid=p.pid,stage=i);write();rc=p.wait()
  if rc: raise RuntimeError('native stage failed '+str(i)+' '+str(rc))
 state.update(status='native_references_complete');write()
except BaseException as e:
 state.update(status='failed_no_retry',error=str(e),traceback=traceback.format_exc());write();raise
