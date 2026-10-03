#!/usr/bin/env python3
"""Seal one owned IP-dev ledger at the next persisted complete input."""
import argparse,ctypes,json,os,pathlib,select,signal,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.paper_core.ip_dev_pilot import tasks_for,audit,write
from workflows.paper_core.dev_viz import roster,now
from kdm.pipeline import task_id
from kdm.io import file_hash,read_jsonl,within

def main():
 p=argparse.ArgumentParser();p.add_argument('--claim',required=True);p.add_argument('--run',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--start-tick',required=True);p.add_argument('--target-ready');p.add_argument('--check-source',action='store_true');a=p.parse_args()
 b=ROOT/'outputs/paper_core_20261002_dev_viz'/a.run/a.claim;plan=json.loads((b/'claim.json').read_text());identity=json.loads((b/'identity.json').read_text());proc=pathlib.Path('/proc')/str(a.pid)
 def live():return proc.exists()and(proc/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
 assert live()and(proc/'stat').read_text().split(') ',1)[1].split()[19]==a.start_tick
 cmd=(proc/'cmdline').read_bytes();assert str(ROOT/'workflows/paper_core/ip_dev_full.py').encode()in cmd and a.claim.encode()in cmd and identity['pid']==a.pid
 assert identity['plan']==plan and file_hash(ROOT/'workflows/paper_core/ip_dev_full.py')==plan['runner_sha256']
 cards=identity['actual_admission']['physical_gpus'];assert [os.readlink(proc/'fd'/str(20+i))for i in range(len(cards))]==[str(ROOT/'outputs/locks'/('gpu_'+g+'.lock'))for g in cards]
 assert not any((b/n).exists()for n in ['complete_receipt.json','failure.json','sealed_partial_receipt.json'])
 inp=within(ROOT,plan['missing_keys_path']);assert file_hash(inp)==plan['missing_keys_sha256']
 original=[x['key']for x in read_jsonl(inp)];_,samples=roster('dev404');tasks={task_id(plan['model'],t):t for t in tasks_for(samples,plan['methods'],plan['markers'])}
 assert set(original)<=tasks.keys()and plan['scientific_parameters_changed']is False
 def inventory():
  parts=[];keys=[]
  for raw in sorted((b/'sealed_parts').glob('*.jsonl')):
   data=raw.read_bytes();assert data.endswith(b'\n');rows=list(read_jsonl(raw));assert rows
   selected=[tasks[x['key']]for x in rows];audit(plan['model'],selected,raw)
   keys.extend(x['key']for x in rows);parts.append((raw,rows))
  assert len(keys)==len(set(keys))and set(keys)<=set(original)
  return parts,keys
 initial=sum(len(x.read_bytes().splitlines())for x in(b/'sealed_parts').glob('*.jsonl'));assert 0<initial<len(original)-8
 if a.check_source:print(json.dumps(dict(source_checked=True,actual_rows=initial,remaining=len(original)-initial,source_modified=False)));return
 target=json.loads(within(ROOT,a.target_ready).read_text());assert target['target_ready']and target['model']==plan['model']and target['scientific_parameters_changed']is False
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK);assert fd>=0 and libc.inotify_add_watch(fd,os.fsencode(b/'sealed_parts'),0x8)>=0
 stopped=False
 try:
  deadline=time.monotonic()+60
  while True:
   left=deadline-time.monotonic();assert left>0 and select.select([fd],[],[],left)[0],'No complete-input close event; source remains running'
   os.read(fd,65536);count=sum(len(x.read_bytes().splitlines())for x in(b/'sealed_parts').glob('*.jsonl'))
   if count>initial:break
  assert live()and(proc/'cmdline').read_bytes()==cmd
  os.kill(a.pid,signal.SIGSTOP);stopped=True;parts,keys=inventory();assert initial<len(keys)<len(original)-8
  frozen={str(raw):file_hash(raw)for raw,rows in parts};os.kill(a.pid,signal.SIGTERM);os.kill(a.pid,signal.SIGCONT);stopped=False
  pidfd=libc.syscall(434,a.pid,0)
  if pidfd>=0:
   assert select.select([pidfd],[],[],10)[0],'Owned source did not exit';os.close(pidfd)
  assert not live()and all(file_hash(path)==sha for path,sha in frozen.items())
  for raw,rows in parts:
   complete=raw.with_suffix('.complete.json')
   if complete.exists():assert json.loads(complete.read_text())['raw_sha256']==file_hash(raw);continue
   write(complete,dict(passed=True,model=plan['model'],method=rows[0]['method'],marker=rows[0]['marker'],rows=len(rows),raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),completed_keys=[x['key']for x in rows],generation_wall_s=sum(x['wall_s']for x in rows),output_tokens=sum(len(x['tokens'])for x in rows),completed_at_utc=now(),scope='budgeted_dev404',immutable_source_prefix=True))
  remaining=[x for x in read_jsonl(inp)if x['key']not in set(keys)]
  with(b/'sealed_remaining_keys.jsonl').open('x')as stream:
   for x in remaining:stream.write(json.dumps(x)+'\n')
  receipt=dict(schema='kdm_IP_dev_owned_complete_input_seal_v1',passed=True,source_claim=a.claim,source_pid=a.pid,source_start_tick=a.start_tick,completed=len(keys),remaining=len(remaining),completed_keys=keys,remaining_keys=str((b/'sealed_remaining_keys.jsonl').relative_to(ROOT)),remaining_keys_sha256=file_hash(b/'sealed_remaining_keys.jsonl'),original_claim_sha256=file_hash(b/'claim.json'),original_identity_sha256=file_hash(b/'identity.json'),target_ready_sha256=file_hash(within(ROOT,a.target_ready)),parts=[dict(raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),rows=len(rows))for raw,rows in parts],original_raw_modified=False,scientific_parameters_changed=False,stop_trigger='next actual ledger close then verified immutable complete-response rows',sealed_utc=now())
  # Actual source time is taken from its launch UTC, with loading included.
  import datetime
  started=datetime.datetime.fromisoformat(json.loads((b/'launch/actual_launch.json').read_text())['launched_utc']);receipt['actual_gpu_seconds_including_load']=(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds()*identity['runtime_spec']['gpu_count']
  write(b/'sealed_partial_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items()if k not in ['parts','completed_keys']}))
 finally:
  os.close(fd)
  if stopped and live()and(proc/'cmdline').read_bytes()==cmd:os.kill(a.pid,signal.SIGCONT)
if __name__=='__main__':main()
