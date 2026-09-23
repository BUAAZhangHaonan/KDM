#!/usr/bin/env python3
"""Adopt running jobs with Linux pidfds; dispatch each fresh successor once."""
import argparse,json,os,select,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);out=root/cfg['status_path']
def save():
 temp=out.with_suffix('.tmp');temp.write_text(json.dumps(cfg,indent=2));os.replace(temp,out)
poll=select.poll();fds={}
for lane in cfg['lanes']:
 try:fd=os.pidfd_open(lane['pid'])
 except ProcessLookupError:fd=None
 lane['state']='adopted';lane['history']=[]
 if fd is not None:poll.register(fd,select.POLLIN);fds[fd]=lane
 else:lane['state']='adopted_already_exited'
save()
def advance(lane):
 done=root/lane['complete_path']
 if not done.is_file():
  lane['state']='stopped_missing_completion';save();return
 lane['history'].append({'pid':lane['pid'],'complete_path':lane['complete_path'],'ended_unix':time.time()})
 if not lane['pending']:lane['state']='complete';save();return
 job=lane['pending'].pop(0);log=root/job['log'];log.parent.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(job.get('env',{}))
 with log.open('x') as f:child=subprocess.Popen(job['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 lane.update(pid=child.pid,complete_path=job['complete_path'],state='running',current=job,started_unix=time.time())
 fd=os.pidfd_open(child.pid);poll.register(fd,select.POLLIN);fds[fd]=lane;save()
for lane in cfg['lanes']:
 if lane['state']=='adopted_already_exited':advance(lane)
while fds:
 for fd,event in poll.poll():
  lane=fds.pop(fd);poll.unregister(fd);os.close(fd);advance(lane)
save()
