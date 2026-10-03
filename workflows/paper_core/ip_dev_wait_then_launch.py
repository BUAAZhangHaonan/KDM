#!/usr/bin/env python3
"""One authorized missing-key claim after its owned predecessor actually completes."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import select
import subprocess
import time

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')

def pidfd_open(pid):
    # The registered interpreters do not all expose Python's os.pidfd_open.
    # This is the same Linux pidfd syscall, with no polling or change to inference.
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('The explicit Linux x86_64 event guardian is unavailable')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype=ctypes.c_long
    fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return int(fd)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schedule',required=True);args=p.parse_args()
    schedule=json.loads(Path(args.schedule).read_text());parent=schedule['predecessor'];d=schedule['definition'];root=Path(d['root'])
    base=root/'outputs/paper_core_20261002_dev_viz'/d['run']/d['claim'];launch=base/'launch'
    proc=Path('/proc')/str(parent['launch']['pid'])
    if proc.exists()and(proc/'stat').read_text().split(') ',1)[1].split()[0]!='Z':
        if (proc/'stat').read_text().split(') ',1)[1].split()[19]!=parent['launch']['start_tick']:
            raise ValueError('Predecessor PID identity changed')
        if parent['definition']['claim'].encode()not in(proc/'cmdline').read_bytes().split(b'\0'):
            raise ValueError('Predecessor entry/claim changed')
        fd=pidfd_open(parent['launch']['pid']);select.select([fd],[],[]);os.close(fd)
    old=root/'outputs/paper_core_20261002_dev_viz'/parent['definition']['run']/parent['definition']['claim']
    receipt=json.loads((old/'complete_receipt.json').read_text());identity=json.loads((old/'identity.json').read_text())
    if (not receipt['passed'] or receipt['actual_rows']!=parent['definition']['n'] or (old/'failure.json').exists()
            or identity['plan']['missing_keys_sha256']!=parent['definition']['input_sha256']):
        raise ValueError('Actual predecessor completion is absent or belongs to another key claim')
    write(launch/'predecessor_actual_complete.json',{'source':str(old/'complete_receipt.json'),
          'sha256':hashlib.sha256((old/'complete_receipt.json').read_bytes()).hexdigest(),'actual_rows':receipt['actual_rows']})
    env=os.environ.copy()
    if d.get('slot')is not None:env['KDM_REQUEST_SLOT']=str(d['slot'])
    with(launch/'stdout.log').open('x')as out,(launch/'stderr.log').open('x')as err:
        job=subprocess.Popen(schedule['argv'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    write(launch/'actual_launch.json',{'pid':job.pid,'start_tick':(Path('/proc')/str(job.pid)/'stat').read_text().split(') ',1)[1].split()[19],
          'argv':schedule['argv'],'actual_planned_rows':d['n'],'slot_requested':d.get('slot'),'launched_unix':time.time(),
          'predecessor_exit_verified':True,'claim_id':d['claim']})
    code=job.wait();write(launch/'supervisor_actual_exit.json',{'producer_pid':job.pid,'exit_code':code,'exit_unix':time.time(),'automatic_retry':False})

if __name__=='__main__':main()
