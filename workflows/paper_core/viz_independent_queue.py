"""Finite released-GPU queue; no retries, new experiments, or periodic monitor."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from kdm.io import atomic_json,read_jsonl
BASE=ROOT/'outputs/paper_core_20261002_dev_viz/closeout_20261004/viz'

def take(models,host,gpu):
    directory=BASE/'generation';directory.mkdir(parents=True,exist_ok=True)
    with open(directory/'queue.lock','a+') as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        state_path=directory/'queue_claims.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
        candidates=[]
        for model in models:
            pilot=directory/model/'pilot8/complete.json'
            if not pilot.exists():continue
            evidence=json.loads(pilot.read_text());assert evidence['generation_complete'] and evidence['answers']==80
            if any(v['model']==model and v['status']=='failed' for v in state.values()):continue
            groups={}
            for row in read_jsonl(directory/model/'pilot8/independent.jsonl'):groups.setdefault(row['sample']['id'],row['wall_s'])
            costs=list(groups.values())[1:];mean=sum(costs)/len(costs)
            free=[]
            for n in range(4):
                piece=f'tail_s{n}of4';key=model+'/'+piece
                if key in state:continue
                complete=directory/model/piece/'complete.json'
                assert not complete.exists(),'Unregistered existing result '+str(complete)
                free.append((piece,key))
            # Assign longest remaining model first; other released workers take
            # the next unclaimed piece under the same short critical section.
            for piece,key in free:candidates.append((mean*len(free),model,piece,key,mean))
        if not candidates:return None
        _,model,piece,key,mean=max(candidates)
        task={'model':model,'piece':piece,'status':'claimed','host':host,'gpu':gpu,'supervisor_pid':os.getpid(),'created_unix':time.time(),'pilot_warm_mean_image_s':mean}
        state[key]=task;atomic_json(state_path,state)
        return key,task

def update(key,changes):
    d=BASE/'generation'
    with open(d/'queue.lock','a+') as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX);p=d/'queue_claims.json';s=json.loads(p.read_text());s[key].update(changes);atomic_json(p,s)

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    while selected:=take(a.models,a.host,a.gpu):
        key,task=selected;model=task['model'];piece=task['piece'];d=BASE/'generation'/model;d.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(ROOT/'workflows/paper_core/viz_independent_resume.py'),'--model',model,'--host',a.host,'--gpu',a.gpu,'--pieces',piece,'--run',piece]
        with open(d/(piece+'.log'),'x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            update(key,{'status':'running','pid':child.pid,'command':cmd,'started_unix':time.time()});code=child.wait()
        receipt=d/piece/'complete.json'
        good=code==0 and receipt.exists() and json.loads(receipt.read_text()).get('generation_complete') is True
        update(key,{'status':'complete' if good else 'failed','exit_code':code,'finished_unix':time.time(),'complete_receipt':str(receipt) if receipt.exists() else None})
    atomic_json(BASE/'generation'/f'queue_{a.host}_{a.gpu}_exit.json',{'pid':os.getpid(),'models':a.models,'finished_unix':time.time(),'status':'no_more_eligible_unclaimed_shards','new_timer_created':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--host',required=True);p.add_argument('--gpu',required=True);run(p.parse_args())
