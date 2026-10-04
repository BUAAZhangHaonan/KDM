"""Adapt the existing input-close sealer to native Viz nested immutable pieces."""
import argparse,ctypes,json,math,os,select,signal,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json,file_hash,read_jsonl,stable_seed,within
from kdm.pipeline import task_id
from kdm.decoding import DecodeConfig
from kdm.prompts import task_prompt
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.viz_author_native import native_task,now
from workflows.paper_core.seal_native_baseline_part import process,wait_exit,complete_lines

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--pid',type=int,required=True)
p.add_argument('--start-tick',required=True);p.add_argument('--owner',required=True);p.add_argument('--target-ready')
p.add_argument('--check-source',action='store_true');p.add_argument('--wait-max-s',type=int,default=60);a=p.parse_args()
assert 1<=a.wait_max_s<=60
source=within(ROOT,a.source);claim_path=source/'claim.json';claim=json.loads(claim_path.read_text())
assert claim['pid']==a.pid and claim['starttick']==a.start_tick and claim['owner']==a.owner
proc=Path('/proc')/str(a.pid);command=process(a.pid)
def owned():
    return process(a.pid)==command and (proc/'stat').read_text().split(') ',1)[1].split()[19]==a.start_tick
assert command and 'workflows/paper_core/viz_author_native.py ' in command
assert f"--model {claim['model']} " in command and f'--output {a.source} ' in command and owned()
for i,card in enumerate(claim['physical_gpus']):
    assert os.readlink(proc/'fd'/str(20+i))==str(ROOT/'outputs/locks'/f'gpu_{card}.lock')
assert not any((source/name).exists() for name in ('complete.json','failure.json','sealed_for_handoff.json'))
listing,samples=roster('viz512')
all_tasks={task_id(claim['model'],native_task(s,m)):native_task(s,m) for m in claim['methods'] for s in samples}
planned=claim.get('planned_keys',list(all_tasks));assert len(planned)==len(set(planned))==claim['expected'] and set(planned)<=all_tasks.keys()

def inventory():
    parts=[];keys=[]
    for raw in sorted(source.glob('*/new_predictions.jsonl')):
        data,rows=complete_lines(raw)
        if not rows:continue
        assert data.endswith(b'\n'),'A source write is incomplete; do not seal it'
        identity=json.loads((raw.parent/'identity.json').read_text())
        ledger=json.loads(raw.with_suffix('.identity.json').read_text())
        assert identity['roster_sha256']==file_hash(listing) and identity['model']==claim['model']
        expected_ids=identity['sample_ids'];assert [r['sample']['id'] for r in rows]==expected_ids[:len(rows)]
        for row in rows:
            task=all_tasks[row['key']]
            assert row['key'] in planned and row['sample']==task['sample'] and row['identity']==ledger['identity']
            assert all(row[k]==task[k] for k in ('method','kind','marker','reference_marker','guided','reference_guided','replicate','implementation_revision'))
            assert row['config']==asdict(DecodeConfig(method=row['method'])) and row['seed']==stable_seed(row['sample']['id'],claim['model'],0)
            assert row['prompt']==task_prompt(row['sample']['question'],guided=False) and row['status']=='ok'
            assert 0<len(row['tokens'])<=32 and len(row['selected_log_probabilities'])==len(row['tokens'])
            assert all(math.isfinite(v) for v in row['selected_log_probabilities']) and math.isfinite(row['first_probability'])
        complete=raw.parent/'complete.json'
        if complete.exists():assert json.loads(complete.read_text())['raw_sha256']==file_hash(raw)
        keys.extend(r['key'] for r in rows)
        parts.append(dict(raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),completed=len(rows),
            expected_original_part=len(expected_ids),original_identity=str((raw.parent/'identity.json').relative_to(ROOT)),
            original_identity_sha256=file_hash(raw.parent/'identity.json'),ledger_identity=ledger['identity']))
    assert len(keys)==len(set(keys)) and set(keys)<=set(planned)
    return parts,keys

initial_parts,initial_keys=inventory();assert 0<len(initial_keys)<len(planned)
if a.check_source:
    print(json.dumps(dict(source_checked=True,model=claim['model'],completed=len(initial_keys),remaining=len(planned)-len(initial_keys),source_modified=False)));sys.exit(0)
target=json.loads(within(ROOT,a.target_ready).read_text());assert target['target_ready'] and target['model']==claim['model'] and target['scientific_parameters_changed'] is False
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK);assert fd>=0
watched=set()
def watches():
    for path in [source,*[x for x in source.iterdir() if x.is_dir()]]:
        if str(path) not in watched:
            assert libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x100)>=0;watched.add(str(path))
watches();stopped=False
try:
    deadline=time.monotonic()+a.wait_max_s
    while True:
        left=deadline-time.monotonic();assert left>0 and select.select([fd],[],[],left)[0],'No new complete input: source stays running'
        os.read(fd,65536);watches()
        count=sum(len(complete_lines(raw)[1]) for raw in source.glob('*/new_predictions.jsonl'))
        if count>len(initial_keys):break
    assert owned();os.kill(a.pid,signal.SIGSTOP);stopped=True
    parts,keys=inventory();assert len(initial_keys)<len(keys)<=len(planned)
    frozen={str(ROOT/part['raw']):part['raw_sha256'] for part in parts}
    assert owned();os.kill(a.pid,signal.SIGTERM);os.kill(a.pid,signal.SIGCONT);stopped=False
    wait_exit(a.pid);assert all(file_hash(path)==sha for path,sha in frozen.items())
    remaining=[dict(key=key,model=claim['model'],method=all_tasks[key]['method'],sample_id=all_tasks[key]['sample']['id']) for key in planned if key not in set(keys)]
    remaining_path=source/'sealed_remaining_keys.jsonl'
    with remaining_path.open('x') as f:
        for row in remaining:f.write(json.dumps(row)+'\n')
    receipt=dict(passed=True,producer_exited=True,model=claim['model'],completed=len(keys),remaining=len(remaining),
        completed_keys=keys,remaining_keys=str(remaining_path.relative_to(ROOT)),remaining_keys_sha256=file_hash(remaining_path),
        original_claim=str(claim_path.relative_to(ROOT)),original_claim_sha256=file_hash(claim_path),
        source_pid=a.pid,source_starttick=a.start_tick,source_owner=a.owner,source_command=command,
        source_parts=parts,roster_sha256=file_hash(listing),target_ready_sha256=file_hash(within(ROOT,a.target_ready)),
        original_raw_modified=False,scientific_parameters_changed=False,stop_trigger='next actual ledger IN_CLOSE_WRITE after fsync',
        zero_completed_remaining_intersection=not set(keys)&{r['key'] for r in remaining},
        original_key_union_complete=set(keys)|{r['key'] for r in remaining}==set(planned),sealed_utc=now())
    assert receipt['zero_completed_remaining_intersection'] and receipt['original_key_union_complete']
    atomic_json(source/'sealed_for_handoff.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('completed_keys','source_parts')}),flush=True)
finally:
    os.close(fd)
    if stopped and process(a.pid)==command:os.kill(a.pid,signal.SIGCONT)
