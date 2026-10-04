"""Use the existing close-write boundary sealer for owned Gemma SID parts."""
import argparse,ctypes,json,math,os,select,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import file_hash,atomic_json,stable_seed
from seal_native_baseline_part import process,wait_exit,complete_lines

def seal(relative):
    source=ROOT/relative
    identity=json.loads((source/'identity.json').read_text())
    claim=json.loads((source/'claim.json').read_text())
    pid=claim['pid'];proc=Path(f'/proc/{pid}')
    if claim['owner']!='/root/sid_gemma_qwen35' or proc.joinpath('stat').read_text().split()[21]!=claim['starttick']:
        raise ValueError('Producer owner or PID birth identity differs')
    command=process(pid)
    if not command or 'workflows/paper_core/sid_gemma_fill.py ' not in command or relative not in command:
        raise ValueError('Producer command does not bind exact source')
    if (source/'complete.json').exists() or (source/'seal_receipt.json').exists():
        raise ValueError('Source already finalized')
    raw=source/'new_predictions.jsonl';original,initial=complete_lines(raw)
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    stopped=False
    try:
        if libc.inotify_add_watch(fd,os.fsencode(raw),0x00000008)<0:
            raise OSError(ctypes.get_errno(),'inotify_add_watch')
        if not select.select([fd],[],[],60)[0]:raise TimeoutError('No completed input; original worker continues')
        os.read(fd,65536);os.kill(pid,signal.SIGSTOP);stopped=True
        content,values=complete_lines(raw)
        ids=[r['sample']['id'] for r in values]
        if len(values)<=len(initial) or not content.endswith(b'\n') or ids!=identity['sample_ids'][:len(values)]:
            raise ValueError('Source is not an exact newly persisted input prefix')
        for r in values:
            if (r['config']!=identity['config'] or r['sid_adapter_sha256']!=identity['sid_adapter_sha256']
                or r['seed']!=stable_seed(r['sample']['id'],'gemma3_4b',0)
                or r['status']!='ok' or not r['tokens'] or len(r['tokens'])>32
                or not all(math.isfinite(x) for x in r['selected_log_probabilities'])):
                raise ValueError('Source identity/configuration/probabilities differ')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False;wait_exit(pid)
        if raw.read_bytes()!=content:raise ValueError('Source changed after stop')
        sealed=source.with_name(source.name+'_sealed_prefix');sealed.mkdir(exist_ok=False)
        (sealed/'new_predictions.jsonl').write_bytes(content)
        new_identity={**identity,'sample_ids':ids,'source_original_identity_sha256':file_hash(source/'identity.json'),
                      'scope':'exact complete prefix sealed before moving remaining keys'}
        atomic_json(sealed/'identity.json',new_identity)
        receipt={'passed':True,'completed':len(values),'expected':len(values),'raw_sha256':file_hash(raw),
                 'whole_original_part_complete':False,'source_path':str(source),
                 'original_expected':len(identity['sample_ids']),'source_identity_sha256':file_hash(source/'identity.json'),
                 'remaining_ids':identity['sample_ids'][len(values):],
                 'completed_ids':ids,'source_claim':claim,'source_command':command,
                 'stop_trigger':'actual next IN_CLOSE_WRITE; then SIGSTOP, verified bytes, TERM/CONT, exit verified',
                 'producer_running':False,'sealed_prefix_path':str(sealed),
                 'score_status':'pending'}
        atomic_json(source/'seal_receipt.json',receipt);atomic_json(sealed/'complete.json',receipt)
        return {'source':relative,'sealed_prefix':str(sealed.relative_to(ROOT)),
                'completed':len(values),'remaining':len(receipt['remaining_ids']),
                'receipt':str((source/'seal_receipt.json').relative_to(ROOT))}
    finally:
        os.close(fd)
        if stopped and process(pid)==command:os.kill(pid,signal.SIGCONT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(seal(a.source)))
