import hashlib,json,os,re,socket,sys
from argparse import Namespace
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]);os.chdir(root);sys.path[:0]=[str(root/'src'),str(root)]
from workflows.hallusion_blind.generate import load_plan,seal
from kdm.io import file_hash,read_jsonl
b=root/'outputs/hallusion_blind_20261005'
conditions=json.loads((b/'registration/conditions.json').read_text())
results=[]
for output_name in ['runs','runs_continuation']:
    for model in (b/output_name).glob('*'):
        if not model.is_dir():continue
        plan=load_plan(Namespace(model=model.name,methods=[c['method'] for c in conditions if c['model']==model.name],protocol=str((b/'registration/protocol.json').relative_to(root)),output=str((b/output_name).relative_to(root)),sample_ids=None))
        eos=set()
        evidence=[]
        for directory in [b/'runs'/model.name,b/'runs_continuation'/model.name]:
            for receipt in directory.glob('claims/*/chunk_*.complete.json'):
                record=json.loads(receipt.read_text())
                if record['identity']==plan['identity']:
                    eos.update(record['eos_token_ids']);evidence.append({'path':str(receipt.relative_to(root)),'sha256':file_hash(receipt)})
        for failed in model.glob('claims/*/failed.json'):
            run=failed.parent;owner=json.loads((run/'owner.json').read_text())
            if owner['hostname']!=socket.gethostname():continue
            proc=Path('/proc')/str(owner['pid'])/'stat'
            if proc.exists():
                fields=proc.read_text().rsplit(')',1)[1].split()
                if fields[0]!='Z' and int(fields[19])==owner['start_tick']:raise ValueError('Source process still live')
            for path in run.glob('chunk_*.pending.jsonl'):
                rows=list(read_jsonl(path))
                if not rows:continue
                if not eos:raise ValueError('No existing bound EOS evidence')
                if any(file_hash(root/name)!=sha for name,sha in owner['source_sha256'].items()):raise ValueError('Frozen source differs')
                if any(r['key'] not in owner['keys'] or not r['terminated'] or r['tokens'][-1] not in eos for r in rows):raise ValueError('Pending row lacks owned actual EOS')
                keys=[r['key'] for r in rows];number=int(re.fullmatch(r'chunk_(\d+)\.pending\.jsonl',path.name).group(1))
                original_sha=file_hash(path)
                sealed=seal(plan,run,owner,number,path,keys,eos)
                proof={'status':'actual_existing_EOS_sealed_without_generation','utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'keys':keys,'original_pending_sha256':original_sha,'source_pid':owner['pid'],'source_start_tick':owner['start_tick'],'source_process_running':False,'failure_source':str(failed.relative_to(root)),'failure_sha256':file_hash(failed),'original_sealer_sha256':file_hash(root/'workflows/hallusion_blind/generate.py'),'eos_evidence':evidence[:1],'sealed_return':sealed,'new_generation':False,'retry':False,'ownership_release':False}
                target=run/f'post_failure_seal_{number:05d}.json'
                with target.open('x') as stream:json.dump(proof,stream,indent=2)
                results.append({'model':model.name,'claim':run.name,'rows':len(rows)})
print(json.dumps({'host':socket.gethostname(),'salvaged_rows':sum(x['rows'] for x in results),'claims':results},ensure_ascii=False))
