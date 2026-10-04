"""Score one finite snapshot of newly sealed Viz512 pieces. No polling."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--batch',required=True)
p.add_argument('--all',action='store_true',help='Recompute the finite full snapshot with accepted labels; no GPU work.')
a=p.parse_args();root=Path(a.root).resolve();base=(root/a.run).resolve()
assert base.is_relative_to(root) and base.is_dir()
seen=set()
for receipt in list(base.glob('score_*/receipt.json'))+list(base.glob('cpu_batches/*/receipt.json')):
    for source in json.loads(receipt.read_text()).get('sources',[]):
        if 'path' in source and not a.all:seen.add(str(Path(source['path']).parent))
parts=[]
for raw in sorted(base.glob('*/new_predictions.jsonl'))+sorted(base.glob('*/*/new_predictions.jsonl')):
    folder=raw.parent
    if str(folder) in seen or not (folder/'identity.json').is_file():continue
    if (folder/'complete.json').is_file():
        receipt=json.loads((folder/'complete.json').read_text())
        if receipt.get('passed') is True and receipt.get('completed')==receipt.get('expected'):parts.append(folder)
    elif (folder.parent/'sealed_for_handoff.json').is_file():
        receipt=json.loads((folder.parent/'sealed_for_handoff.json').read_text())
        if receipt.get('passed') and receipt.get('producer_exited') and any(root/x['raw']==raw for x in receipt['source_parts']):parts.append(folder)
if not parts:
    print(json.dumps({'new_sealed_parts':0}));sys.exit(0)
out=base/'cpu_batches'/a.batch
if out.exists():raise FileExistsError(out)
out.parent.mkdir(exist_ok=True)
cmd=[sys.executable,str(root/'workflows/paper_core/score_viz_native.py'),'--out',str(out.relative_to(root)),
     '--reuse-cache',str((base/'qa_reuse/qa_authority.jsonl').relative_to(root))]
for folder in parts:cmd+=['--part',str(folder.relative_to(root))]
# Only explicitly accepted sources enter scoring. A name/glob is not evidence
# of semantic completion; rejected heuristic batches remain outside this list.
manifest=json.loads((base/'annotation_authority.json').read_text())
resolved={}
for entry in manifest['accepted_sources_in_precedence_order']:
    file=base/entry['path'];sha=hashlib.sha256(file.read_bytes()).hexdigest()
    assert sha==entry['sha256']
    for line,txt in enumerate(file.read_text().splitlines(),1):
        value=json.loads(txt)
        assert value['label'] in {'answer_assertive','answer_uncertain','abstain','invalid'}
        assert value['abstain']==(value['label']=='abstain')
        resolved[value['qa_key']]={**value,'accepted_source':{'path':str(file),'sha256':sha,'line':line}}
closed=[x for x in resolved.values() if not x.get('needs_root',False)]
assert all(not x.get('answer_text') or x['answer_text'] in x['answer'] for x in closed)
labels=base/'annotation_resolved'/f'{a.batch}.jsonl';labels.parent.mkdir(exist_ok=True)
if labels.exists():raise FileExistsError(labels)
labels.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in closed))
cmd+=['--decision-file',str(labels.relative_to(root))]
print(json.dumps({'new_sealed_parts':len(parts),'parts':[str(f.relative_to(base)) for f in parts]},ensure_ascii=False),flush=True)
subprocess.run(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
