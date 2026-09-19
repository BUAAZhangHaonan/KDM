"""Run command-line contracts in a fresh CPU fixture and validate completeness."""
from pathlib import Path
import json,os,subprocess,sys
from datetime import datetime,timezone
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
fixture=ROOT/'outputs'/'verification'/('cli_fixture_'+stamp)
fixture.mkdir(parents=True)
fixture_rel=str(fixture.relative_to(ROOT))
image=fixture/'fixture.png';Image.new('RGB',(16,16),(100,120,130)).save(image)
samples=[{'id':f'fixture-{i}','cluster':str(i),'dataset':'fixture','split':'eval','question':'What is shown?',
          'gold':['milk'],'image_path':str(image)} for i in range(4)]
manifest=fixture/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in samples))
tmp=ROOT/'cache/tmp';tmp.mkdir(parents=True,exist_ok=True)
env={**os.environ,'TMPDIR':str(tmp),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),'MPLBACKEND':'Agg','CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
results=[]
def run(args):
    print('RUN',args[0:3],flush=True)
    p=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,text=True,capture_output=True,check=True,timeout=60)
    results.append({'command':args,'exit_code':p.returncode,'stdout':p.stdout[-1500:],'stderr':p.stderr[-1000:]})

for script in sorted((ROOT/'scripts').glob('*.py')):run([str(script),'--help'])
run(['-m','kdm.cli','--help'])
for mode in ('census','experiment','probe'):
    dest=f'{fixture_rel}/{mode}.jsonl'
    command=['-m','kdm.cli','--root',str(ROOT),'run','--manifest',str(manifest),'--model-spec',str(ROOT/'configs/kdm/model_spec_mock.json'),
             '--model','cpu_fixture','--mode',mode,'--gpu','0','--out',dest]
    run(command);run(command)
    run(['scripts/verify_complete.py','--root',str(ROOT),'--manifest',str(manifest),
         '--records',str(ROOT/dest),'--model','cpu_fixture','--mode',mode,
         '--out',f'{fixture_rel}/{mode}_complete.json'])
run(['-m','kdm.cli','--root',str(ROOT),'annotation-queue','--records',str(fixture/'experiment.jsonl'),str(fixture/'probe.jsonl'),
     '--out',f'{fixture_rel}/annotation_queue.jsonl'])
for command in ('replay','mechanism'):
    run(['-m','kdm.cli','--root',str(ROOT),command,'--records',str(fixture/'experiment.jsonl'),
         '--model-spec',str(ROOT/'configs/kdm/model_spec_mock.json'),'--model','cpu_fixture','--gpu','0',
         '--out',f'{fixture_rel}/{command}.jsonl'])
annotations=[]
for line in (fixture/'annotation_queue.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['label'] is None:
        r.update(label='answer_assertive',evidence=r['text'],answer_text=r['text'],source='synthetic_fixture_annotation')
    annotations.append(r)
annotation_path=fixture/'complete_annotations.jsonl'
annotation_path.write_text(''.join(json.dumps(r)+'\n' for r in annotations))
run(['scripts/complete_response_audit.py','--root',str(ROOT),'--records',str(fixture/'experiment.jsonl'),'--manifest',str(manifest),
     '--annotations',str(annotation_path),'--model-spec',str(ROOT/'configs/kdm/model_spec_mock.json'),
     '--model','cpu_fixture','--gpu','0','--out',f'{fixture_rel}/complete_response.jsonl'])
run(['scripts/human_review.py','--root',str(ROOT),'queue','--records',str(fixture/'experiment.jsonl'),str(fixture/'probe.jsonl'),
     '--annotations',str(annotation_path),'--out',f'{fixture_rel}/human_review_queue.jsonl'])
# External decision files below are explicit CPU-test fixtures, never real human review.
review_rows=[json.loads(line) for line in (fixture/'human_review_queue.jsonl').read_text().splitlines()]
external=fixture/'SYNTHETIC_EXTERNAL_DECISIONS.jsonl'
external.write_text(''.join(json.dumps({'key':row['key'],'text_sha256':row['text_sha256'],
    'action':'approve','revision':1,'previous_revision':0,'reviewer':'CPU_SYNTHETIC_TEST_NOT_A_PERSON',
    'reviewed_at':datetime.now(timezone.utc).isoformat(),'fixture_only':True})+'\n' for row in review_rows))
run(['scripts/human_review.py','--root',str(ROOT),'merge','--queue',str(fixture/'human_review_queue.jsonl'),
     '--decisions',str(external),'--out',f'{fixture_rel}/SYNTHETIC_REVIEWED_ANNOTATIONS.jsonl'])
(fixture/'CLI_REVIEW.json').write_text(json.dumps({'data':'CPU mock model and synthetic fixture; no real model inference',
     'commands':results,'all_passed':True},indent=2))
print('Validated',len(results),'command invocations')
