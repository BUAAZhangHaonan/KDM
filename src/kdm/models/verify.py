"""Native generation token equivalence on the frozen 16-image interface set."""
import argparse,json,time
import numpy as np
from pathlib import Path
from PIL import Image
from kdm.io import read_jsonl,atomic_json,within,file_hash
from kdm.pipeline import make_backend
from kdm.prompts import task_prompt

def main():
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--manifest',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    root=Path(__file__).resolve().parents[3]
    samples=list(read_jsonl(args.manifest))
    if len(samples)!=16 or len({x['id'] for x in samples})!=16: raise ValueError('Expected fixed 16 distinct interface samples')
    spec=json.loads(Path(args.spec).read_text());spec_sha256=file_hash(args.spec);runtime_sources={x.name:file_hash(x) for x in Path(__file__).parent.glob('*.py') if x.name in ['hf.py','backbone.py','internvl_preprocessing.py','sid.py','verify.py']}
    b=make_backend(spec,'cuda:0');rows=[]
    for sample in samples:
        prompt=task_prompt(sample['question'],marker='UNKNOWN',guided=True)
        with Image.open(sample['image_path']) as im:
            image=im.convert('RGB');inputs=b.em.build(image,prompt)
            if getattr(b.em,'mt','')=='internvl_chat': inputs.pop('image_flags')
            with b.torch.inference_mode():
                native=b.model.generate(**inputs,do_sample=False,max_new_tokens=32,eos_token_id=sorted(b.eos),pad_token_id=b.tokenizer.pad_token_id or b.tokenizer.eos_token_id)
            offset=0 if getattr(b.em,'mt','')=='internvl_chat' else inputs['input_ids'].shape[-1]
            native=native[0,offset:].tolist()
            session=b.session(image,prompt);tokens=[]
            for _ in range(32):
                token=int(session.next(tokens).logits.argmax());tokens.append(token)
                if token in b.eos:break
            conditions=[(prompt,'clean'),(prompt,'noise'),(task_prompt(sample['question'],guided=False),'clean'),(task_prompt(sample['question'],guided=False),'noise')]
            prefixes=[(),tuple(native[:1]),tuple(native[:2])]
            independent=[]
            for text,reference in conditions:
                one=b.session(image,text,reference,seed=1729)
                independent.append([one.next(prefix).logits.copy() for prefix in prefixes]);del one
            branches=[b.session(image,text,reference,seed=1729) for text,reference in conditions]
            errors=[]
            for j,prefix in enumerate(prefixes):
                for i,branch in enumerate(branches):
                    errors.append(float(np.max(np.abs(branch.next(prefix).logits-independent[i][j]))))
            del branches
            state_ok=max(errors)==0.
            row={'id':sample['id'],'condition_max_abs_error':max(errors),'condition_state_equal':state_ok,'native_tokens':native,'backend_tokens':tokens,'equal':native==tokens and state_ok,'native_text':b.decode(native),'backend_text':b.decode(tokens)}
            rows.append(row);print(json.dumps(row),flush=True)
            del session
        atomic_json(within(root,args.out),{'runtime_adapter_sha256':runtime_sources,'spec':spec,'spec_sha256':spec_sha256,'manifest_sha256':file_hash(args.manifest),'completed':len(rows),'expected':16,'passed':len(rows)==16 and all(r['equal'] for r in rows),'rows':rows})
    if not all(r['equal'] for r in rows):raise SystemExit('Native token equivalence failed')
if __name__=='__main__':main()
