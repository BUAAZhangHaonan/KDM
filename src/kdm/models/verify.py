"""Native generation token equivalence on the frozen 16-image interface set."""
import argparse,json,time,math,os
import numpy as np
from pathlib import Path
from PIL import Image
from kdm.execution import resolve_image_path
from kdm.io import read_jsonl,atomic_json,within,file_hash
from kdm.pipeline import make_backend
from kdm.prompts import task_prompt

def main():
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--manifest',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    root=Path(__file__).resolve().parents[3]
    samples=list(read_jsonl(args.manifest))
    if len(samples)!=16 or len({x['id'] for x in samples})!=16: raise ValueError('Expected fixed 16 distinct interface samples')
    spec=json.loads(Path(args.spec).read_text());spec_sha256=file_hash(args.spec);runtime_sources={x.name:file_hash(x) for x in Path(__file__).parent.glob('*.py') if x.name in ['hf.py','backbone.py','internvl_preprocessing.py','remote.py']}
    if spec.get("factory", "").partition(":")[0] == "kdm.models.internvl_dual":
        runtime_sources["internvl_dual.py"]=file_hash(Path(__file__).parent/"internvl_dual.py")
    verification_script_sha256=file_hash(__file__)
    from kdm.protocol import validate_execution_runtime
    execution=validate_execution_runtime(root,spec,spec['key'],os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))
    b=make_backend(spec,'cuda:0');rows=[];layer_check={'status':'not_run'}
    if getattr(b.em,'mt','') not in {'minicpmv','phi3_v'}:runtime_sources.pop('remote.py',None)
    if getattr(b.em,'mt','')!='internvl_chat':runtime_sources.pop('internvl_preprocessing.py',None)
    for sample in samples:
        prompt=task_prompt(sample['question'],marker='UNKNOWN',guided=True)
        with Image.open(resolve_image_path(sample['image_path'], root)) as im:
            image=im.convert('RGB');inputs=b.em.build(image,prompt)
            with b.torch.inference_mode():
                if hasattr(b.em,'native_generate'):
                    native=b.em.native_generate(inputs,max_new_tokens=32)
                else:
                    if getattr(b.em,'mt','')=='internvl_chat':inputs.pop('image_flags')
                    native=b.model.generate(**inputs,do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=32,eos_token_id=sorted(b.eos),pad_token_id=b.tokenizer.pad_token_id or b.tokenizer.eos_token_id)
                    offset=0 if getattr(b.em,'mt','')=='internvl_chat' else inputs['input_ids'].shape[-1]
                    native=native[0,offset:].tolist()
            if not rows:
                try:
                    layer_session=b.session(image,prompt,need_layers=True)
                    layer_step=layer_session.next(())
                    hidden=layer_session.output.hidden_states;last=hidden[-1][:,-1]
                    with b.torch.inference_mode():
                        projected=b.em.lm_head(hidden[-1])[:,-1]
                        cfg=getattr(b.em.cfg,'text_config',b.em.cfg)
                        cap=getattr(cfg,'final_logit_softcapping',None)
                        if cap is not None:projected=(projected/cap).tanh()*cap
                        final_error=float(np.max(np.abs(projected[0].float().cpu().numpy()-layer_step.logits)))
                        idx=min(layer_step.early_normalized)
                        raw=b.em.lm_head(hidden[idx][:,-1])[0].float().cpu().numpy()
                        normalized=b.em.lm_head(b.em.norm(hidden[idx][:,-1]))[0].float().cpu().numpy()
                    raw_error=float(np.max(np.abs(raw-layer_step.early_raw[idx])))
                    norm_error=float(np.max(np.abs(normalized-layer_step.early_normalized[idx])))
                    layer_check={'status':'passed' if max(final_error,raw_error,norm_error)==0. else 'failed',
                         'sample_id':sample['id'],'raw_layers':len(layer_step.early_raw),'normalized_layers':sorted(layer_step.early_normalized),
                         'head_final_error':final_error,'raw_projection_error':raw_error,'norm_projection_error':norm_error,
                         'scope':'layer projection interface only; method formulas and SID need separate validation'}
                    del layer_session
                except Exception as exc:
                    layer_check={'status':'failed','error':type(exc).__name__+': '+str(exc)}
                layer_session=hidden=last=projected=layer_step=raw=normalized=None
            session=b.session(image,prompt);tokens=[]
            for _ in range(32):
                token=int(session.next(tokens).logits.argmax());tokens.append(token)
                if token in b.eos:break
            conditions=[(prompt,'clean'),(prompt,'noise'),(task_prompt(sample['question'],guided=False),'clean'),(task_prompt(sample['question'],guided=False),'noise'),(prompt,'text_only'),(task_prompt(sample['question'],guided=False),'text_only')]
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
            row={'id':sample['id'],'checked_conditions':['guided_clean','guided_noise','unguided_clean','unguided_noise','guided_text_only','unguided_text_only'],'condition_max_abs_error':max(errors),'condition_state_equal':state_ok,'native_tokens':native,'backend_tokens':tokens,'equal':native==tokens and state_ok,'native_text':b.decode(native),'backend_text':b.decode(tokens)}
            rows.append(row);print(json.dumps(row),flush=True)
            del session
        atomic_json(within(root,args.out),{'execution':execution,'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES','').split(','),'hf_device_map':{k:str(v) for k,v in getattr(b.model,'hf_device_map',{}).items()},'peak_allocated_bytes':{str(i):b.torch.cuda.max_memory_allocated(i) for i in range(b.torch.cuda.device_count())},'native_greedy_overrides':({'do_sample':False,'max_new_tokens':32} if hasattr(b.em,'native_generate') else {'do_sample':False,'num_beams':1,'repetition_penalty':1.0,'max_new_tokens':32}),'checkpoint_generation_config':(b.model.generation_config.to_dict() if getattr(b.model,'generation_config',None) is not None else None),'verification_script_sha256':verification_script_sha256,'layer_projection_check':layer_check,'runtime_adapter_sha256':runtime_sources,'spec':spec,'spec_sha256':spec_sha256,'manifest_sha256':file_hash(args.manifest),'completed':len(rows),'expected':16,'passed':len(rows)==16 and all(r['equal'] for r in rows),'rows':rows})
    if not all(r['equal'] for r in rows):raise SystemExit('Native token equivalence failed')
if __name__=='__main__':main()
