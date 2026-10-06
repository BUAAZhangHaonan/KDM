"""Reuse real greedy trajectories at the newly authorized 128-token stopping point.

No model generation is performed. The original EOS rows are immutable and each
new row points to their exact file, line and SHA. Only the output budget changes.
"""
from __future__ import annotations
import argparse, copy, json, shutil, sys, time
from argparse import Namespace
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from kdm.io import file_hash, read_jsonl, stable_hash
from workflows.hallusion_blind import generate128 as runner
from workflows.general_vqa_direct.generate import now

@lru_cache(maxsize=64)
def source_rows(path, expected_sha):
    path = Path(path)
    if file_hash(path) != expected_sha:
        raise ValueError('Original trajectory source SHA differs')
    return list(read_jsonl(path))

def validate_source(row, plan, root=ROOT):
    binding = row.get('budget_reuse')
    if not isinstance(binding, dict):
        raise ValueError('Reused output lacks its complete original binding')
    path = Path(binding['original_raw'])
    if not path.is_absolute():
        path = root / path
    old = source_rows(str(path), binding['original_raw_sha256'])[binding['original_line']-1]
    if (old['key'] != binding['original_key'] or old['identity'] != binding['original_identity']
        or old['claim_identity'] != binding['original_claim_identity']
        or old.get('terminated') is not True or old.get('status') != 'ok'
        or old['tokens'][-1] not in binding['original_eos_token_ids']):
        raise ValueError('Original EOS/source/ownership binding differs')
    for field in ('model','method','kind','marker','reference_marker','guided','reference_guided','replicate','sample','seed','prompt'):
        if row.get(field) != old.get(field):
            raise ValueError('Reuse changed a non-budget input: '+field)
    expected = dict(old['config']); expected['max_tokens'] = 128
    if row['config'] != expected or row['tokens'] != old['tokens'][:128]:
        raise ValueError('Reuse must preserve actual tokens and all non-budget parameters')
    if len(old['tokens']) <= 128:
        if row['text'] != old['text'] or binding['mode'] != 'EOS_within_budget':
            raise ValueError('EOS-before-budget text cannot change')
    else:
        proof = binding.get('decoder_proof', {})
        if (binding['mode'] != 'prefix_of_recorded_greedy_trajectory'
            or proof.get('full_original_text_roundtrip') is not True
            or proof.get('prefix_tokens_sha256') != stable_hash(row['tokens'])
            or proof.get('prefix_text_sha256') != stable_hash(row['text'])
            or not proof.get('tokenizer_files_sha256')):
            raise ValueError('Prefix reuse requires verified original tokenizer decoding')

def convert_row(old, source, line, sha, eos, plan, key, owner, tokenizer=None, tokenizer_proof=None):
    item = plan['tasks'][key]
    if len(old['tokens']) > 128 and tokenizer is None:
        raise ValueError('Long response needs original-tokenizer prefix decoding')
    value = copy.deepcopy(old)
    value.update(item, key=key, identity=plan['identity'], claim_identity=stable_hash(owner),
                 generation_source='reused_budget128', tokens=old['tokens'][:128],
                 config=runner.expected_config(plan,item,old), timing_usable=False)
    long = len(old['tokens']) > 128
    if long:
        full = tokenizer.decode(old['tokens'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if full != old['text']:
            raise ValueError('Registered tokenizer does not reproduce original full text')
        value['text'] = tokenizer.decode(value['tokens'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if isinstance(old.get('trace'),list):value['trace']=old['trace'][:128]
        if isinstance(old.get('selected_log_probabilities'),list):
            value['selected_log_probabilities']=old['selected_log_probabilities'][:128]
            value['sequence_log_probability']=sum(value['selected_log_probabilities'])
        else:
            value.pop('sequence_log_probability',None)
        for field, flag in [('negative_wa_steps','negative_wa'),('zero_sum_extension_steps','zero_sum_extension')]:
            if field in value:value[field]=sum(bool(t.get(flag)) for t in value.get('trace',[]))
        # Original complete-response timing is not the unknown first128 generation time.
        value['wall_s']=0.0
    ended=value['tokens'][-1] in eos
    value.update(terminated=ended,truncated=not ended,finish_reason='eos' if ended else 'length')
    binding={'mode':'prefix_of_recorded_greedy_trajectory' if long else 'EOS_within_budget',
             'original_raw':str(source),'original_line':line,'original_raw_sha256':sha,
             'original_key':old['key'],'original_identity':old['identity'],
             'original_claim_identity':old['claim_identity'],'original_eos_token_ids':sorted(eos),
             'original_tokens':len(old['tokens']),'original_wall_s':old['wall_s'],
             'original_generation_source':old['generation_source'],'original_budget':old['config']['max_tokens'],
             'no_new_generation':True}
    if long:
        binding['decoder_proof']={**tokenizer_proof,'full_original_text_roundtrip':True,
            'prefix_tokens_sha256':stable_hash(value['tokens']),'prefix_text_sha256':stable_hash(value['text'])}
    value['budget_reuse']=binding
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-raw-list',type=Path,required=True)
    parser.add_argument('--output',default='outputs/hallusion_blind128_20261006/runs')
    parser.add_argument('--protocol',default='outputs/hallusion_blind128_20261006/registration/protocol.json')
    args=parser.parse_args()
    protocol=json.loads((ROOT/args.protocol).read_text())
    conditions=json.loads((ROOT/args.protocol).parent.joinpath(protocol['conditions']).read_text())
    inputs={m:[] for m in protocol['models']}
    source_cache=ROOT/'outputs/hallusion_blind128_20261006/source_trajectories'
    source_cache.mkdir(parents=True,exist_ok=True)
    seen=set()
    for name in args.old_raw_list.read_text().splitlines():
        if not name.strip():continue
        source=Path(name);sha=file_hash(source)
        receipt=json.loads(source.with_name(source.stem+'.complete.json').read_text())
        if receipt['raw_sha256']!=sha or receipt['status']!='complete':raise ValueError('Original chunk receipt differs')
        eos=set(receipt['eos_token_ids'])
        retained=source_cache/(sha+'.jsonl')
        if retained.exists():
            if file_hash(retained)!=sha:raise ValueError('Retained original trajectory collision')
        else:
            shutil.copyfile(source,retained);retained.chmod(0o444)
        for line,old in enumerate(read_jsonl(source),1):
            idt=(old['model'],old['key'])
            if idt in seen:raise ValueError('Original validated list repeats a key')
            seen.add(idt);inputs[old['model']].append((old,retained.relative_to(ROOT),line,sha,eos))
    summary={'created_utc':now(),'original_raw_list_sha256':file_hash(args.old_raw_list),'models':[]}
    for model,data in inputs.items():
        if not data:continue
        ns=Namespace(model=model,methods=[c['method'] for c in conditions if c['model']==model],
                     protocol=args.protocol,output=args.output,sample_ids=None,chunk_rows=8,
                     claim_id=model+'_compatible_budget128_reuse',owner='/root')
        plan=runner.load_plan(ns)
        lookup={(t['method'],t['sample']['id']):k for k,t in plan['tasks'].items()}
        keys=[lookup[(r['method'],r['sample']['id'])] for r,*_ in data]
        if len(keys)!=len(set(keys)):raise ValueError('Budget transition creates duplicate keys')
        plan['selected']=keys
        tokenizer=None;proof=None
        if any(len(r['tokens'])>128 for r,*_ in data):
            from transformers import AutoTokenizer
            declared=ROOT/'outputs/hallusion_blind128_20261006/registration/tokenizer_paths.json'
            clones=json.loads(declared.read_text()) if declared.exists() else {}
            path=Path(clones[model]['path'] if model in clones else plan['spec']['kwargs']['model_path'])
            if model in clones:
                if any(file_hash(path/n)!=sha for n,sha in clones[model]['files_sha256'].items()):
                    raise ValueError('Explicit CPU tokenizer clone differs from its original checkpoint')
            tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
            files={p.name:file_hash(p) for p in path.iterdir() if p.is_file() and
                   (p.name.startswith('tokenizer') or p.name in ['special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'])}
            proof={'checkpoint':plan['spec']['hf_model_id'],'tokenizer_path':str(path),
                   'tokenizer_files_sha256':files,'decoder':'registered_skip_special_tokens_true_cleanup_false'}
            for old,*_ in data:
                if len(old['tokens'])>128:
                    decoded=tokenizer.decode(old['tokens'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    if decoded!=old['text']:raise ValueError('Full original tokenizer roundtrip differs before claim: '+old['key'])
        run,remaining,owner=runner.claim(ns,plan,{'host':'4028','role':'CPU_exact_greedy_output_budget_reuse',
            'physical_gpus':[],'source_raw_list':str(args.old_raw_list),'source_raw_list_sha256':file_hash(args.old_raw_list)})
        if run is None:raise ValueError('Refusing implicit repeat of a reuse claim')
        number=0;chunk=[];pending=run/f'chunk_{number:05d}.pending.jsonl';nlong=0
        for (old,source,line,sha,eos),key in zip(data,keys):
            value=convert_row(old,source,line,sha,eos,plan,key,owner,tokenizer,proof)
            nlong+=int(value['truncated'])
            with pending.open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
            chunk.append(key)
            if len(chunk)==8:
                runner.seal(plan,run,owner,number,pending,chunk,eos);number+=1;chunk=[]
                pending=run/f'chunk_{number:05d}.pending.jsonl'
        if chunk:runner.seal(plan,run,owner,number,pending,chunk,eos)
        runner.write_once(run/'complete.json',{'status':'complete','completed_rows':len(keys),
            'completed_utc':now(),'identity':plan['identity'],'claim_identity':stable_hash(owner),
            'EOS_within_budget':len(keys)-nlong,'recorded_prefix_length128':nlong,'new_generations':0})
        result={'model':model,'reused':len(keys),'EOS_within_budget':len(keys)-nlong,'recorded_prefix_length128':nlong}
        summary['models'].append(result);print(json.dumps(result),flush=True)
    out=ROOT/'outputs/hallusion_blind128_20261006/registration/reuse_receipt.json'
    runner.write_once(out,summary)

if __name__=='__main__':main()
