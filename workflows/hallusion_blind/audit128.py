#!/usr/bin/env python3
"""Validate immutable budget128 Hallusion chunks and write a compact checkpoint."""
from __future__ import annotations
import argparse,json,socket,subprocess,sys
from argparse import Namespace
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import atomic_json,file_hash,stable_hash,read_jsonl
from workflows.hallusion_blind.generate128 import load_plan,validate_rows,now
from workflows.hallusion_blind.score128 import eligible_conditions
BASE=ROOT/'outputs/hallusion_blind128_20261006'
OUTPUT_NAMESPACES=('runs','runs_continuation','runs_recovered_a100','runs_recovered_3090')

def brief_failure(path,root=ROOT):
    x=json.loads(path.read_text())
    return {k:x.get(k) for k in ('failed_utc','key','claim_identity','error','automatic_retry','automatic_parameter_change')}|{'source':str(path.relative_to(root))}

def run_roots(base,namespaces=OUTPUT_NAMESPACES):
    parents=[base]
    collected=base/'collected'
    if collected.exists():
        parents.extend(p for p in collected.iterdir() if p.is_dir())
    return sorted(parent/name for parent in parents for name in namespaces if (parent/name).is_dir())

def plans_from_registration(base,root=ROOT):
    protocol_path=base/'registration/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    conditions=json.loads((protocol_path.parent/protocol['conditions']).read_text())
    eligible,applicability=eligible_conditions(conditions)
    if (protocol.get('output_token_cap')!=128 or protocol.get('termination_policy')!='EOS_or_128_generated_tokens'
        or len(conditions)!=70 or len(eligible)!=68):
        raise ValueError('Expected the authorized budget128 protocol and frozen 70/68 panel')
    plans={}
    for model in protocol['models']:
        args=Namespace(model=model,methods=[c['method'] for c in conditions if c['model']==model],
            protocol=str(protocol_path.relative_to(root)),output=str((base/'runs').relative_to(root)),sample_ids=None)
        plans[model]=load_plan(args,root=root)
    return protocol,eligible,applicability,plans

def verified_chunks(base,plans,eligible,root=ROOT):
    allowed={stable_hash(c) for c in eligible}
    seen={};rawlist=[];counts=defaultdict(Counter);replicas=0
    for folder in run_roots(base):
        for modelbase in sorted(folder.iterdir()):
            if not modelbase.is_dir():continue
            if modelbase.name not in plans:raise ValueError('Foreign model directory in the registered output namespace: '+str(modelbase))
            plan=plans[modelbase.name]
            for receipt_path in sorted(modelbase.glob('claims/*/chunk_*.complete.json')):
                receipt=json.loads(receipt_path.read_text())
                raw=modelbase/receipt['raw_path'];ownerpath=modelbase/receipt['owner_path']
                if (raw.resolve().parent!=receipt_path.resolve().parent
                    or ownerpath.resolve()!=receipt_path.resolve().parent/'owner.json'):
                    raise ValueError('Chunk raw/owner paths leave their actual claim directory')
                owner=json.loads(ownerpath.read_text());sha=file_hash(raw)
                if (receipt.get('status')!='complete' or receipt['identity']!=plan['identity']
                    or sha!=receipt['raw_sha256'] or stable_hash(owner)!=receipt['claim_identity']
                    or file_hash(ownerpath)!=receipt['owner_sha256'] or owner.get('identity')!=plan['identity']
                    or owner.get('source_sha256')!=plan['definition']['source_sha256']):
                    raise ValueError('Immutable budget128 receipt/owner/raw binding differs: '+str(receipt_path))
                owned=owner.get('keys')
                if (not isinstance(owned,list) or len(owned)!=len(set(owned))
                    or not set(owned)<=set(plan['tasks']) or not set(receipt['keys'])<=set(owned)):
                    raise ValueError('Sealed keys lie outside the recorded model/claim ownership')
                checked=validate_rows(raw,plan,receipt['keys'],receipt['claim_identity'],set(receipt['eos_token_ids']))
                if checked!=receipt['validation']:raise ValueError('Receipt validation differs: '+str(receipt_path))
                repeated=[k for k in receipt['keys'] if k in seen]
                if repeated:
                    if len(repeated)!=len(receipt['keys']) or any(seen[k]!=sha for k in repeated):
                        raise ValueError('Non-identical duplicate model/condition/sample output')
                    replicas+=1;continue
                rawlist.append(str(raw))
                for row in read_jsonl(raw):
                    if row['condition_identity'] not in allowed:
                        raise ValueError('Output belongs to an excluded full951 SID panel')
                    seen[row['key']]=sha
                    counter=counts[row['condition_identity']]
                    counter['generated']+=1;counter['truncated']+=int(row['truncated'])
                    counter['natural_eos']+=int(row['terminated'])
                    counter['new']+=int(row['generation_source']=='new_budget128')
                    counter['reused']+=int(row['generation_source']=='reused_budget128')
                    if counter['generated']>951:raise ValueError('Condition exceeds the frozen 951 inputs')
    return rawlist,seen,counts,replicas

def local_workers(base,root=ROOT):
    workers=[];seen=set();boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for folder in run_roots(base):
        for ownerfile in sorted(folder.glob('*/claims/*/owner.json')):
            owner=json.loads(ownerfile.read_text());identity=stable_hash(owner)
            if identity in seen:continue
            seen.add(identity)
            if owner.get('admission',{}).get('host')!='4028':continue
            claim=ownerfile.parent;proc=Path('/proc')/str(owner['pid']);alive=False
            if (proc/'stat').exists():
                stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
                alive=(int(stat[19])==owner['start_tick'] and stat[0]!='Z'
                       and owner.get('boot_id')==boot_id and owner.get('hostname')==socket.gethostname())
            row={'model':ownerfile.parent.parent.parent.name,'claim_id':owner['claim_id'],
                'claim_identity':identity,'pid':owner['pid'],'start_tick':owner['start_tick'],
                'process_running':alive,'physical_gpus':owner['admission']['physical_gpus'],
                'status':'running' if alive else 'exited',
                'sealed_rows':sum(json.loads(f.read_text())['validation']['rows'] for f in claim.glob('chunk_*.complete.json')),
                'first8':[json.loads(f.read_text()) for f in claim.glob('first8_*.json')]}
            for name in ('complete.json','failed.json','released.json'):
                path=claim/name
                if path.exists():
                    row[name[:-5]]=brief_failure(path,root) if name=='failed.json' else json.loads(path.read_text())
                    if name=='failed.json':row['status']='failed'
            workers.append(row)
    for path in sorted((base/'collected').glob('*/status.json')):
        for row in json.loads(path.read_text()).get('workers',[]):
            if 'failed' in row:
                row['failed']={k:row['failed'].get(k) for k in ('failed_utc','key','claim_identity','error')}
            workers.append(row)
    return workers

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    args=p.parse_args();base=args.base.resolve();base.relative_to(ROOT/'outputs/hallusion_blind128_20261006')
    protocol,eligible,applicability,plans=plans_from_registration(base)
    rawlist,seen,counts,replicas=verified_chunks(base,plans,eligible)
    rawpath=base/'validated_raw_list.txt';rawpath.write_text('\n'.join(rawlist)+('\n' if rawlist else ''))
    latest=None
    for path in sorted((base/'scoring').glob('snapshot_*/receipt.json')):
        receipt=json.loads(path.read_text())
        if receipt.get('output_token_budget')!=128:raise ValueError('Foreign scoring snapshot in budget128 directory')
        latest={'path':str(path.relative_to(ROOT)),**receipt}
    condition_rows=[]
    for c in eligible:
        cid=stable_hash(c);counter=counts[cid];n=counter['generated']
        condition_rows.append({'model':c['model'],'method':c['method'],'condition_identity':cid,
            'expected':951,'completed_generation':n,'remaining_generation':951-n,
            'truncated':counter['truncated'],'natural_eos':counter['natural_eos'],
            'new':counter['new'],'reused':counter['reused']})
    state={'updated_utc':now(),'dataset':'hallusionbench','blind_questions':951,
        'hall_development_split':False,'hall_configuration_selection':False,
        'registered_upper_bound':{'conditions':70,'answers':66570},
        'cpu_admitted_upper_bound':{'conditions':68,'answers':64668,'sid_runtime_admission':'per_model_gate'},
        'token_budget':128,'termination':'EOS_or_128_generated_tokens',
        'score_policy':'score_actual_generated_text; length termination is independent of correctness and abstention',
        'excluded_server':'d4030','total_generated_validated':len(seen),
        'new_generation_validated':sum(c['new'] for c in condition_rows),
        'reused_generation_validated':sum(c['reused'] for c in condition_rows),
        'natural_eos_validated':sum(c['natural_eos'] for c in condition_rows),
        'truncated_validated':sum(c['truncated'] for c in condition_rows),
        'remaining_to_cpu_admitted_upper_bound':64668-len(seen),
        'generation_complete_conditions':sum(c['completed_generation']==951 for c in condition_rows),
        'verified_raw_files':len(rawlist),'validated_raw_list_sha256':file_hash(rawpath),
        'exact_transport_replica_chunks_skipped':replicas,'scoring':latest,
        'rows_not_in_latest_scoring_snapshot':len(seen)-(latest['generated'] if latest else 0),
        'conditions':condition_rows,'method_applicability':applicability,'workers':local_workers(base),
        'generation_failures':[brief_failure(path) for folder in run_roots(base) for path in folder.glob('*/claims/*/failed.json')],
        'commits':subprocess.check_output(['git','log','-4','--format=%h'],cwd=ROOT,text=True).splitlines(),
        'git_pushed':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==
                     subprocess.check_output(['git','rev-parse','refs/remotes/origin/master'],cwd=ROOT,text=True).strip()}
    atomic_json(base/'CURRENT_STATE.json',state)
    text='# HallusionBench 951题盲测：128-token 接续\n\n'
    text+=f"更新UTC：{state['updated_utc']}。已验收生成 {len(seen)} / 64668；其中自然EOS {state['natural_eos_validated']}，达到128上限 {state['truncated_validated']}。\n\n"
    text+='全部九模型和方法采用相同128-token预算，保留Food冻结工作点。根据已生成文本分别判正确性和弃权；length不自动记错或弃权。未决标签不计最终指标。旧EOS-only及Food/VizWiz资产保持原样。4030不连接。\n\n'
    if latest:text+=f"评分已决 {latest['resolved']} / {latest['generated']}；完整已决条件 {latest['complete_conditions']} / 68。\n\n"
    text+='| 模型 | 方法 | 已验收生成 | 达到上限 | 剩余 | 最终分母 |\n|---|---|---:|---:|---:|---:|\n'
    text+=''.join(f"| {c['model']} | {c['method']} | {c['completed_generation']} | {c['truncated']} | {c['remaining_generation']} | 951 |\n" for c in condition_rows)
    text+='\nSID：MiniCPM/Qwen3.5沿原架构不适用；Qwen2.5的30题和Qwen3-VL的59题不足100视觉词元，两个完整面板排除，不筛题改分母；其余5面板遵守原SID实际准入。\n'
    (base/'RUN_STATUS.md').write_text(text)
    print(json.dumps({k:v for k,v in state.items() if k not in {'conditions','workers','method_applicability'}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
