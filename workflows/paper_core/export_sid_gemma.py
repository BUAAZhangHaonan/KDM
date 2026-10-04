"""Verify completed Gemma SID sources and export a compact auditable addendum."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,shutil
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,required=True,help='Accepted Gemma SID production/scoring directory')
b=parser.parse_args().run_dir.resolve();s=b/'current_scores';e=b/'final_export'
e.mkdir(exist_ok=True)
receipt=json.loads((s/'receipt.json').read_text())
assert receipt['passed'] and receipt['primary_complete'] and receipt['rows']==2424
assert receipt['pending_unique_QA']==0 and receipt['missing_sample_ids']==[]
scored=pd.read_parquet(s/'score_rows.parquet')
assert len(scored)==scored.sample_id.nunique()==2424
assert set(scored.target_class.value_counts())=={24} and scored.target_class.nunique()==101
assert not scored[['correct_canonical','correct_literal','abstain','uniform_reference']].isna().any().any()
sources=[];raw_ids=[];runtime=[];model_times=0;terminated=Counter();commits=set()
for source in receipt['sources']:
    p=Path(source['path']);part=p.parent
    assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
    ident=json.loads((part/'identity.json').read_text())
    end=json.loads((part/'complete.json').read_text())
    claim=json.loads((part/'claim.json').read_text()) if (part/'claim.json').exists() else end['source_claim']
    admission=ident['runtime_admission'];fixed=admission['fixed_identity']
    if 'execution' in admission:
        execution=admission['execution']
    else:
        execution={**admission,'gpu_uuids':{k:v['uuid'] for k,v in admission['gpu_observation_before_loading'].items()}}
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows)==source['n']==end['completed']
    for row in rows:
        assert all(math.isfinite(x) for x in row['selected_log_probabilities'])
        assert 0<len(row['tokens'])<=32 and len(row['tokens'])==len(row['selected_log_probabilities'])
        assert row['config']==ident['config']
        assert row['sid_adapter_sha256']==ident['sid_adapter_sha256']
        raw_ids.append(row['sample']['id']);model_times+=row['generation_wall_s']
        terminated[str(row['terminated'])]+=1
    source_item={**source,'part':part.name,'host':execution['host'],
        'physical_gpus':','.join(execution['physical_gpus']),
        'gpu_uuids':json.dumps(execution['gpu_uuids'],sort_keys=True),
        'started_utc':claim['started_utc'],'completed_utc':end.get('completed_utc','sealed at next-input boundary'),
        'production_seconds':sum(row['generation_wall_s'] for row in rows),
        'pid':claim['pid'],'starttick':claim['starttick'],
        'environment_python':fixed['environment']['environment_python'],
        'model_path':fixed['checkpoint']['model_path'],
        'model_config_sha256':fixed['checkpoint']['model_config_sha256'],
        'sid_adapter_sha256':ident['sid_adapter_sha256'],
        'official_sid_commit':ident['official_sid_commit']}
    sources.append(source_item)
    runtime.append({**source_item,'environment':fixed['environment'],'checkpoint':fixed['checkpoint'],
                    'config':ident['config'],'sid_fixed':ident['sid_fixed']})
assert len(raw_ids)==len(set(raw_ids))==2424 and set(raw_ids)==set(scored.sample_id)
with (e/'sources.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(sources)
(e/'runtime_identities.json').write_text(json.dumps(runtime,indent=2)+'\n')
for filename in ('score_rows.parquet','metrics_all.csv','receipt.json'):
    shutil.copyfile(s/filename,e/filename)
for filename in ('root_full_review_sid16.json','root_targeted_review93.json','qwen35_architecture.json'):
    shutil.copyfile(b/filename,e/filename)
for filename in ('luna_actual_review126.jsonl','luna_actual_review126_corrections.jsonl',
                 'root_after_luna126_corrections.jsonl','root_luna126_disagreements.csv',
                 'root_luna126_receipt.json'):
    shutil.copyfile(b/filename,e/filename)
root=b.parents[2]
shutil.copyfile(root/'docs/supplemental/sid_architecture_scope_20261004.md',e/'sid_architecture_scope.md')
arguments=['venv/bin/python workflows/paper_core/score_sid_gemma.py']
for source in receipt['sources']:
    arguments.extend(['--part',str(Path(source['path']).parent.relative_to(root))])
for decision in receipt['extra_decisions']:
    arguments.extend(['--decision-file',str(Path(decision['path']).relative_to(root))])
arguments.extend(['--output',str(s.relative_to(root))])
(e/'REPRODUCE.md').write_text('# 复算入口\n\n中央仓库工作目录：`'+str(root)+'`。先读取冻结评分/参考与已封存 raw。下列命令只评分与合并，不重新生成：\n\n```bash\n'+' '.join(arguments)+'\n```\n\n紧凑导出脚本：`'+str(b/'finalize_gemma_export.py')+'`。参数及运行依赖见 `runtime_identities.json`，每条回答与 token IDs 在 `score_rows.parquet`。\n')
columns=['model','dataset','split','method','kind','marker','reference_marker','guided','reference_guided',
  'replicate','implementation_revision','sample_id','target_class','question','answer','qa_key',
  'correct_canonical','correct_literal','abstain','uniform_reference','score_reason','source_path','source_line',
  'source_sha256','source_identity_sha256','seed','terminated','sid_adapter_sha256']
scored[columns].to_csv(e/'per_sample.csv',index=False)
annotations=[]
for decision in receipt['extra_decisions']:
    for line,row in enumerate(map(json.loads,Path(decision['path']).read_text().splitlines()),1):
        annotations.append({k:row.get(k,'') for k in ('qa_key','question','answer','abstain','primary_span','evidence_span','multiple_primary','reason','author','annotation_model','annotation_effort','annotation_call_id')}|
                           {'source_path':decision['path'],'source_line':line,'source_sha256':decision['sha256']})
pd.DataFrame(annotations).to_csv(e/'content_judgments.csv',index=False)
metrics=pd.read_csv(e/'metrics_all.csv').iloc[0].to_dict()
state={'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':'Gemma3-4B original SID, Food eval',
 'status':'complete','expected':2424,'generated':2424,'scored':2424,'remaining':0,
 'pending_unique_QA':0,'duplicate_sample_keys':0,'missing_sample_keys':0,'classes':101,'per_class':24,
 'frozen_reference_positive':int(scored.uniform_reference.sum()),'terminations':dict(terminated),
 'summed_generation_seconds':model_times,'gpu_producers_running':0,'gpu_release_check_utc':'2026-10-04T07:51:00+00:00',
 'generation_parts':[{'part':r['part'],'host':r['host'],'gpu':r['physical_gpus'],'n':r['n'],'completed_utc':r['completed_utc']} for r in sources],
 'semantic_decision_unique_QA':len({r['qa_key'] for r in annotations}),
 'annotation_provenance':'142 unique QA: first 16 fully root-reviewed; remaining 126 independently reviewed by actual gpt-5.6-luna medium, then root resolved 20 primary-state disagreements and appended two corrections; original judgments preserved',
 'qwen35_status':'original SID not applicable: second decoder block is GatedDeltaNet; no variant implemented',
 'k100_status':'registered cu124 runtime failed before input; zero outputs; released; no runtime fallback',
 'next_action':'integrate Gemma row and architecture footnotes in unified paper/data package'}
(b/'CURRENT_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
shutil.copyfile(b/'CURRENT_STATE.json',e/'CURRENT_STATE.json')
readme=f'''# Gemma3-4B 原生 SID 补齐结果

Food eval 共 2,424 条，101 类每类 24 张。生成、canonical 评分和冻结参考连接完成，缺失、重复和未决均为 0。

| C | W | A | TP | FP | FN | 参考阳性 | Acc (%) | J (%) | P | R (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| {int(metrics['C'])} | {int(metrics['W_decided'])} | {int(metrics['A'])} | {int(metrics['TP'])} | {int(metrics['FP'])} | {int(metrics['FN'])} | {int(metrics['reference_positive'])} | {metrics['accuracy']*100:.5f} | {metrics['J']*100:.5f} | 未定义：弃权数为 0 | {metrics['recall']*100:.5f} |

采用作者形式的原生 SID：第二解码块 attention、rank=100、alpha=0.5、完整词表贪心、BF16、batch=1、max_tokens=32。Gemma 原生滑动/全局与图像块 mask 在参考路径内保留，仅屏蔽未选视觉键列。8 输入真实全路径准入计入上述分母；后续来源互斥。

`metrics_all.csv` 为完整条件行；`per_sample.csv` 与 `score_rows.parquet` 提供逐样本原文、评分、参考和精确来源。`sources.csv` 指向服务器原始预测及真实 PID/starttick/运行时间。`runtime_identities.json` 保留版本、权重登记、参数与物理 GPU。`content_judgments.csv` 保存 142 个新唯一完整 QA 的原始判断及追加裁定。前 16 项由 root 全量复核；其余 126 项由实际 GPT-5.6 Luna medium 独立复核，再由 root 裁定 20 项主答案提取分歧并追加 2 项修正。Luna 原始文件、3 项弃权判断纠正及 root 裁定均保留；未知调用 ID 留空。

旧六模型 SID 全量与作者算子 101 题核对保持独立身份，本目录不替换这些旧结果。Qwen3.5 第二块为递归线性注意力，原 SID 不适用；没有改聚合层或构造算法变体。

GPU 生产已完成，后续 GPU 时间为 0。有限概率、完整键、类配额、固定 seed/config、身份与分母均核对通过。
'''
(e/'README.md').write_text(readme)
(b/'RUN_STATUS.md').write_text(readme)
manifest=[{'file':str(p.relative_to(e)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(e.iterdir()) if p.is_file() and p.name!='file_manifest.csv']
pd.DataFrame(manifest).to_csv(e/'file_manifest.csv',index=False)
print(json.dumps({'rows':2424,'pending':0,'export_bytes':sum(x['bytes'] for x in manifest),'summed_generation_seconds':model_times,'state_path':str(b/'CURRENT_STATE.json')}))
