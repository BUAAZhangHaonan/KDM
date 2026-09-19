import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--status',action='append',default=[]);a=p.parse_args()
registry=ROOT/'outputs/verification/sid_capability_status.json'
status=json.loads(registry.read_text()) if registry.exists() else {}
for entry in a.status:
 key,value=entry.split('=',1);status[key]=value
registry.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
text='''# SID 固定候选能力核验

范围为官方 commit `127dd412fa6b61ab1c9babf6979ec4da98002438` 的 agg_layer=2/rank100/mask-mode 参考构造。native16通过不等于SID通过。表内SID passed是固定图像上的软件接口证据，不代表整个dataset可计算；完整9,167图的rank100前置约束见 `docs/current/FULL_VISUAL_COUNT_REVIEW.md`，Qwen2.5/Qwen3/GLM的VizWiz与Mini两个dataset存在不足100输入。passed只表示固定官方选择与掩码核心在同一native backbone前向的数值比对通过；oracle共享hook transport，不是完整旧官方fork复现，不证明研究收益。

单卡任务使用每个spec自己的environment_python、物理GPU4或5单卡、双卡使用物理4,5、`bash scripts/worker.sh`锁与项目内缓存。不改输入、生成配置或权重映射，不在模型失败后重试/降规模。LLaVA1.5-7B沿用主线程已跑证据、不重复。Gemma12B和LLaVA13B原为两卡调度待办，未执行不能算完成。无权重preflight有实际config和源码行证据，区分固定参考所需结构不成立与当前适配映射缺口。

|候选|状态|实际数值/具体原因|证据|
|---|---|---|---|
'''
for path in sorted((ROOT/'configs/runtime').glob('*.json')):
 spec=json.loads(path.read_text())
 if not isinstance(spec,dict) or 'kwargs' not in spec or 'gpu_count' not in spec:continue
 key=spec['key'];proof=ROOT/'outputs/verification'/f'{key}_sid_reference.json'
 for version in range(2,10):
  newer=proof.with_name(f'{key}_sid_reference_v{version}.json')
  if newer.exists():proof=newer
 state=status.get(key,'two_gpu_scheduling_pending' if spec['gpu_count']==2 else 'pending')
 detail='尚未执行';link='—'
 if proof.exists():
  row=json.loads(proof.read_text())
  receipt=proof.with_name(proof.stem+'_summary.json')
  link=f'`{receipt.relative_to(ROOT) if receipt.exists() else proof.relative_to(ROOT)}`; 完整raw `{proof.relative_to(ROOT)}` / 同名 `.log`'
  if row.get('passed'):
   state='passed' if row.get('runtime_adapter_sha256',{}).get('sid.py')==hashlib.sha256((ROOT/'src/kdm/models/sid.py').read_bytes()).hexdigest() else 'prior_source_passed_recheck_pending'
   ev=row.get('evidence',[])
   detail=f"{len(ev)} visits; max logit error={max((x['max_abs_logit_error'] for x in ev),default=0)}; fresh error={max((x['fresh_reference_max_abs_logit_error'] for x in ev),default=0)}"
  else:
   state=row.get('classification',state if state!='pending' else 'runtime_error_pending_diagnosis')
   detail=row.get('error','无成功证据').replace('|','/')
 text+=f'|{key}|{state}|{detail}|{link}|\n'
text+='''
映射调查详见 `docs/current/SID_VISUAL_MAPPING_REVIEW.md` 和 `outputs/verification/{minicpm26,minicpm45,phi35}_sid_visual_mapping.json`。Mini两版均有64<100的原生输入，属于固定rank在实际样本未定义；Phi原生757连续位置映射已在v2实现并真实前向验收通过，native/clean前后logits零差。InternVL双卡为另外获准的新factory，旧单卡OOM记录保留。

分类解释：`fixed_reference_structure_incompatible` 表示实际结构不具备固定第二层attention定义；`adapter_structural_mapping_gap` 表示当前适配器未实现实际结构/视觉区间映射，不能推断该模型本质上不能定义SID；`oom` 是资源失败；`runtime_error` 是软件/运行错误；`numerical_mismatch` 是实际对照未通过。后三类不能写成架构不支持。

启动异常保留：LLaVA-v1.6-Vicuna第一次只在shell层因worker.sh无执行位失败（模型未加载），日志为 `outputs/verification/llava16_vicuna_sid_reference.launch_error.log`；随后用bash运行同一脚本，未改变权限或模型参数。
'''
(ROOT/'docs/current/SID_CAPABILITY_REVIEW.md').write_text(text)
print('updated SID_CAPABILITY_REVIEW.md')
