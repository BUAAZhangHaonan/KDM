"""Synchronize current result descriptions and aggregate tables after the one-time merge."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
P=Path(__file__).resolve().parents[1];S=P/'statistics'
v=pd.read_csv(S/'vizwiz_nine_models_recorded_working_points.csv');assert len(v)==68
macro=pd.read_csv(S/'nine_model_macro_summary.csv');food=macro[macro.task.eq('Food')]
rows=[]
for method,g in v.groupby('method'):
 rows.append(dict(task='VizWiz',method=method,models=len(g),quality_metric='answer_quality_mean',quality_mean=g.answer_quality_mean.mean(),precision_macro_defined_only=g.precision.mean(),precision_defined_models=int(g.precision.notna().sum()),recall_mean=g.recall.mean(),J_mean=g.J.mean()))
pd.concat([food,pd.DataFrame(rows)],ignore_index=True).to_csv(S/'nine_model_macro_summary.csv',index=False)
labels={'direct':'Direct','vcd':'VCD','dola':'DoLa','deco':'DeCo','sid':'SID','cda_visual':'CDA','instruction_vcd':'IP-VCD','instruction_m3id':'IP-M3ID'}
lines=['### B.2 VizWiz 全部记录工作点','', '固定512题；原生 DoLa/DeCo 覆盖九模型，SID 覆盖五个适用模型。各指标取同一工作点。','', '![完整 VizWiz 九模型比较](figures/figA01_vizwiz_all_baselines_J.png)','', '| 模型 | 方法 | Q | P | R | J | TP | FP | 表达 |','|---|---|---:|---:|---:|---:|---:|---:|---|']
def pct(x):return '—' if pd.isna(x) else f'{100*x:.2f}'
for model in dict.fromkeys(v.model):
 for method in labels:
  q=v[v.model.eq(model)&v.method.eq(method)]
  if q.empty:continue
  r=q.iloc[0];lines.append('| '+' | '.join([r.model_name,labels[method]]+[pct(r[k]) for k in ['answer_quality_mean','precision','recall','J']]+[str(int(r.TP)),str(int(r.FP)),str(r.marker)])+' |')
lines+=['','原 SID 在 MiniCPM（视觉词元不足100）与 Qwen3.5（第二层线性注意力）上不适用；Qwen2.5-VL、Qwen3-VL 在本面板各有7题少于固定100视觉词元，整面板记为不适用。','']
ap=P/'appendix_v3_zh.md';s=ap.read_text(encoding='utf8');left,tail=s.split('### B.2 VizWiz 全部记录工作点',1);_,right=tail.split('## C 控制实验与测量单位',1)
s=left+'\n'.join(lines)+'\n## C 控制实验与测量单位'+right
s=s.replace('VizWiz的九模型五方法共45个单一登记工作点。','VizWiz共有68个单一登记工作点，九模型的七种方法与五模型的原SID均使用固定512题。')
ap.write_text(s,encoding='utf8')
ap=P/'appendix_v3_zh.tex';s=ap.read_text(encoding='utf8');anchor=r'\subsection{VizWiz 全部记录工作点}'
figure=r'''
\begin{figure}[htbp]\centering
\includegraphics[width=.90\textwidth]{figures/figA01_vizwiz_all_baselines_J.pdf}
\caption{\textbf{VizWiz 的完整九模型基线比较。} 固定512题，共68个工作点。金色框标出三条件方法，粗体为每模型最高联合效用；n/a为原SID的架构或视觉词元数量限制。}
\label{fig:viz-all}\end{figure}
'''
assert s.count(anchor)==1 and 'fig:viz-all' not in s
s=s.replace(anchor,anchor+figure);ap.write_text(s,encoding='utf8')
sm=S/'SOURCE_MAPPING.md';s=sm.read_text(encoding='utf8')
s=s.replace('all45 rows of `vizwiz_final/main45_metrics.csv`','68 rows: the frozen45 plus23 native baseline rows in `addendum/viz_native_baselines/metrics_actual.csv`')
s+='\n- `vizwiz_native_merge_receipt.json`: accepted23 new native conditions,11776 responses, zero semantic/reference pending; old45 headline and old56 sample scores unchanged. New source_row uses the one-based CSV line including the header.\n- `vizwiz_SID_applicability.csv`: four whole-panel original-SID exclusions. No adapted or reduced-token SID enters these tables.\n'
sm.write_text(s,encoding='utf8')
readme=P/'README.md';s=readme.read_text(encoding='utf8').replace('45个VizWiz工作点','68个VizWiz工作点').replace('五张当前主图','五张当前主图及一张完整VizWiz附录图')
s+='\n本次补入 VizWiz 原生 DoLa、DeCo 各4,608条及原SID 2,560条。23条件均512题，生成、语义评分、官方参考连接已闭合。所有当前表和图直接读取这次接受的汇总。\n'
readme.write_text(s,encoding='utf8')
(P/'change_summary.md').write_text('''# v3 更新说明

- 补齐 VizWiz 原生 DoLa、DeCo 的九模型结果及五个适用模型的原 SID：11,776条、23条件，评分未决0。双任务主表、68点完整VizWiz附录表和全景图已同步。
- 主文保留五张图和两张表。架构图与配对森林图采用单栏；案例、四面板机理和Food全景采用跨栏。图内字体、线宽和配色按实际印刷宽度统一。
- ImageGen用于重绘案例版式与三条件架构。真实案例照片、回复、公式和数值由来源文件组合；数据图由EasyPlot/Python生成。
- Food保留70个工作点；Gemma原SID的2,424题已纳入。SID的架构适用范围在表下注明，探索性SID矩阵不进入主文或附录。
- 附录保留开发选择、有限四路、更新DoLa/SID的同101题比较、CDA五熵测量、重放核查和实际时延。原始完整数据与当前来源索引随配套数据包提供。
- 版本仍为v3。官方VizWiz512题主比较已闭合，全量独立试答的能力交叉验证语义队列继续单列。
''',encoding='utf8')
print({'VizWiz_points':len(v),'macro_rows':len(food)+len(rows),'updated_utc':datetime.now(timezone.utc).isoformat()})
