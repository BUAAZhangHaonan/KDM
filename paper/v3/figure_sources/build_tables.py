"""Generate the two main manuscript tables from identified result rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / 'statistics'
OUT = ROOT / 'tables'
OUT.mkdir(exist_ok=True)
FOOD = pd.read_csv(ST / 'food_five_representative_models.csv')
VIZ = pd.read_csv(ST / 'vizwiz_five_representative_models.csv')
assert len(VIZ)==37 and VIZ.n.eq(512).all(), 'Requires completed native VizWiz baseline merge.'
MODELS = ['qwen25vl', 'llava16_mistral', 'minicpm26', 'internvl35_8b', 'qwen3vl']
METHODS = ['direct','vcd','dola','deco','sid','cda_visual','instruction_vcd','instruction_m3id']
LABELS = ['Direct','VCD','DoLa','DeCo','SID','CDA','IP-VCD','IP-M3ID']

def rate(value, bold=False):
    if pd.isna(value):
        return '--'
    s = f'{100 * float(value):.2f}'
    return r'\textbf{' + s + '}' if bold else s

lines = [r'\begin{table*}[t]\centering',
 r'\caption{\textbf{Food-101 与 VizWiz 的联合决策比较。} 五个代表模型；指标单位为百分比。Food 每行 $N=2,424$，VizWiz 每行 $N=512$。IP 的 Food 指标取同一观测最高 $J$ 配置，VizWiz 为单一工作点。粗体为各模型、各任务的最高 $J$。}',
 r'\label{tab:main}',r'\fontsize{7.4}{9.0}\selectfont\setlength{\tabcolsep}{4.0pt}',
 r'\begin{tabular}{llrrrr@{\hspace{11pt}}rrrr}\toprule',
 r'模型 & 方法 & \multicolumn{4}{c}{Food-101} & \multicolumn{4}{c}{VizWiz}\\',
 r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
 r' & & Acc & P & R & $J$ & Q & P & R & $J$\\\midrule']
for model in MODELS:
    fm = FOOD[FOOD.model.eq(model)].set_index('method')
    vm = VIZ[VIZ.model.eq(model)].set_index('method')
    model_name = fm.iloc[0].model_name
    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        left = model_name if i == 0 else ''
        fr = ['n/a'] * 4 if method not in fm.index else [rate(fm.loc[method, c], c == 'J' and np.isclose(fm.loc[method, c], fm.J.max())) for c in ['accuracy','precision','recall','J']]
        vr = ['n/a'] * 4 if method not in vm.index else [rate(vm.loc[method, c], c == 'J' and np.isclose(vm.loc[method, c], vm.J.max())) for c in ['answer_quality_mean','precision','recall','J']]
        lines.append(' & '.join([left,label]+fr+vr)+r'\\')
    if model != MODELS[-1]:
        lines.append(r'\addlinespace[3pt]')
lines += [r'\bottomrule\end{tabular}',
 r'\par\vspace{3pt}\begin{minipage}{.98\textwidth}\fontsize{7.2}{9}\selectfont',
 r'Acc：全部输入正确率；Q：答案共识信用；P/R：弃权精确率/召回率。--：零弃权时精确率未定义；n/a：原 SID 不适用。MiniCPM 的视觉词元不足100，Qwen3.5 第二层为线性注意力；VizWiz 中 Qwen2.5-VL、Qwen3-VL 各有7题少于固定100视觉词元。CDA 为视觉迁移。',
 r'\end{minipage}\end{table*}']
(OUT / 'main_joint.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

rt = pd.read_csv(ST / 'runtime_matched_full_cohorts.csv')
cost_models = ['qwen25vl','minicpm26']
cl = [r'\begin{table*}[t]\centering',
 r'\caption{\textbf{相同指令下的结构作用与推理开销。} 左：九模型、四表达的36个配对，$\Delta J$ 单位为百分点。右：同硬件、同表达的匹配运行，每方法2,424题；VCD 为同措辞引导条件，时间包含输入处理、会话构造与生成。}',
 r'\label{tab:ablation-cost}',
 r'\begin{minipage}[t]{.51\textwidth}\centering\fontsize{8}{10}\selectfont\setlength{\tabcolsep}{3pt}',
 r'\begin{tabular}{llrr}\toprule',r'实例 & 对照 & 平均 $\Delta J$ & 范围\\\midrule',
 r'IP-VCD & 双侧引导 & +0.48 & [$-$2.72, +2.52]\\',
 r'IP-VCD & 参考去引导 & +3.19 & [$-$8.87, +25.78]\\',
 r'IP-M3ID & 双侧引导 & $-$0.06 & [$-$4.74, +3.63]\\',
 r'IP-M3ID & 参考去引导 & $-$1.33 & [$-$7.55, +4.87]\\\bottomrule',
 r'\end{tabular}\end{minipage}\hfill',
 r'\begin{minipage}[t]{.47\textwidth}\centering\fontsize{8}{10}\selectfont\setlength{\tabcolsep}{3pt}',
 r'\begin{tabular}{llrrr}\toprule',r'模型 & 表达 & VCD & IP-VCD & CDA\\',
 r' & & \multicolumn{3}{c}{秒/题（相对 VCD）}\\\midrule']
for model in cost_models:
    q = rt[(rt.model == model) & (rt.marker == 'UNKNOWN')].set_index('method')
    assert len(q)==3 and (q.N==2424).all()
    names={'qwen25vl':'Qwen2.5-VL','minicpm26':'MiniCPM-V2.6'}
    vals=[f'{q.loc[m,"wall_s_mean"]:.3f}' for m in ['vcd','instruction_vcd','cda_visual']]
    ratios=[f'({q.loc[m,"ratio_to_guided_vcd"]:.2f}$\\times$)' for m in ['vcd','instruction_vcd','cda_visual']]
    cl.append(' & '.join([names[model],'UNKNOWN']+vals)+r'\\')
    cl.append(' & & '+' & '.join(ratios)+r'\\')
cl += [r'\bottomrule\end{tabular}\end{minipage}',r'\end{table*}']
(OUT/'ablation_cost.tex').write_text('\n'.join(cl)+'\n',encoding='utf-8')
print(json.dumps({'main_table_models':len(MODELS),'Food_rows':len(FOOD),'Viz_rows':len(VIZ),'cost_models':cost_models}))
