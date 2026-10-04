"""Merge accepted new native baselines without changing frozen scientific rows."""
from pathlib import Path
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];ST=ROOT/'statistics'
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args()
src=a.source;data=a.data_dir
receipt=json.loads((src/'receipt.json').read_text(encoding='utf8'))
assert receipt['passed'] and receipt['rows']==11776 and receipt['conditions']==23
assert receipt['complete_conditions']==23 and receipt['pending_QA']==0
new=pd.read_csv(src/'metrics_actual.csv')
assert len(new)==23 and new.n.eq(512).all() and new.primary_complete.eq(True).all()
assert new.quality_pending.eq(0).all() and new.abstain_pending.eq(0).all()
assert new.groupby('method').size().to_dict()=={'deco':9,'dola':9,'sid':5}
assert new.reference_positive.eq(166).all() and new.official_answerable_n.eq(346).all()
assert ((new.FULL+new.PARTIAL+new.ZERO+new.A)==512).all()
assert (new.TP+new.FP).eq(new.A).all() and (new.TP+new.FN).eq(166).all()
assert ((new.J-(new.answer_quality_sum+new.TP)/512).abs()<1e-12).all()
old_path=ST/'vizwiz_nine_models_recorded_working_points.csv'
old=pd.read_csv(old_path);assert len(old)==45,'Run once on the frozen 45-row source.'
assert not old.method.isin(['dola','deco','sid']).any()
names=old.set_index('model').model_name.to_dict()
source_rel='addendum/viz_native_baselines/metrics_actual.csv'
add=new.copy();add['model_name']=add.model.map(names)
add['source_file']=source_rel;add['source_row']=range(2,len(add)+2)
add['selection_scope']='single_registered_native_unguided_VizWiz_working_point';add['candidate_rows']=1
merged=pd.concat([old,add],ignore_index=True,sort=False)
assert len(merged)==68 and not merged.duplicated(['model','method']).any()
pd.testing.assert_frame_equal(merged.iloc[:45][old.columns].reset_index(drop=True),old,check_dtype=False)
merged.to_csv(old_path,index=False)
representative=['qwen25vl','llava16_mistral','minicpm26','internvl35_8b','qwen3vl']
five=merged[merged.model.isin(representative)];assert len(five)==37
five.to_csv(ST/'vizwiz_five_representative_models.csv',index=False)
dest=data/'addendum/viz_native_baselines';dest.mkdir(parents=True,exist_ok=False)
for name in ['metrics_actual.csv','score_rows.jsonl.gz','receipt.json','official_fixed512_references.csv','source_index.csv','annotation_review_receipt.json']:
 if (src/name).exists():shutil.copy2(src/name,dest/name)
assert (dest/'score_rows.jsonl.gz').exists()
vf=data/'vizwiz_final'
oldall=pd.read_csv(vf/'all56_metrics.csv');assert len(oldall)==56
assert not oldall.method.isin(['dola','deco','sid']).any()
all79=pd.concat([oldall,new],ignore_index=True,sort=False)
all79['J']=(all79.answer_quality_sum+all79.TP)/all79.n
keys=['model','dataset','split','method','kind','marker','reference_marker','guided','reference_guided','replicate']
assert len(all79)==79 and not all79.duplicated(keys).any() and all79.primary_complete.eq(True).all()
all79.to_csv(vf/'all79_metrics.csv',index=False);merged.to_csv(vf/'main68_metrics.csv',index=False)
coverage=all79[keys+['n','quality_pending','abstain_pending','reference_positive']]
coverage.to_csv(vf/'condition_coverage79.csv',index=False)
# Replacement tables are verified before removing only their redundant copied predecessors.
for name in ['all56_metrics.csv','main45_metrics.csv','condition_coverage56.csv','new25_metrics.csv']:
 f=vf/name
 if f.exists():f.unlink()
status=pd.read_csv(data/'DATA_STATUS.csv')
for i,r in status[status.stage.eq('VizWiz_official_eval512_all')].iterrows():
 n=int(all79.model.eq(r.model).sum());status.loc[i,['condition_count','completed_rows','expected_rows','semantics_pending_rows','reference_pending_rows','source']]=[n,n*512,n*512,0,0,'vizwiz_final/all79_metrics.csv']
status.loc[status.stage.eq('VizWiz_official_eval512_main'),['condition_count','completed_rows','expected_rows','source','scope']]=[68,34816,34816,'vizwiz_final/main68_metrics.csv','nine models seven primary methods plus five applicable original SID; 166 unanswerable and 346 answerable per condition']
status.loc[status.stage.eq('VizWiz_official_eval512_mechanism'),'source']='vizwiz_final/all79_metrics.csv'
status.to_csv(data/'DATA_STATUS.csv',index=False)
na=pd.DataFrame([{'model':m,'dataset':'vizwiz','method':'sid','status':'not_applicable','reason':w} for m,w in {
 'qwen25vl':'7/512 inputs have fewer than the original fixed 100 visual tokens',
 'qwen3vl':'7/512 inputs have fewer than the original fixed 100 visual tokens',
 'qwen35_4b':'second block uses linear attention; original SID softmax-attention weights unavailable',
 'minicpm26':'visual-token count is below the original fixed 100; no architecture adaptation'}.items()])
na.to_csv(ST/'vizwiz_SID_applicability.csv',index=False);na.to_csv(dest/'SID_applicability.csv',index=False)
ap=ROOT/'appendix_v3_zh.tex';text=ap.read_text(encoding='utf8')
text=text.replace('VizWiz 的九模型、五方法主比较共有45行，每个模型的各方法使用一个登记工作点。','VizWiz 主比较共68个工作点：九模型的 Direct、VCD、DoLa、DeCo、CDA 与两个 IP 实例，以及五个适用模型的原 SID；每个工作点512题。')
sid='VizWiz 的 SID 使用原作者算子，覆盖 Gemma、InternVL、LLaVA-Mistral、OneVision 与 Phi；Qwen2.5-VL 和 Qwen3-VL 各有7/512题的视觉词元少于固定100个，因而整面板记为不适用。DoLa、DeCo 在九模型上使用本轮核对后的原生无引导实现。\n\n'
marker='\\subsection{模型与解码参数}'
assert marker in text;text=text.replace(marker,sid+marker,1)
section='\\subsection{VizWiz 全部记录工作点}';head,tail=text.split(section,1)
start=tail.index('\\begingroup');end=tail.index('\\end{longtable}\\endgroup')+len('\\end{longtable}\\endgroup')
labels={'direct':'Direct','vcd':'VCD','dola':'DoLa','deco':'DeCo','sid':'SID','cda_visual':'CDA','instruction_vcd':'IP-VCD','instruction_m3id':'IP-M3ID'}
order=list(dict.fromkeys(old.model));methods=list(labels)
lines=[r'\begingroup\fontsize{8.2}{10.5}\selectfont\setlength{\tabcolsep}{3.7pt}',r'\begin{longtable}{llrrrrrrl}',r'\toprule',r'模型 & 方法 & Q & P & R & J & TP & FP & 表达\\',r'\midrule\endfirsthead',r'\toprule',r'模型 & 方法 & Q & P & R & J & TP & FP & 表达\\',r'\midrule\endhead',r'\bottomrule\endfoot']
def pct(x):return '--' if pd.isna(x) else f'{100*x:.2f}'
for model in order:
 for method in methods:
  rows=merged[merged.model.eq(model)&merged.method.eq(method)]
  if rows.empty:continue
  r=rows.iloc[0]
  lines.append(' & '.join([r.model_name,labels[method]]+[pct(r[k]) for k in ['answer_quality_mean','precision','recall','J']]+[str(int(r.TP)),str(int(r.FP)),str(r.marker)])+r'\\')
lines+=[r'\end{longtable}\endgroup']
text=head+section+tail[:start]+'\n'.join(lines)+tail[end:];ap.write_text(text,encoding='utf8')
readme=data/'README.md';s=readme.read_text(encoding='utf8')
s=s.replace('vizwiz_final/main45_metrics.csv','vizwiz_final/main68_metrics.csv').replace('vizwiz_final/all56_metrics.csv','vizwiz_final/all79_metrics.csv')
s=s.replace('56 个已决条件、28,672 行：45 个主比较条件及 11 个机制条件','79 个已决条件、40,448 行：68 个主比较条件及 11 个机制条件')
s=s.replace('合计保留全部 56 条件逐样本评分','保留原56条件逐样本评分；`addendum/viz_native_baselines/score_rows.jsonl.gz` 补入23条件、11,776行原生 DoLa/DeCo/SID 逐样本评分')
s=s.replace('本轮数据包组装没有 GPU 推理、训练、新语义判断或评分变更。','本次补入真实原生 DoLa/DeCo/SID 生成与语义评分；原56条件评分保持原值。')
s+='\n新增 VizWiz 原生基线：DoLa 和 DeCo 各9×512；SID为5×512，4个模型的原法适用性原因见 `addendum/viz_native_baselines/SID_applicability.csv`。\n'
readme.write_text(s,encoding='utf8')
audit={'created_utc':datetime.now(timezone.utc).isoformat(),'new_rows':11776,'new_conditions':23,'new_pending':0,'VizWiz_all_conditions':79,'VizWiz_all_rows':40448,'VizWiz_main_points':68,'frozen45_unchanged':True,'frozen56_scores_changed':False,'Food_scores_changed':False,'source_receipt_sha256':hashlib.sha256((src/'receipt.json').read_bytes()).hexdigest()}
(ST/'vizwiz_native_merge_receipt.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
(dest/'merge_receipt.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit))
