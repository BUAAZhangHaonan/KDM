"""Add the accepted Gemma native-SID condition without changing frozen rows."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ST=ROOT/'statistics'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads((args.source/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['passed'] and receipt['primary_complete'] and receipt['rows']==2424
    assert receipt['pending_unique_QA']==0 and not receipt['missing_sample_ids']
    s=pd.read_csv(args.source/'metrics_all.csv').iloc[0]
    assert int(s['n'])==int(s['decided_rows'])==2424
    target=ST/'supplementary/gemma_sid'
    target.mkdir(exist_ok=True,parents=True)
    for name in ['metrics_all.csv','receipt.json','sources.csv','runtime_identities.json']:
        shutil.copy2(args.source/name,target/name)
    food=ST/'food_nine_models_best_observed_J.csv'
    old=pd.read_csv(food)
    old=old.loc[~(old.model.eq('gemma3_4b')&old.method.eq('sid'))].copy()
    assert len(old)==69
    before=old.to_json(orient='records',double_precision=15)
    row={k:s.get(k,None) for k in old.columns}
    row.update({'model_name':'Gemma-3','W':int(s['W_decided']),
        'TN':int(s['n']-s['reference_positive']-s['FP']),
        'condition_id':'gemma3_4b|food101|eval|sid|native_unguided_author_core|NONE|NONE|0|gemma_native_mask_sid_20261004',
        'source_file':'statistics/supplementary/gemma_sid/metrics_all.csv','source_row':2,
        'selection_scope':'single_registered_native_author_operator','candidate_rows':1,'best_J_tie_count':1})
    merged=pd.concat([old,pd.DataFrame([row],columns=old.columns)],ignore_index=True)
    assert not merged.duplicated(['model','method']).any()
    assert (merged.C+merged.W+merged.A==merged.n).all()
    assert ((merged.C+merged.TP)/merged.n-merged.J).abs().max()<1e-12
    assert old.to_json(orient='records',double_precision=15)==before
    merged.to_csv(food,index=False,float_format='%.17g')
    # Refresh only the available-model SID macro row.
    macrofile=ST/'nine_model_macro_summary.csv';macro=pd.read_csv(macrofile)
    sid=merged[merged.method.eq('sid')];idx=macro.index[macro.task.eq('Food')&macro.method.eq('sid')]
    assert len(sid)==7 and len(idx)==1
    for key,value in {'models':len(sid),'quality_mean':sid.accuracy.mean(),
        'precision_macro_defined_only':sid.precision.mean(),'precision_defined_models':sid.precision.notna().sum(),
        'recall_mean':sid.recall.mean(),'J_mean':sid.J.mean()}.items():macro.loc[idx,key]=value
    macro.to_csv(macrofile,index=False,float_format='%.17g')
    out={'source_receipt_sha256':hashlib.sha256((args.source/'receipt.json').read_bytes()).hexdigest(),
        'frozen_rows_preserved':69,'added_rows':1,'total_main_working_points':70,'sid_models':7,
        'complete_food_conditions':484,'complete_food_rows':1173216,
        'gemma_counts':{k:int(s[k]) for k in ['n','C','W_decided','A','TP','FP','reference_positive']}}
    (target/'paper_merge_receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
