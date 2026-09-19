"""Join independent probes to paired behavior, keeping every difficulty group."""
from collections import defaultdict
import numpy as np
from .analysis import condition
from .statistics import mean_ci,ratio_ci


def behavioral_validity(rows,probes,closed,bootstrap=2000):
    index={(p['model'],p['sample_id']):p for p in probes}
    ranks={(r['model'],r['sample']['id']):r['gold_rank'] for r in closed}
    groups=defaultdict(list)
    for row in rows:
        if row['method']!='direct' or not row['guided'] or row['kind']!='main':continue
        key=(row['model'],row['sample']['id'])
        if key not in index:raise ValueError('Missing independent probe')
        probe=index[key];rank=ranks.get(key)
        warranted=(probe['n_full_correct']==0 and rank is not None and rank>1)
        # Human answerability is kept as a separate evidence source.
        groups[(row['model'],row['sample']['dataset'],row['marker'])].append({
            'sample_id':row['sample']['id'],'cluster':row['sample']['cluster'],
            'abstained':row['label']=='abstain','probe':probe,'closed_rank':rank,
            'behavioral_warrant':warranted,'annotated_answerable':row['sample'].get('annotated_answerable')})
    result=[]
    for key,values in groups.items():
        for abstained in (False,True):
            part=[x for x in values if x['abstained']==abstained]
            valid=[x for x in part if x['probe']['mean_correctness'] is not None]
            result.append({'model':key[0],'dataset':key[1],'marker':key[2],
                'baseline_abstained':abstained,'n':len(part),
                'independent_correctness':mean_ci([x['probe']['mean_correctness'] for x in valid],
                    [x['cluster'] for x in valid],bootstrap) if valid else None,
                'n_zero_correct':sum(x['probe']['n_full_correct']==0 for x in valid),
                'n_intermediate':sum(1<=x['probe']['n_full_correct']<=7 for x in valid),
                'n_stable_correct':sum(x['probe']['n_full_correct']>=8 for x in valid),
                'n_behaviorally_supported_abstentions':sum(x['behavioral_warrant'] for x in part),
                'n_annotated_unanswerable':sum(x['annotated_answerable']==0 for x in part)})
    return result


def method_comparison(rows,probes,closed,bootstrap=2000):
    groups=defaultdict(dict)
    for r in rows:
        if r['sample']['split']=='eval' and r['kind']!='independent_attempt':groups[condition(r)][r['sample']['id']]=r
    pindex={(p['model'],p['sample_id']):p for p in probes}
    rankindex={(r['model'],r['sample']['id']):r['gold_rank'] for r in closed}
    out=[]
    for key,new in groups.items():
        model,dataset,marker,method,refmarker,refguide,kind=key
        if method not in {'instruction_vcd','instruction_m3id'}:continue
        reference=method.removeprefix('instruction_')
        oldkey=(model,dataset,marker,reference,marker,True,'main')
        directkey=(model,dataset,marker,'direct',marker,True,'main')
        if oldkey not in groups or directkey not in groups:raise ValueError('Missing comparison baseline')
        old,direct=groups[oldkey],groups[directkey]
        if set(new)!=set(old) or set(new)!=set(direct):raise ValueError('Unpaired method comparison')
        ids=sorted(new);clusters=[new[x]['sample']['cluster'] for x in ids]
        a=np.array([new[x]['correct'] for x in ids]);o=np.array([old[x]['correct'] for x in ids]);d=np.array([direct[x]['correct'] for x in ids])
        abstain_base=np.array([direct[x]['label']=='abstain' for x in ids])
        abstain_new=np.array([new[x]['label']=='abstain' for x in ids]);abstain_old=np.array([old[x]['label']=='abstain' for x in ids])
        rescue=(o==1)&(d<1);copy_correct=np.where(abstain_base,d,o)
        row={'model':model,'dataset':dataset,'marker':marker,'method':method,'n':len(ids),
             'accuracy_vs_original':mean_ci(a-o,clusters,bootstrap),
             'accuracy_vs_copy_baseline':mean_ci(a-copy_correct,clusters,bootstrap),
             'corrected_answer_retention':ratio_ci((rescue&(a==1)).astype(float),rescue.astype(float),clusters,bootstrap),
             'copy_corrected_answer_retention':ratio_ci((rescue&(copy_correct==1)).astype(float),rescue.astype(float),clusters,bootstrap),
             'n_new_abstentions_on_originally_correct':int(((d==1)&abstain_new).sum())}
        for source in ('behavioral','human'):
            supported=[]
            for sid in ids:
                pi=pindex.get((model,sid));rank=rankindex.get((model,sid))
                if source=='behavioral':value=bool(pi and pi['n_full_correct']==0 and rank is not None and rank>1)
                else:value=direct[sid]['sample'].get('annotated_answerable')==0
                supported.append(value)
            den=abstain_base&np.array(supported)
            row[source+'_n_supported_original_abstentions']=int(den.sum())
            row[source+'_retention_new']=ratio_ci((den&abstain_new).astype(float),den.astype(float),clusters,bootstrap)
            row[source+'_retention_original']=ratio_ci((den&abstain_old).astype(float),den.astype(float),clusters,bootstrap)
            idx=np.flatnonzero(den)
            row[source+'_retention_change']=mean_ci((abstain_new[idx].astype(float)-abstain_old[idx].astype(float)),
                np.array(clusters)[idx],bootstrap) if len(idx) else None
        out.append(row)
    return out
