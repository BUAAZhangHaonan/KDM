"""Join independent probes to paired behavior, keeping every difficulty group."""
from collections import defaultdict
import numpy as np
from .analysis import condition
from .statistics import mean_ci,ratio_ci


def evidence_indices(probes,closed):
    index={};ranks={}
    for p in probes:
        key=(p['model'],p['sample_id'])
        if key in index:raise ValueError('Duplicate independent probe summary')
        index[key]=p
    for r in closed:
        key=(r['model'],r['sample']['id'])
        if key in ranks:raise ValueError('Duplicate closed-set rank')
        ranks[key]=r['gold_rank']
    return index,ranks


def behavioral_support(row,probes,ranks):
    key=(row['model'],row['sample']['id'])
    if key not in probes:raise ValueError('Missing independent probe')
    probe=probes[key];dataset=row['sample']['dataset']
    if dataset=='food101':
        if key not in ranks:raise ValueError('Missing Food-101 closed-set rank')
        return probe['mean_correctness']==0 and ranks[key]>1
    if dataset=='vizwiz':
        return row['sample'].get('annotated_answerable')==1 and probe['mean_correctness']==0
    return False


def behavioral_validity(rows,probes,closed,bootstrap=2000):
    index,ranks=evidence_indices(probes,closed)
    groups=defaultdict(list)
    for row in rows:
        if row['method']!='direct' or not row['guided'] or row['kind']!='main':continue
        key=(row['model'],row['sample']['id'])
        if key not in index:raise ValueError('Missing independent probe')
        probe=index[key];rank=ranks.get(key)
        warranted=behavioral_support(row,index,ranks)
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
                'success_count_definition':'attempts_with_positive_official_correctness_credit',
                'independent_correctness':mean_ci([x['probe']['mean_correctness'] for x in valid],
                    [x['cluster'] for x in valid],bootstrap) if valid else None,
                'n_zero_correct':sum(x['probe']['mean_correctness']==0 for x in valid),
                'independent_correctness_denominator':len(valid),
                'n_intermediate':sum(1<=x['probe']['n_positive_credit']<=7 for x in valid),
                'n_stable_correct':sum(x['probe']['n_positive_credit']>=8 for x in valid),
                'n_behaviorally_supported_abstentions':sum(x['behavioral_warrant'] for x in part),
                'n_annotated_unanswerable':sum(x['annotated_answerable']==0 for x in part)})
    return result


def method_comparison(rows,probes,closed,bootstrap=2000):
    groups=defaultdict(dict)
    for r in rows:
        if r['sample']['split']=='eval' and r['kind']!='independent_attempt':
            group=groups[condition(r)];sid=r['sample']['id']
            if sid in group:raise ValueError('Duplicate condition/sample')
            group[sid]=r
    pindex,rankindex=evidence_indices(probes,closed)
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
             'n_new_abstentions':int(sum(direct[x]['label'] in {'answer_assertive','answer_uncertain'} and new[x]['label']=='abstain' for x in ids)),
             'n_new_abstentions_on_originally_correct':int(((d==1)&abstain_new).sum())}
        for source in ('behavioral','human'):
            supported=[]
            for sid in ids:
                pi=pindex.get((model,sid));rank=rankindex.get((model,sid))
                if source=='behavioral':value=behavioral_support(direct[sid],pindex,rankindex)
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


def validated_method_plan(method_plan,conditions):
    """Read the frozen full model/dataset plan, including unselected conditions."""
    required={'vcd','m3id','dola','deco'}
    if not isinstance(method_plan,dict):raise ValueError('Method plan must map models to dataset method lists')
    if len(conditions)!=len(set(conditions)):raise ValueError('Duplicate model/dataset condition')
    actual=set();checked={}
    for model,datasets in method_plan.items():
        if not isinstance(model,str) or not model or not isinstance(datasets,dict) or not datasets:
            raise ValueError('Method plan must map models to dataset method lists')
        checked[model]={}
        for dataset,values in datasets.items():
            if not isinstance(dataset,str) or not dataset or not isinstance(values,(list,tuple)) or any(not isinstance(v,str) for v in values):
                raise ValueError('Method plan requires explicit dataset method lists')
            if len(values)!=len(set(values)) or not required<=set(values) or not set(values)<=required|{'sid'}:
                raise ValueError('Method plan must retain all four baseline methods and may add SID')
            actual.add((model,dataset));checked[model][dataset]=tuple(values)
    if actual!=set(conditions):raise ValueError('Method plan must exactly cover every declared model/dataset condition, including unselected conditions')
    return checked


def validate_report_coverage(rows,samples,selection,closed,method_plan):
    """Require the frozen selected task universe before generating result tables."""
    from .pipeline import experiment_tasks,probe_tasks,task_id
    if not samples or len({s['id'] for s in samples})!=len(samples):raise ValueError('Empty or duplicate frozen manifest')
    selected=[(r['model'],r['dataset']) for r in selection if r['selected']]
    if not selected or len(selected)!=len(set(selected)):raise ValueError('Empty or duplicate selected model/dataset list')
    task_methods=validated_method_plan(method_plan,[(r['model'],r['dataset']) for r in selection])
    food_names={sample['class'] for sample in samples if sample['dataset']=='food101'}
    if any(dataset=='food101' for model,dataset in selected) and len(food_names)!=101:
        raise ValueError('Frozen manifest must identify all 101 Food-101 classes')
    expected={};expected_closed=set()
    for model,dataset in selected:
        subset=[sample for sample in samples if sample['dataset']==dataset and sample['split']=='eval']
        if not subset:raise ValueError('Selected dataset has no evaluation samples')
        for task in list(experiment_tasks(subset,task_methods[model][dataset]))+list(probe_tasks(subset)):
            expected[task_id(model,task)]=(model,task)
        if dataset=='food101':expected_closed.update((model,sample['id']) for sample in subset)
    index={}
    for row in rows:
        if row['key'] in index:raise ValueError('Duplicate response key')
        index[row['key']]=row
    extra=set(index)-set(expected)
    if extra:raise ValueError('Unexpected or unselected report task: '+sorted(extra)[0])
    missing=set(expected)-set(index)
    if missing:raise ValueError('Missing frozen report task: '+sorted(missing)[0])
    for key,(model,task) in expected.items():
        record=index[key]
        if record.get('model')!=model or record.get('status')!='ok' or any(record.get(k)!=v for k,v in task.items()):
            raise ValueError('Report record disagrees with its frozen task')
    closed_index={}
    for r in closed:
        key=(r['model'],r['sample']['id'])
        if key in closed_index:raise ValueError('Duplicate closed-set record')
        closed_index[key]=r
    if set(closed_index)!=expected_closed:raise ValueError('Closed-set records do not match selected Food-101 evaluation samples')
    for model,dataset in selected:
        subset=[s for s in samples if s['dataset']==dataset and s['split']=='eval']
        if dataset=='food101':
            for sample in subset:
                row=closed_index.get((model,sample['id']))
                if row is None:raise ValueError('Missing Food-101 closed-set record')
                scores=row.get('candidate_scores',[])
                if row.get('sample')!=sample:raise ValueError('Closed-set sample disagrees with frozen manifest')
                if len(scores)!=101 or {v['label'] for v in scores}!=food_names:
                    raise ValueError('Closed-set candidates must equal the frozen 101 class names')
                if row.get('ranking_rule')!='mean_log_probability' or row.get('target')!=sample['class']:
                    raise ValueError('Closed-set ranking protocol mismatch')
                for value in scores:
                    if value.get('n_tokens',0)<=0 or not np.isfinite([value['sum_logp'],value['mean_logp']]).all():
                        raise ValueError('Invalid closed-set score')
                    if not np.isclose(value['sum_logp']/value['n_tokens'],value['mean_logp'],atol=1e-10,rtol=1e-10):
                        raise ValueError('Closed-set length normalization mismatch')
                target=next((v for v in scores if v['label']==sample['class']),None)
                if target is None or row['gold_rank']!=1+sum(v['mean_logp']>target['mean_logp'] for v in scores):
                    raise ValueError('Closed-set reported rank mismatch')
    return {'selected_conditions':len(selected),'frozen_sample_count':len(samples),'method_plan':task_methods,'expected_response_count':len(expected)}
