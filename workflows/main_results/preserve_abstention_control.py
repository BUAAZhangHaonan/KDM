"""Derive the registered copy-original-abstention control from frozen final rows."""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SCORE=ROOT/'outputs/annotations/main_results/score_rows.jsonl.gz'
SCORE_SHA='39d0dc145207304abc7dc9a73f91413c8ed78acb1632b454ec666b8484e59dd1'
GT=ROOT/'outputs/annotations/reference_gt/reference_gt.jsonl'
UNIFORM_GT=ROOT/'outputs/annotations/reference_gt/uniform_reference_gt.jsonl'
GT_MANIFEST=ROOT/'outputs/annotations/reference_gt/reference_gt_manifest.json'
GT_MANIFEST_SHA='2895a41c6a72cf3bd8e1f4e1c68db5b0356db89351b1c7d8f570c72bbd4ef916'
GT_RECONCILIATION=ROOT/'outputs/records/main_results/reference_gt_final_replay_reconciliation.json'
GT_RECONCILIATION_SHA='849892de2acf9c00d970237fc8909f68af7b0a8f114d95ec11b4d568e32e8e18'
STUDY=ROOT/'configs/kdm/study.json'
FREEZE_MANIFEST=ROOT/'data/provenance/frozen_contract/manifest.json'
OUT=ROOT/'outputs/analysis/main_results/controls'
COND=('model','method','kind','marker','reference_marker','guided','reference_guided','replicate')
PAIR_METHOD={'instruction_vcd':'vcd','instruction_m3id':'m3id'}
BOOT=2000
SEED=20260929


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def cond_key(row): return tuple(row[k] for k in COND)
def jsonl_bytes(rows):
    return ''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode('utf-8')
def fmt_ratio(n,d): return n/d if d else None

def load_inputs():
    if sha(SCORE)!=SCORE_SHA: raise ValueError('Final score_rows SHA differs from the authorized source')
    if sha(GT_MANIFEST)!=GT_MANIFEST_SHA: raise ValueError('Current accepted reference-GT manifest SHA differs from the registered source')
    if not GT_RECONCILIATION.is_file() or sha(GT_RECONCILIATION)!=GT_RECONCILIATION_SHA:
        raise ValueError('Accepted reference-GT replay reconciliation missing or SHA mismatch')
    manifest=json.loads(GT_MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('status')!='accepted_main_reference_GT_and_complete_independent_counts': raise ValueError('Reference-GT manifest is not accepted')
    if manifest.get('schema')!='main_results_reference_gt_manifest': raise ValueError('Reference-GT manifest schema mismatch')
    for path,key in ((GT,'outputs/annotations/reference_gt/reference_gt.jsonl'),
                     (UNIFORM_GT,'outputs/annotations/reference_gt/uniform_reference_gt.jsonl')):
        if sha(path)!=manifest['outputs'].get(key): raise ValueError(f'Reference-GT output SHA differs from manifest: {path}')
    input_checks=[]
    for rel,digest in manifest.get('inputs',{}).items():
        p=ROOT/rel;actual=sha(p) if p.is_file() else None
        if actual!=digest: raise ValueError(f'Reference-GT source missing or SHA mismatch: {rel}')
        input_checks.append({'path':rel,'manifest_sha256':digest,'current_sha256':actual,'status':'match'})
    acceptance=manifest.get('acceptance',{})
    acceptance_path=ROOT/acceptance.get('receipt_path','')
    if not acceptance_path.is_file() or sha(acceptance_path)!=acceptance.get('receipt_sha256'):
        raise ValueError('Accepted reference-GT receipt missing or SHA mismatch')
    acceptance_receipt=json.loads(acceptance_path.read_text(encoding='utf-8'))
    if acceptance_receipt.get('status')!=manifest['status']:
        raise ValueError('Reference-GT acceptance receipt status mismatch')
    input_checks.append({'path':str(acceptance_path.relative_to(ROOT)),'manifest_sha256':acceptance['receipt_sha256'],
                         'current_sha256':sha(acceptance_path),'status':'match'})
    input_checks.append({'path':str(GT_RECONCILIATION.relative_to(ROOT)),'manifest_sha256':GT_RECONCILIATION_SHA,
                         'current_sha256':sha(GT_RECONCILIATION),'status':'match'})
    study=json.loads(STUDY.read_text(encoding='utf-8'))
    if 'copy_original_abstention' not in study.get('comparison',[]):
        raise ValueError('The frozen study config does not register copy_original_abstention')
    freeze=json.loads(FREEZE_MANIFEST.read_text(encoding='utf-8'))
    item=next((x for x in freeze['entries'] if x['original_path']=='configs/kdm/study.json'),None)
    if item is None or item['sha256']!=sha(STUDY) or sha(ROOT/item['canonical_blob_path'])!=item['sha256']:
        raise ValueError('Current study config is not byte-bound to its frozen canonical blob')
    gt_maps=[]
    for path in (GT,UNIFORM_GT):
        mapping={}
        for r in read_jsonl(path):
            k=(r['model'],r['sample_id'])
            if k in mapping: raise ValueError(f'Duplicate GT row: {k}')
            if type(r.get('gt')) is not bool: raise ValueError(f'Nonbinary GT row: {k}')
            mapping[k]=r
        if len(mapping)!=24240: raise ValueError(f'Expected 24240 GT rows, got {len(mapping)} in {path}')
        gt_maps.append(mapping)
    return manifest,study,freeze,gt_maps,input_checks

def load_score_projection():
    direct={}; treatments=collections.defaultdict(dict); instructions=collections.defaultdict(dict)
    opener=gzip.open if SCORE.suffix=='.gz' else open
    with opener(SCORE,'rt',encoding='utf-8') as stream:
        for line_no,line in enumerate(stream,1):
            if not line.strip():continue
            row=json.loads(line)
            method=row['method'];kind=row['kind']
            if method not in {'direct','vcd','m3id','instruction_vcd','instruction_m3id'}:continue
            if method=='direct' and kind!='main':continue
            if method in {'instruction_vcd','instruction_m3id'} and kind!='instruction_preserving':continue
            if method in {'vcd','m3id'} and kind not in {'main','reference_instruction_removed'}:continue
            # The class target is retained only for evaluation after source selection.
            item={k:row[k] for k in ('model','method','kind','marker','reference_marker','guided','reference_guided','replicate','key','sample_id','qa_key','target_class','abstain','canonical_name_in_primary_score','behavior','main_prompt_sha256','seed')}
            if type(item['abstain']) is not bool: raise ValueError(f'Unknown abstention label at score line {line_no}')
            sc=item['canonical_name_in_primary_score']
            if sc is not None and sc not in (0,1): raise ValueError(f'Invalid canonical score at line {line_no}')
            sid=item['sample_id']
            if method=='direct':
                k=(item['model'],item['marker'],item['guided'],item['replicate'],sid)
                if k in direct: raise ValueError(f'Duplicate direct row for {k}')
                direct[k]=item
            elif method in {'vcd','m3id'}:
                g=cond_key(item)
                if sid in treatments[g]: raise ValueError(f'Duplicate method row in {g}/{sid}')
                treatments[g][sid]=item
            else:
                # Only same-marker instruction rows are included in the 40 paired comparisons.
                if item['marker']!=item['reference_marker']:continue
                g=cond_key(item)
                if sid in instructions[g]: raise ValueError(f'Duplicate instruction row in {g}/{sid}')
                instructions[g][sid]=item
    if len(treatments)!=200 or sum(len(v) for v in treatments.values())!=200*2424:
        raise ValueError(f'Expected 200 complete VCD/M3ID condition cells; got {len(treatments)}')
    if len(instructions)!=40 or sum(len(v) for v in instructions.values())!=40*2424:
        raise ValueError(f'Expected 40 complete matched-marker instruction cells; got {len(instructions)}')
    if len(direct)!=5*4*2424:
        # direct rows are one per model, marker, and sample; guided main direct only.
        raise ValueError(f'Unexpected direct baseline rows: {len(direct)}')
    return direct,treatments,instructions

def select_control(direct,treatments):
    """Selection depends solely on direct abstention and the registered treatment row."""
    controls={}; selection_rows=[]
    for group,by_sid in treatments.items():
        model,method,kind,marker,reference_marker,guided,reference_guided,replicate=group
        chosen={}
        for sid,treatment in by_sid.items():
            dk=(model,marker,guided,replicate,sid)
            if dk not in direct: raise ValueError(f'Missing exact direct comparator: {dk}')
            original=direct[dk]
            if original['seed']!=treatment['seed'] or original['main_prompt_sha256']!=treatment['main_prompt_sha256']:
                raise ValueError(f'Direct/treatment seed or main-prompt mismatch: {group}/{sid}')
            # No target or correctness field participates in this branch.
            source=original if original['abstain'] else treatment
            if source['sample_id']!=sid or source['model']!=model: raise ValueError('Selected row identity mismatch')
            chosen[sid]=source
            selection_rows.append({
                'model':model,'method':method,'kind':kind,'marker':marker,
                'reference_marker':reference_marker,'guided':guided,
                'reference_guided':reference_guided,'replicate':replicate,
                'sample_id':sid,'qa_key':treatment['qa_key'],
                'treatment_key':treatment['key'],'direct_key':original['key'],
                'selected_key':source['key'],'selected_source_method':source['method'],
                'seed':source['seed'],'main_prompt_sha256':source['main_prompt_sha256'],
                'selection_reason':'direct_abstained' if original['abstain'] else 'direct_answered',
                'direct_abstain':original['abstain'],'selected_abstain':source['abstain']})
        if len(chosen)!=2424: raise ValueError(f'Incomplete selected rows in {group}')
        controls[group]=chosen
    if len(selection_rows)!=200*2424: raise ValueError('Selection provenance count mismatch')
    return controls,selection_rows

def gt_for(mapping,model,row):
    g=mapping.get((model,row['sample_id']))
    if g is None: raise ValueError(f'Missing GT for {(model,row["sample_id"])}')
    if g['target']!=row['target_class'] or g['cluster']!=row['target_class']:
        raise ValueError(f'GT target mismatch for {(model,row["sample_id"])}')
    return g['gt']

def count_metrics(rows,gt_map,direct):
    n=len(rows);correct=unknown=answered=answered_correct=0
    tp=fp=fn=tn=0;ret_den=ret_num=0;from_direct=from_method=0
    for r in rows:
        score=r['canonical_name_in_primary_score']
        if score is None: unknown+=1
        else: correct+=int(score==1)
        abstain=r['abstain'];answered+=int(not abstain)
        if not abstain and score is not None: answered_correct+=int(score==1)
        positive=gt_map[(r['model'],r['sample_id'])]['gt']
        if abstain and positive:tp+=1
        elif abstain and not positive:fp+=1
        elif not abstain and positive:fn+=1
        else:tn+=1
        d=direct[(r['model'],r['marker'],r['guided'],r['replicate'],r['sample_id'])]
        if d['abstain'] and positive:
            ret_den+=1;ret_num+=int(abstain)
        if r['method']=='direct':from_direct+=1
        else:from_method+=1
    abstained=n-answered
    return {
        'n':n,'accuracy_correct_numerator':correct,'accuracy_unknown_n':unknown,'accuracy_denominator':n,
        'accuracy_lower':correct/n if n else None,'accuracy_upper':(correct+unknown)/n if n else None,
        'coverage_answered_n':answered,'coverage_denominator':n,'coverage':answered/n if n else None,
        'selective_accuracy_correct_numerator':answered_correct,'selective_accuracy_unknown_n':sum(r['canonical_name_in_primary_score'] is None and not r['abstain'] for r in rows),
        'selective_accuracy_denominator':answered,'selective_accuracy_lower':fmt_ratio(answered_correct,answered),
        'abstention_n':abstained,'abstention_denominator':n,
        'reasonable_abstention_retained_numerator':ret_num,'reasonable_abstention_retained_denominator':ret_den,
        'reasonable_abstention_retention':fmt_ratio(ret_num,ret_den),
        'abstention_tp':tp,'abstention_fp':fp,'abstention_fn':fn,'abstention_tn':tn,
        'abstention_precision_numerator':tp,'abstention_precision_denominator':tp+fp,
        'abstention_precision':fmt_ratio(tp,tp+fp),
        'abstention_recall_numerator':tp,'abstention_recall_denominator':tp+fn,
        'abstention_recall':fmt_ratio(tp,tp+fn),
        'selected_from_direct_n':from_direct,'selected_from_method_n':from_method}

def condition_metrics(group,selected,direct,gt_maps):
    base=dict(zip(COND,group));rows=list(selected.values())
    if len(rows)!=2424: raise ValueError(f'Wrong row count for {group}')
    # Fixed eval set is balanced: every canonical class contributes 24 samples.
    class_counts=collections.Counter(r['target_class'] for r in rows)
    if len(class_counts)!=101 or set(class_counts.values())!={24}:
        raise ValueError(f'Expected 101 target classes with 24 rows each for {group}')
    for gt_name,gt_map in zip(('mixed_reference_gt','uniform_reference_gt'),gt_maps):
        metrics=count_metrics(rows,gt_map,direct)
        if sum(metrics[k] for k in ('abstention_tp','abstention_fp','abstention_fn','abstention_tn'))!=len(rows):
            raise ValueError(f'Abstention confusion counts do not conserve rows: {group}/{gt_name}')
        base.update({f'{gt_name}_{k}':v for k,v in metrics.items() if k in {
            'reasonable_abstention_retained_numerator','reasonable_abstention_retained_denominator','reasonable_abstention_retention',
            'abstention_tp','abstention_fp','abstention_fn','abstention_tn','abstention_precision_numerator',
            'abstention_precision_denominator','abstention_precision','abstention_recall_numerator',
            'abstention_recall_denominator','abstention_recall'}})
    # Accuracy/coverage depend on the selected response only, not the GT variant.
    base.update({k:v for k,v in count_metrics(rows,gt_maps[0],direct).items() if k not in {
        'reasonable_abstention_retained_numerator','reasonable_abstention_retained_denominator','reasonable_abstention_retention',
        'abstention_tp','abstention_fp','abstention_fn','abstention_tn','abstention_precision_numerator',
        'abstention_precision_denominator','abstention_precision','abstention_recall_numerator',
        'abstention_recall_denominator','abstention_recall'}})
    return base

def metric_counts(rows,gt_map,direct):
    by=collections.defaultdict(lambda:collections.Counter())
    for r in rows:
        c=r['target_class'];score=r['canonical_name_in_primary_score'];ab= r['abstain'];pos=gt_map[(r['model'],r['sample_id'])]['gt']
        x=by[c];x['n']+=1;x['answered']+=int(not ab);x['abstain']+=int(ab)
        if score is None:x['unknown']+=1
        else:x['correct']+=int(score==1)
        if not ab and score is None:x['answered_unknown']+=1
        elif not ab:x['answered_correct']+=int(score==1)
        x['tp']+=int(ab and pos);x['fp']+=int(ab and not pos);x['fn']+=int(not ab and pos);x['tn']+=int(not ab and not pos)
        d=direct[(r['model'],r['marker'],r['guided'],r['replicate'],r['sample_id'])]
        if d['abstain'] and pos:
            x['reasonable_den']+=1;x['reasonable_keep']+=int(ab)
    return by

def ratio(counts,numer,denom,indices):
    a=b=0
    for idx in indices:
        cs=counts[CLASSES[idx]]
        a+=cs[numer];b+=cs[denom]
    return a/b if b else None
CLASSES=[]
def bootstrap_differences(instruction,control,gt_map,direct,draws):
    im=metric_counts(instruction,gt_map,direct);cm=metric_counts(control,gt_map,direct)
    definitions={
        'accuracy_lower':('correct','n'),
        'coverage':('answered','n'),
        'selective_accuracy_lower':('answered_correct','answered'),
        'abstention_precision':('tp','tp_fp'),
        'abstention_recall':('tp','tp_fn'),
        'reasonable_abstention_retention':('reasonable_keep','reasonable_den')}
    def vectors(mapping):
        nums={};dens={}
        for metric,(nk,dk) in definitions.items():
            nums[metric]=np.asarray([mapping[c][nk] for c in CLASSES],dtype=float)
            if dk=='tp_fp':dens[metric]=np.asarray([mapping[c]['tp']+mapping[c]['fp'] for c in CLASSES],dtype=float)
            elif dk=='tp_fn':dens[metric]=np.asarray([mapping[c]['tp']+mapping[c]['fn'] for c in CLASSES],dtype=float)
            else:dens[metric]=np.asarray([mapping[c][dk] for c in CLASSES],dtype=float)
        return nums,dens
    inum,iden=vectors(im);cnum,cden=vectors(cm)
    out={}
    for metric in definitions:
        ia=inum[metric][draws].sum(axis=1);ib=iden[metric][draws].sum(axis=1)
        ca=cnum[metric][draws].sum(axis=1);cb=cden[metric][draws].sum(axis=1)
        valid=(ib>0)&(cb>0)
        vals=ia[valid]/ib[valid]-ca[valid]/cb[valid]
        def point(nums,dens):
            den=float(dens.sum());return float(nums.sum()/den) if den else None
        ni=point(inum[metric],iden[metric]);nc=point(cnum[metric],cden[metric])
        out[metric]={'instruction_minus_copy_control':ni-nc if ni is not None and nc is not None else None,
            'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if len(vals) else [None,None],
            'bootstrap_draws_used':int(len(vals))}
    return out

def paired_rows(direct,treatments,controls,instructions,gt_maps,draws):
    out=[];seen=set()
    for igroup,irows in instructions.items():
        model,imethod,kind,marker,refmarker,guided,refguided,replicate=igroup
        base_method=PAIR_METHOD[imethod]
        base_group=(model,base_method,'main',marker,refmarker,guided,True,replicate)
        if base_group not in controls:raise ValueError(f'Missing matched base control {base_group}')
        base_control=controls[base_group]
        if set(irows)!=set(base_control):raise ValueError(f'Instruction/control sample mismatch {igroup}')
        ivals=list(irows.values())
        cvals=[base_control[sid] for sid in irows]
        for iv,cv in zip(ivals,cvals):
            if iv['sample_id']!=cv['sample_id'] or iv['target_class']!=cv['target_class']:
                raise ValueError(f'Instruction/control sample or target mismatch: {igroup}/{iv["sample_id"]}')
            dk=(model,marker,guided,replicate,iv['sample_id'])
            d=direct[dk]
            if iv['seed']!=cv['seed'] or iv['seed']!=d['seed']:
                raise ValueError(f'Instruction/control seed mismatch: {igroup}/{iv["sample_id"]}')
            if iv['main_prompt_sha256']!=cv['main_prompt_sha256'] or iv['main_prompt_sha256']!=d['main_prompt_sha256']:
                raise ValueError(f'Instruction/control main-prompt mismatch: {igroup}/{iv["sample_id"]}')
        row={**dict(zip(COND,igroup)),'comparison':'instruction_vs_copy_original_abstention','copy_method':base_method,
             'n_paired':len(ivals),'cluster_unit':'Food-101 target class','cluster_count':101}
        for gname,gt_map in zip(('mixed_reference_gt','uniform_reference_gt'),gt_maps):
            im=count_metrics(ivals,gt_map,direct);cm=count_metrics(cvals,gt_map,direct)
            for metric in ('accuracy_lower','coverage','selective_accuracy_lower','abstention_precision','abstention_recall','reasonable_abstention_retention'):
                a=im[metric];b=cm[metric]
                row[f'{gname}_{metric}_instruction']=a
                row[f'{gname}_{metric}_copy_control']=b
                row[f'{gname}_{metric}_difference_instruction_minus_control']=(a-b if a is not None and b is not None else None)
            row[f'{gname}_bootstrap_differences']=bootstrap_differences(ivals,cvals,gt_map,direct,draws)
        row['selection_control_group']=list(base_group)
        out.append(row);seen.add(igroup)
    if len(out)!=40:raise ValueError(f'Expected 40 matched-marker instruction comparisons, got {len(out)}')
    return out

def pooled_summary(control_rows,pair_rows,gt_maps,direct):
    rows=[]
    for method in ('vcd','m3id'):
        pairs=[p for p in pair_rows if p['copy_method']==method]
        inst=[];ctrl=[]
        for p in pairs:
            model=p['model'];imethod=p['method'];marker=p['marker'];rg=p['reference_guided'];guided=p['guided'];rep=p['replicate']
            ig=(model,imethod,'instruction_preserving',marker,marker,guided,rg,rep)
            base=(model,method,'main',marker,marker,guided,True,rep)
            # Already evaluated in paired construction; recover just aggregate values from those 20 condition rows.
            inst.append(p);ctrl.append(p)
        bymodel=collections.Counter()
        for p in pairs:
            bymodel[p['model']]+=1
        for gname in ('mixed_reference_gt','uniform_reference_gt'):
            keys=[f'{gname}_accuracy_lower_difference_instruction_minus_control',
                  f'{gname}_coverage_difference_instruction_minus_control',
                  f'{gname}_selective_accuracy_lower_difference_instruction_minus_control',
                  f'{gname}_abstention_recall_difference_instruction_minus_control',
                  f'{gname}_reasonable_abstention_retention_difference_instruction_minus_control']
            for key in keys:
                vals=[p.get(key) for p in pairs if p.get(key) is not None]
                rows.append({'copy_method':method,'reference_gt':gname,'metric':key[len(gname)+1:],
                             'paired_conditions':len(vals),'condition_mean_difference':sum(vals)/len(vals) if vals else None,
                             'interpretation':'mean of 20 matched condition-level micro differences; descriptive, paired-sample reuse across markers'})
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite existing control output: {out}')
    manifest,study,freeze,gt_maps,gt_input_checks=load_inputs()
    direct,treatments,instructions=load_score_projection()
    controls,selection_rows=select_control(direct,treatments)
    global CLASSES
    CLASSES=sorted({r['target_class'] for x in treatments.values() for r in x.values()})
    if len(CLASSES)!=101:raise ValueError(f'Expected 101 target classes, got {len(CLASSES)}')
    # All condition sample sets must contain 24 examples per target class.
    control_metric_rows=[]
    for group,selected in sorted(controls.items(),key=lambda x:x[0]):
        control_metric_rows.append(condition_metrics(group,selected,direct,gt_maps))
    rng=np.random.RandomState(SEED)
    draws=rng.randint(0,101,size=(BOOT,101))
    pair_metric_rows=paired_rows(direct,treatments,controls,instructions,gt_maps,draws)
    # Build per-method aggregate tradeoffs directly from the 20 rows per method.
    aggregate=[]
    for method in ('vcd','m3id'):
        relevant=[p for p in pair_metric_rows if p['copy_method']==method]
        for gname in ('mixed_reference_gt','uniform_reference_gt'):
            def mean_nonnull(key):
                vals=[p[key] for p in relevant if p.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            aggregate.append({'copy_method':method,'reference_gt':gname,'matched_condition_pairs':len(relevant),
                'matched_sample_comparisons':sum(p['n_paired'] for p in relevant),
                'mean_accuracy_difference_instruction_minus_control':mean_nonnull(f'{gname}_accuracy_lower_difference_instruction_minus_control'),
                'mean_coverage_difference_instruction_minus_control':mean_nonnull(f'{gname}_coverage_difference_instruction_minus_control'),
                'mean_selective_accuracy_difference_instruction_minus_control':mean_nonnull(f'{gname}_selective_accuracy_lower_difference_instruction_minus_control'),
                'mean_abstention_recall_difference_instruction_minus_control':mean_nonnull(f'{gname}_abstention_recall_difference_instruction_minus_control'),
                'mean_reasonable_abstention_retention_difference_instruction_minus_control':mean_nonnull(f'{gname}_reasonable_abstention_retention_difference_instruction_minus_control'),
                'defined_metric_cells':{k:sum(p.get(f'{gname}_{k}_difference_instruction_minus_control') is not None for p in relevant) for k in ('accuracy_lower','coverage','selective_accuracy_lower','abstention_precision','abstention_recall','reasonable_abstention_retention')},
                'note':'Arithmetic mean across defined matched condition cells; per-condition 101-class cluster bootstrap intervals are in instruction_pairs.jsonl'})
    out.mkdir(parents=True,exist_ok=False)
    # Deterministic gzip for compact, reproducible per-question source decisions.
    with (out/'selections.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as gz:
            gz.write(jsonl_bytes(selection_rows))
    def write_jsonl(path,items):path.write_bytes(jsonl_bytes(items))
    write_jsonl(out/'condition_metrics.jsonl',control_metric_rows)
    write_jsonl(out/'instruction_pairs.jsonl',pair_metric_rows)
    write_jsonl(out/'instruction_tradeoffs.jsonl',aggregate)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (SCORE,GT,UNIFORM_GT,GT_MANIFEST,STUDY,FREEZE_MANIFEST)}
    inputs.update({k:v for k,v in manifest.get('inputs',{}).items() if k in {
        'data/reference_gt/legacy_accepted_reference_gt.jsonl','data/reference_gt/legacy_accepted_conflicts.jsonl'}})
    inputs['workflows/main_results/preserve_abstention_control.py']=sha(Path(__file__))
    summary={'schema':'copy_original_abstention_control','registered_comparison':'copy_original_abstention',
        'selection_rule':'For each VCD/M3ID condition and sample, use the complete direct row iff direct.abstain is true; otherwise use the complete VCD/M3ID row. Selection reads only the direct abstain boolean.',
        'score_rows':853248,'vcd_m3id_condition_rows':len(control_metric_rows),'condition_samples':2424,
        'selection_rows':len(selection_rows),'matched_instruction_pairs':len(pair_metric_rows),
        'matched_instruction_sample_pairs':sum(r['n_paired'] for r in pair_metric_rows),
        'bootstrap':{'replicates':BOOT,'seed':SEED,'cluster_unit':'Food-101 target class','clusters':101,
                     'ratio_aggregation':'micro numerator/denominator within each resampled class multiset'},
        'reference_gt':{'mixed_rows':manifest['rows'],'uniform_rows':manifest['uniform_rows'],'manifest_status':manifest['status'],'manifest_sha256':sha(GT_MANIFEST),'acceptance_receipt':manifest['acceptance']['receipt_path'],'acceptance_receipt_sha256':manifest['acceptance']['receipt_sha256'],'reconciliation_path':str(GT_RECONCILIATION.relative_to(ROOT)),'reconciliation_sha256':GT_RECONCILIATION_SHA},
        'config_source':{'path':'configs/kdm/study.json','copy_original_abstention_registered':True,
                         'frozen_original_sha256':next(x['sha256'] for x in freeze['entries'] if x['original_path']=='configs/kdm/study.json')},
        'inputs':inputs,'reference_gt_manifest_input_checks':gt_input_checks,'tradeoffs':aggregate,
        'output_sha256':{p.name:sha(p) for p in (out/'selections.jsonl.gz',out/'condition_metrics.jsonl',out/'instruction_pairs.jsonl',out/'instruction_tradeoffs.jsonl')}}
    (out/'receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Concise, reproducible report; condition-level ratios remain available above.
    report=['# Copy-original-abstention control','',
        'This derives the registered `copy_original_abstention` condition from the final scored rows. For each VCD/M3ID condition and sample, the control uses the full direct row when direct abstains; otherwise it uses the full corresponding VCD/M3ID row. The choice uses only the direct abstention flag. All 352 main condition rows remain unchanged.',
        '',f"Inputs: final score rows SHA-256 `{inputs['outputs/annotations/main_results/score_rows.jsonl.gz']}`; mixed and uniform reference-GT files are hash-bound by `reference_gt_manifest.json`.",
        '',f"The derivation contains {len(control_metric_rows)} condition-level controls (100 each for VCD and M3ID), {len(selection_rows):,} per-question source decisions, and {len(pair_metric_rows)} matched instruction comparisons ({sum(r['n_paired'] for r in pair_metric_rows):,} paired sample-condition observations). Each condition has 2,424 samples across 101 target classes.",
        '',f"The matched comparisons use {BOOT:,} paired bootstrap resamples of the 101 Food-101 target-class clusters with seed {SEED}. Ratio metrics use pooled numerators and denominators within each resampled cluster multiset.",
        '', '## Matched instruction and copy-control tradeoff','',
        '| Base method | Reference GT | Matched conditions | Sample-condition pairs | Accuracy Δ | Coverage Δ | Selective accuracy Δ | Abstention recall Δ | Reasonable-abstention retention Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        report.append(f"| {r['copy_method']} | {r['reference_gt']} | {r['matched_condition_pairs']} | {r['matched_sample_comparisons']:,} | {r['mean_accuracy_difference_instruction_minus_control']:+.4f} | {r['mean_coverage_difference_instruction_minus_control']:+.4f} | {r['mean_selective_accuracy_difference_instruction_minus_control']:+.4f} | {r['mean_abstention_recall_difference_instruction_minus_control']:+.4f} | {r['mean_reasonable_abstention_retention_difference_instruction_minus_control']:+.4f} |")
    report+=['','Accuracy-direction counts across the 20 matched conditions per method (mixed GT; uniform GT gives the same accuracy differences):']
    for method in ('vcd','m3id'):
        vals=[p['mixed_reference_gt_accuracy_lower_difference_instruction_minus_control'] for p in pair_metric_rows if p['copy_method']==method]
        cis=[p['mixed_reference_gt_bootstrap_differences']['accuracy_lower']['ci95'] for p in pair_metric_rows if p['copy_method']==method]
        pos=sum(v>0 for v in vals); neg=sum(v<0 for v in vals)
        ci_pos=sum(lo>0 for lo,hi in cis); ci_neg=sum(hi<0 for lo,hi in cis)
        report.append(f'- {method}: accuracy Δ positive {pos}/20, negative {neg}/20; 95% cluster-bootstrap CI fully positive {ci_pos}/20, fully negative {ci_neg}/20.')
    report+=['','Two registered joint conditions: instruction minus copy-control differences, by reference GT:','',
        '| Model | Condition | GT | Accuracy Δ (95% CI) | Reasonable-abstention retention Δ (95% CI) |',
        '|---|---|---|---:|---:|']
    joints=[('minicpm26','UNCLEAR','instruction_vcd'),('qwen35_4b','I cannot identify it','instruction_vcd')]
    for model,marker,method in joints:
        found=next((p for p in pair_metric_rows if p['model']==model and p['marker']==marker and p['method']==method),None)
        if found is None: raise ValueError(f'Missing requested joint condition: {model}/{marker}/{method}')
        for gname in ('mixed_reference_gt','uniform_reference_gt'):
            acc=found[f'{gname}_accuracy_lower_difference_instruction_minus_control']
            aci=found[f'{gname}_bootstrap_differences']['accuracy_lower']['ci95']
            ret=found[f'{gname}_reasonable_abstention_retention_difference_instruction_minus_control']
            rci=found[f'{gname}_bootstrap_differences']['reasonable_abstention_retention']['ci95']
            fmt=lambda v,ci: 'NA' if v is None else f'{v:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]'
            report.append(f"| {model} | {method}/{marker} | {gname} | {fmt(acc,aci)} | {fmt(ret,rci)} |")
    report+=['','Pooled differences above are arithmetic means across matched condition cells. The per-metric defined-cell counts are recorded in `instruction_tradeoffs.jsonl` (reasonable-abstention retention has 19 cells when the Gemma UNKNOWN cell has a zero denominator). Accuracy, coverage, selective accuracy, abstention precision/recall, and retention differences are paired within the same model, marker, prompt, seed, replicate, and sample; selection itself reads only direct abstention. Confusion totals and exact denominators are in `condition_metrics.jsonl`; all per-question source choices are in `selections.jsonl.gz`.','']
    (out/'report.md').write_text('\n'.join(report),encoding='utf-8')
    # Rewrite receipt only after every output exists, adding the report digest.
    summary['output_sha256']['report.md']=sha(out/'report.md')
    (out/'receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'controls':len(control_metric_rows),'selections':len(selection_rows),
        'instruction_pairs':len(pair_metric_rows),'receipt_sha256':sha(out/'receipt.json'),
        'report_sha256':sha(out/'report.md')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
