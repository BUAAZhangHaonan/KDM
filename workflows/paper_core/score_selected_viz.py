#!/usr/bin/env python3
"""Score only explicitly received new main Viz512 rows using the accepted scorer."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.decoding import DecodeConfig
from kdm.io import file_hash,stable_hash,within
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.scoring import vqa_score
from workflows.main_results import score as frozen
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.score_dev_viz import audit_raw,viz_references,viz_inferences,viz_metrics,FIELDS,COND,AUTHORITY
from workflows.paper_core.score_native import save_json,save_rows
from workflows.supplemental.remaining11.score import rows
from workflows.paper_core.score_ip_dev import CORE,EXTRA
OLD_VIZ='outputs/paper_core_20261002_dev_viz/merged_viz31_all_five_closed_20261003_0410/receipt.json'
EXECUTOR={'agent':'/root/final_cpu_intake','model':'gpt-6.1-sol','effort':'max','call_id':''}


def existing_sources(manifest_path,samples,roster_path):
    manifest=json.loads(manifest_path.read_text())
    if (manifest['schema']!='kdm_Viz512_existing_completed_main_source_index_v1' or manifest['passed'] is not True
            or manifest['rows']!=3075 or manifest['native_reused']!=1027 or manifest['Direct_reused']!=2048
            or manifest['old_core31_rescore'] is not False or manifest['active_raw_sources'] is not False
            or manifest['new_generations']!=0 or manifest['roster_sha256']!=file_hash(roster_path)
            or file_hash(within(ROOT,manifest['native_reuse_proof']))!=manifest['native_reuse_proof_sha256']):
        raise ValueError('An explicit exact completed main Viz source index is required')
    selected,sources,seen=[],[],set()
    for source in manifest['sources']:
        model=source['model'];kind=source['source_type']
        if model not in EXTRA or source['original_prompt_seed_config_verified'] is not True:
            raise ValueError('Only the four admitted extension baselines belong to this received cohort')
        files={}
        for field in ['raw','identity']+(['complete'] if 'complete' in source else []):
            path=within(ROOT,source[field])
            if file_hash(path)!=source[field+'_sha256']:raise ValueError('A received completed member changed: '+field)
            files[field]=path
        sidecar=json.loads(files['identity'].read_text());definition=sidecar['definition']
        spec=json.loads((ROOT/f'configs/runtime/{model}.json').read_text())
        backend=definition['backend']
        if (sidecar['identity']!=stable_hash(definition) or definition['model']!=model
                or backend['hf_model_id']!=spec['hf_model_id'] or backend['dtype']!=spec['dtype']):
            raise ValueError('The original checkpoint, precision or identity differs')
        if kind=='original_native_complete_part_fixed512_exact_subset':
            complete=json.loads(files['complete'].read_text())
            if (source['method']!='vcd' or source['full_original_part_scientific_validation_passed'] is not True
                    or complete['generation_complete'] is not True or complete['raw_sha256']!=source['raw_sha256']
                    or complete['identity_sha256']!=source['identity_sha256']
                    or complete['model']!=model or complete['method']!='vcd' or complete['stage']!='native_unguided'):
                raise ValueError('The native source is not an actually completed original part')
        elif kind=='original_completed_census_gzip_fixed512_exact_subset':
            if (source['method']!='direct' or 'complete' in source
                    or source['original_identity_hash']!=sidecar['identity']
                    or source['source_checkpoint_matches_frozen_spec'] is not True):
                raise ValueError('The original admitted census source differs')
        else:raise ValueError('An unsupported source type entered the baseline cohort')
        bindings=source['source_line_bindings'];wanted={x['raw_line']:x for x in bindings}
        if len(wanted)!=source['rows'] or len(bindings)!=source['rows'] or len(set(source['completed_keys']))!=source['rows']:
            raise ValueError('The fixed512 exact line/key bindings duplicate')
        actual=[]
        for line,row,line_sha in rows(files['raw']):
            if line not in wanted:continue
            binding=wanted[line];sample=samples.get(binding['sample_id'])
            if (sample is None or row['key']!=binding['key'] or row['sample']['id']!=binding['sample_id']
                    or row['identity']!=sidecar['identity'] or row['method']!=source['method'] or row['key'] in seen
                    or row['guided'] is not False or row['reference_guided'] is not False or row['replicate']!=0):
                raise ValueError('The actual existing row differs from the fixed512 original binding')
            if kind.startswith('original_native'):
                if (row['kind'],row['marker'],row['reference_marker'])!=('native_unguided','NONE','NONE'):
                    raise ValueError('The original native VCD identity changed')
            elif (row['kind'],row['marker'],row['reference_marker'])!=('unguided','UNKNOWN','UNKNOWN'):
                raise ValueError('The original Direct metadata changed')
            audit_raw(row,sample,model);seen.add(row['key']);actual.append(row['key'])
            selected.append((row,dict(source_path=str(files['raw']),source_line=line,raw_line_sha256=line_sha,
                raw_source_sha256=source['raw_sha256'],source_identity=row['identity'],generation_identity=row['identity'],
                source_scope='existing_completed_exact_fixed512_subset',expected_n=512,source_raw_complete=True,
                complete_source_path=str(files.get('complete','')),identity_source_path=str(files['identity']),
                source_original_path=str(files['raw']),model_checkpoint=spec['hf_model_id'],model_dtype=spec['dtype'])))
        if set(actual)!=set(source['completed_keys']) or len(actual)!=source['rows']:
            raise ValueError('The actual received row/key coverage differs from its finite binding')
        sources.append({**{k:v for k,v in source.items() if k not in ['source_line_bindings','completed_keys']},
                        'raw_path':str(files['raw']),'original_condition_metadata_preserved':True})
    if len(selected)!=manifest['rows'] or len(sources)!=manifest['source_parts']:
        raise ValueError('The received baseline source count differs')
    return selected,sources


def old_closed_authorities():
    path=ROOT/OLD_VIZ;receipt=json.loads(path.read_text())
    if (receipt['passed'] is not True or receipt['rows']!=15872 or receipt['conditions']!=31
            or receipt['quality_pending_rows'] or receipt['abstain_pending_rows'] or receipt['pending_QA']):
        raise ValueError('The accepted old core31 authority is not closed')
    decisions,viz,seen=[],[],set()
    for source in receipt['source_score_receipts']:
        for field,target in [('decision_files',decisions),('viz_authority_files',viz)]:
            for item in source[field]:
                value=within(ROOT,item['path'])
                if file_hash(value)!=item['sha256']:raise ValueError('An accepted old semantic authority changed')
                if (field,str(value)) not in seen:seen.add((field,str(value)));target.append(value)
    return decisions,viz,dict(path=str(path),sha256=file_hash(path),old_rows_rescored=0)


def score_records(source_records,qa_cache,reference_map,reference_path):
    scored,pending,memberships=[],{},[]
    normalizer=VQAEval(None,None)
    normalize=lambda text:normalizer.processDigitArticle(normalizer.processPunctuation(text))
    for row,provenance in source_records:
        sample,answer=row['sample'],row['text']
        qkey=frozen.qah(sample['question'],answer)
        inferred = qa_cache[qkey]
        abstain, span = inferred["abstain"], inferred["answer_text"]
        raw_quality = float(vqa_score(answer, sample["official_answers"], normalize))
        quality = (0.0 if abstain is True or inferred["label"] == "invalid" else
                   float(vqa_score(span, sample["official_answers"], normalize))
                   if abstain is False and inferred["quality_span_resolved"] else None)
        if not 0 <= raw_quality <= 1 or (quality is not None and not 0 <= quality <= 1):
            raise ValueError("Official VizWiz answer quality lies outside [0, 1]")
        state = ("A" if abstain is True else "FULL" if quality == 1 else
                 "PARTIAL" if quality is not None and quality > 0 else "ZERO" if quality == 0 else None)
        ref = reference_map[sample["id"]]
        condition = {"model": row["model"], "dataset": sample["dataset"], "split": sample["split"],
                     **{field: row[field] for field in FIELDS}}
        authors = []
        for entry in inferred["authority_sources"]:
            annotation = entry.get("actual_annotation", entry.get("decision", entry))
            model = annotation.get("annotation_model", annotation.get("judge_response_model"))
            if model:
                authors.append({"model": model, "effort": annotation.get("annotation_effort"),
                    "call_id": annotation.get("annotation_call_id", annotation.get("judge_response_id", "")),
                    "source_path": entry.get("source_path", entry.get("decision_source_path")),
                    "source_line": entry.get("source_line", entry.get("decision_source_line"))})
        record = {**condition, **provenance, "condition_id": stable_hash(condition)[:16], "main_marker": row["marker"],
            "sample_id": sample["id"], "key": row["key"], "qa_key": qkey, "question": sample["question"], "answer": answer,
            "answer_text": span, "label": inferred["label"], "image_source": sample["image_path"], "tokens": row["tokens"],
            "selected_log_probabilities": row["selected_log_probabilities"], "first_probability": row["first_probability"],
            "config": row["config"], "config_sha256": stable_hash(row["config"]), "seed": row["seed"],
            "prompt": row["prompt"], "reference_prompt": row["reference_prompt"], "neutral_prompt": row["neutral_prompt"],
            "offset_prompt_tokens": row.get("offset_prompt_tokens"), "terminated": row["terminated"],
            "official_raw_score": raw_quality, "quality_score": quality, "quality_state": state, "abstain": abstain,
            "official_reference": bool(ref.official_reference), "annotated_answerable": int(ref.annotated_answerable),
            "reference_source_path": str(reference_path), "official_record_index": int(ref.official_record_index),
            "official_answers": sample["official_answers"], "uniform_reference": None,
            "correct_canonical": None, "correct_literal": None, "target_class": None,
            "score_reason": inferred["behavior_source"], "behavior_source": inferred["behavior_source"],
            "semantic_authority_sources": inferred["authority_sources"], "annotation_authors": authors,
            "quality_span_resolved": inferred["quality_span_resolved"], "full_reply_fallback_used": False,
            "new_scientific_judgment": False, "rule_executor": EXECUTOR}
        scored.append(record)
        needs = {"quality": quality is None, "abstain": abstain is None}
        if any(needs.values()):
            item = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": answer,
                "dataset": "vizwiz", "reason": [inferred["behavior_source"]], "needs": dict(needs)})
            item["needs"] = {key: item["needs"][key] or value for key, value in needs.items()}
            memberships.append({"qa_key": qkey, "model": row["model"], "sample_id": sample["id"], "key": row["key"],
                "official_reference": bool(ref.official_reference), **provenance, "needs": needs})
    return scored,pending,memberships


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer-manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--authority-manifest',default=AUTHORITY)
    p.add_argument('--decision-file',action='append',default=[])
    p.add_argument('--viz-authority-file',action='append',default=[])
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This scorer is CPU only')
    started=time.perf_counter();output=within(ROOT,args.out);output.relative_to(ROOT/'outputs/paper_core_20261002_dev_viz')
    if output.exists():raise FileExistsError('A new immutable score result needs an exclusive directory')
    roster_path,roster_samples=roster('viz512');samples={s['id']:s for s in roster_samples}
    manifest_path=within(ROOT,args.producer_manifest);source_records,sources=existing_sources(manifest_path,samples,roster_path)
    refs,reference_path=viz_references(samples);reference_map={r.sample_id:r for r in refs.itertuples()}
    decisions,viz,oldproof=old_closed_authorities()
    decisions += [within(ROOT,x) for x in args.decision_file];viz += [within(ROOT,x) for x in args.viz_authority_file]
    authority_path=within(ROOT,args.authority_manifest);authority=json.loads(authority_path.read_text())
    qa_cache,census_proof=viz_inferences(source_records,authority,decisions,viz)
    scored,pending,memberships=score_records(source_records,qa_cache,reference_map,reference_path)
    output.mkdir(parents=True,exist_ok=False)
    save_rows(output/'score_rows.jsonl.gz',scored)
    save_rows(output/'pending_targetblind.jsonl',[{'qa_key':x['qa_key'],'question':x['question'],'answer':x['answer'],'needs':x['needs']}for x in pending.values()])
    save_rows(output/'pending_private_memberships.jsonl',memberships)
    frame=pd.DataFrame(scored)
    metrics=[viz_metrics(g,dict(zip(COND,k)))for k,g in frame.groupby(list(COND),dropna=False,observed=True)]
    pd.DataFrame(metrics).to_csv(output/'metrics_actual.csv',index=False)
    refs.to_csv(output/'official_fixed512_references.csv',index=False);pd.DataFrame(sources).to_csv(output/'sources.csv',index=False)
    result=dict(schema='kdm_nine_selected_main_Viz512_incremental_cpu_scoring_v1',passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),rows=len(frame),conditions=len(metrics),
        complete_conditions=sum(x['primary_complete'] for x in metrics),
        quality_pending_rows=int(frame.quality_score.isna().sum()),abstain_pending_rows=int(frame.abstain.isna().sum()),
        pending_QA=len(pending),pending_memberships=len(memberships),reference_join_missing=0,
        official_unanswerable_roster_n=166,official_answerable_roster_n=346,
        source_manifest_path=str(manifest_path),source_manifest_sha256=file_hash(manifest_path),
        source_parts=len(sources),roster_path=str(roster_path),roster_sha256=file_hash(roster_path),
        official_reference_path=str(reference_path),official_reference_sha256=file_hash(reference_path),
        accepted_old_core31_authorities=oldproof,old_core31_rescored_rows=0,old_scores_changed=False,
        decision_files=[dict(path=str(x),sha256=file_hash(x))for x in decisions],
        viz_authority_files=[dict(path=str(x),sha256=file_hash(x))for x in viz],census_source=census_proof,
        original_viz_scorer_sha256=file_hash(ROOT/'workflows/paper_core/score_dev_viz.py'),
        official_scoring_sha256=file_hash(ROOT/'src/kdm/scoring.py'),
        official_normalizer_sha256=file_hash(ROOT/'src/kdm/models/official_vqa_normalizer.py'),
        runner_sha256=file_hash(Path(__file__)),actual_command=sys.argv,elapsed_s=time.perf_counter()-started,
        GPU_initialized=False,new_generations=0,full_reply_fallback_used=False,
        Food_reference_substituted=False,partial_official_scores_binarized=False,
        outputs={x.name:file_hash(x)for x in output.iterdir()if x.is_file()})
    save_json(output/'receipt.json',result)
    print(json.dumps({k:result[k]for k in ['passed','rows','conditions','complete_conditions','quality_pending_rows','abstain_pending_rows','pending_QA']}))

if __name__=='__main__':main()
