"""Full-output human-review queues and externally submitted revision histories."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from .io import read_jsonl,stable_hash,file_hash,atomic_json
from .annotation import validate_annotations
from .scoring import LABELS


def _immutable_jsonl(path,rows):
    path=Path(path);payload=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows)
    if path.exists():
        if path.read_text(encoding='utf-8')!=payload:raise ValueError('Existing review artifact differs; use a new output path')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:stream.write(payload)


def _records(paths):
    result={}
    for path in paths:
        for row in read_jsonl(path):
            if row['key'] in result:raise ValueError('Duplicate raw response key')
            result[row['key']]=row
    if not result:raise ValueError('Empty review source')
    return result


def build_human_review_queue(record_paths,annotation_path,out,error_paths=()):
    records=_records(record_paths);annotations=validate_annotations(annotation_path)
    original={row['key']:row for row in read_jsonl(annotation_path)}
    failures=defaultdict(list)
    if error_paths:
        identity_path=Path(annotation_path).with_suffix('.identity.json')
        if not identity_path.is_file():raise ValueError('Annotation failures require their original ledger identity')
        annotation_identity=json.loads(identity_path.read_text())
    for path in error_paths:
        for error in read_jsonl(path):
            key=error.get('key')
            if key not in records:raise ValueError('Unknown annotation failure key')
            if error.get('text')!=records[key]['text'] or error.get('question')!=records[key]['sample']['question']:
                raise ValueError('Annotation failure refers to changed text or question')
            if not error.get('annotation_identity') or not error.get('queue_sha256') or not error.get('error') or 'response_text' not in error:
                raise ValueError('Annotation failure lacks original identity or raw response')
            if error['annotation_identity']!=annotation_identity['identity'] or error['queue_sha256']!=annotation_identity['definition']['queue_sha']:
                raise ValueError('Annotation failure identity differs from the supplied annotation ledger')
            failures[key].append(error)
    if set(records)!=(set(annotations)|set(failures)):raise ValueError('Review requires exact complete automatic-annotation or failure coverage')
    reasons=defaultdict(set);baselines=defaultdict(list);equivalent=defaultdict(set)
    for key,row in records.items():
        if row['method']=='direct' and row.get('guided') and row['kind'] in {'main','census'}:
            baselines[(row['model'],row['sample']['id'],row['marker'])].append(key)
        if key in failures:reasons[key].add('judge_parse_failure')
        if key not in annotations:continue
        ann=annotations[key]
        if ann['text']!=row['text']:raise ValueError('Automatic annotation text mismatch')
        if ann['label']=='invalid':reasons[key].add('invalid_output')
        if ann['label']=='answer_uncertain':reasons[key].add('uncertain_concrete_answer')
        if any(original[key].get(field) is True for field in ('ambiguous','disagreement','needs_review')):
            reasons[key].add('reported_ambiguity_or_disagreement')
        equivalent[(row['sample']['question'],row['text'])].add(ann['label'])
    for key,row in records.items():
        if len(equivalent[(row['sample']['question'],row['text'])])>1:
            reasons[key].add('automatic_annotation_disagreement')
        if row['method']=='direct' or row['kind']=='independent_attempt':continue
        paired=baselines.get((row['model'],row['sample']['id'],row['marker']),[])
        if not paired:raise ValueError('Missing direct baseline for human abstention-change review')
        for base in paired:
            if base not in annotations or key not in annotations:
                reasons[key].add('paired_annotation_unresolved');reasons[base].add('paired_annotation_unresolved');continue
            if (annotations[base]['label']=='abstain')!=(annotations[key]['label']=='abstain'):
                reasons[key].add('abstention_change');reasons[base].add('abstention_change')
    rows=[]
    for key,row in records.items():
        why=sorted(reasons[key]) or ['all_remaining_outputs']
        rows.append({'key':key,'question':row['sample']['question'],'text':row['text'],
            'text_sha256':stable_hash(row['text']),'raw_record_sha256':stable_hash(row),
            'initial_annotation':annotations.get(key),'judge_failures':failures[key],
            'review_reasons':why,'priority':0 if why!=['all_remaining_outputs'] else 1,
            'review_status':'pending'})
    rows.sort(key=lambda row:(row['priority'],row['key']))
    _immutable_jsonl(out,rows)
    receipt={'schema':'kdm_full_human_review_queue_v1','queue_sha256':file_hash(out),
        'record_sources':[{'path':str(Path(path).resolve()),'sha256':file_hash(path)} for path in record_paths],
        'annotation_sha256':file_hash(annotation_path),
        'error_sources':[{'path':str(Path(path).resolve()),'sha256':file_hash(path)} for path in error_paths],
        'required_count':len(rows),
        'required_keys_sha256':stable_hash(sorted(records)),
        'remaining_checklist':'all_remaining_formal_outputs','human_review_completed':False}
    sidecar=Path(out).with_suffix('.review_queue.json')
    if sidecar.exists() and json.loads(sidecar.read_text())!=receipt:raise ValueError('Review queue provenance changed')
    if not sidecar.exists():atomic_json(sidecar,receipt)
    return receipt


def _annotation(annotation,text):
    if set(annotation)!={'label','evidence','answer_text'}:raise ValueError('Decision annotation requires exactly label, evidence and answer_text')
    label=annotation['label'];evidence=annotation['evidence'];answer=annotation['answer_text']
    if label not in LABELS:raise ValueError('Unknown human annotation label')
    if not isinstance(evidence,str) or evidence not in text or (label!='invalid' and not evidence):
        raise ValueError('Human evidence must quote the original response')
    if not isinstance(answer,str):raise ValueError('Human answer span must be a string')
    if label in {'answer_assertive','answer_uncertain'} and (not answer or answer not in text):
        raise ValueError('Human concrete answer requires its exact endorsed span')
    if label in {'abstain','invalid'} and answer:raise ValueError('Human abstention/invalid label cannot endorse an answer')
    return dict(annotation)


def _apply(row,decision):
    if decision.get('text_sha256')!=row['text_sha256']:raise ValueError('Human decision references changed response text')
    history=row['review_history'];revision=len(history)+1
    if type(decision.get('revision')) is not int or type(decision.get('previous_revision')) is not int or decision['revision']!=revision or decision.get('previous_revision')!=revision-1:
        raise ValueError('Human revision chain is not contiguous')
    reviewer=decision.get('reviewer');when=decision.get('reviewed_at')
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('External human decision requires reviewer identity')
    try:timestamp=datetime.fromisoformat(when.replace('Z','+00:00'))
    except (AttributeError,TypeError,ValueError) as exc:raise ValueError('Human review time must be an ISO timestamp') from exc
    if timestamp.tzinfo is None:raise ValueError('Human review time requires a timezone')
    if history and timestamp<datetime.fromisoformat(history[-1]['reviewed_at'].replace('Z','+00:00')):
        raise ValueError('Human revision time moves backwards')
    action=decision.get('action')
    if action not in {'approve','revise','unresolved'}:raise ValueError('Unknown human review action')
    if action=='revise':
        if not isinstance(decision.get('reason'),str) or not decision['reason'].strip():raise ValueError('A human revision requires a reason')
        updated=_annotation(decision.get('annotation',{}),row['text']);row.update(updated)
    elif 'annotation' in decision:raise ValueError('Only revise decisions may supply a changed annotation')
    elif action=='approve' and row.get('label') not in LABELS:
        raise ValueError('Unparsed judge failures require a complete human revise decision; approve is unavailable')
    row['review_status']='unresolved' if action=='unresolved' else 'reviewed'
    history.append(dict(decision))
    row['reviewer']=reviewer;row['reviewed_at']=when


def merge_human_revisions(queue_path,decision_paths,out):
    queue=list(read_jsonl(queue_path));rows={}
    for q in queue:
        if q['key'] in rows:raise ValueError('Duplicate review queue key')
        initial=q['initial_annotation']
        rows[q['key']]={**q,**(initial or {}),'review_history':[],'review_status':'pending'}
    for path in decision_paths:
        for decision in read_jsonl(path):
            key=decision.get('key')
            if key not in rows:raise ValueError('Unknown human decision key')
            _apply(rows[key],decision)
    _immutable_jsonl(out,[rows[key] for key in sorted(rows)])
    unresolved=sorted(key for key,row in rows.items() if row['review_status']!='reviewed')
    receipt={'schema':'kdm_human_review_merge_v1','queue_path':str(Path(queue_path).resolve()),'queue_sha256':file_hash(queue_path),
        'decision_sources':[{'path':str(Path(path).resolve()),'sha256':file_hash(path)} for path in decision_paths],
        'annotations_sha256':file_hash(out),'required_count':len(rows),
        'required_keys_sha256':stable_hash(sorted(rows)),'reviewed_count':len(rows)-len(unresolved),
        'missing_or_unresolved_keys':unresolved,'all_reviewed':not unresolved,
        'human_judgments_source':'externally_submitted_decision_files'}
    sidecar=Path(out).with_suffix('.human_review.json')
    if sidecar.exists() and json.loads(sidecar.read_text())!=receipt:raise ValueError('Human review merge provenance changed')
    if not sidecar.exists():atomic_json(sidecar,receipt)
    return receipt


def validate_human_review(annotation_path,record_paths):
    records=_records(record_paths)
    receipt_path=Path(annotation_path).with_suffix('.human_review.json')
    if not receipt_path.is_file():raise ValueError('Missing human-review coverage receipt')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='kdm_human_review_merge_v1' or receipt.get('annotations_sha256')!=file_hash(annotation_path):
        raise ValueError('Human-review receipt identity mismatch')
    if not receipt.get('all_reviewed') or receipt.get('required_count')!=len(records) or receipt.get('reviewed_count')!=len(records) or receipt.get('missing_or_unresolved_keys'):
        raise ValueError('Human review is incomplete: pending or unresolved outputs remain')
    if receipt.get('required_keys_sha256')!=stable_hash(sorted(records)):raise ValueError('Human-review required key set mismatch')
    annotations=validate_annotations(annotation_path)
    if set(annotations)!=set(records):raise ValueError('Human-reviewed annotations must exactly cover every supplied response')
    queue_path=Path(receipt['queue_path'])
    if file_hash(queue_path)!=receipt['queue_sha256']:raise ValueError('Original human review queue changed')
    queue={row['key']:row for row in read_jsonl(queue_path)}
    if set(queue)!=set(records):raise ValueError('Original human review queue is incomplete')
    submissions=defaultdict(list)
    for source in receipt['decision_sources']:
        if file_hash(source['path'])!=source['sha256']:raise ValueError('External human decision file changed')
        for decision in read_jsonl(source['path']):
            if decision.get('key') not in records:raise ValueError('Unknown external human decision key')
            submissions[decision['key']].append(decision)
    for row in read_jsonl(annotation_path):
        if row.get('initial_annotation')!=queue[row['key']]['initial_annotation'] or row.get('text_sha256')!=queue[row['key']]['text_sha256']:
            raise ValueError('Initial annotation differs from original human review queue')
        if row.get('review_history')!=submissions[row['key']]:raise ValueError('Review history differs from actual external submission files')
        raw=records[row['key']]
        if row.get('raw_record_sha256')!=stable_hash(raw) or row['text']!=raw['text']:
            raise ValueError('Human review belongs to changed original records')
        initial=row['initial_annotation'];reconstructed={k:v for k,v in row.items() if k not in {'label','evidence','answer_text','reviewer','reviewed_at'}}
        reconstructed.update({**(initial or {}),'review_history':[],'review_status':'pending'})
        for decision in row.get('review_history',[]):_apply(reconstructed,decision)
        if row.get('review_status')!='reviewed' or reconstructed['review_status']!='reviewed' or any(reconstructed.get(field)!=row.get(field) for field in ('label','evidence','answer_text','reviewer','reviewed_at')):
            raise ValueError('Final annotation differs from externally submitted review history')
    return annotations
