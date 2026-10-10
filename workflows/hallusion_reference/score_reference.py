#!/usr/bin/env python3
"""CPU-only quality queues and independent Hallusion reference/J aggregation.

No generation, semantic model invocation, old-data writes, or ACTIVE pointer.
Quality decisions are full-question/full-answer/full-reference judgments, never
behavior labels. All public modes read only sealed, ledger-bound new attempts.

Manual decision JSONL contract (one object per deduplicated quality input):
  schema: hallusion_independent_quality_decision_v1
  quality_key, request_sha256, question, answer, gt_answer_details
  source_bindings: nonempty list copied from the request's memberships
  quality_review: the existing Hallusion full-reference Luna review fields
The review's model/effort must be actual gpt-5.6-luna/medium. An explicit
quality_label='unknown' additionally requires score=null and
official_correctness=null. Missing reviews remain pending; unknown is never
an incorrect answer. New matching memberships can reuse an exact input review.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


BASE_REL = 'outputs/hallusion_independent128_20261008'
MANIFEST_REL = 'data/general_vqa_direct_20261004/frozen/manifest_hallusionbench.jsonl'
MANIFEST_SHA = 'a7d2c81813474478e9045e3164ce400a8ad8ab59e86a5f71c674f296d0a06e51'
FROZEN_REL = 'outputs/hallusion_blind128_20261006/scoring/snapshot_074/scores.jsonl'
FROZEN_SHA = '1f8beaeeab7d202f445a4c000943348f7e351de185de84925de73a7f0dc93fcb'
RULE_SOURCES = {
    'workflows/general_vqa_direct/hallusion_scoring.py':
        '95d029a9568da0eef069ccfd12ac5e98a3d3400925c213a33411b5941242516a',
    'workflows/hallusion_blind/complete_response_quality.py':
        '48414f4073fd9f7f6aa8e8d7911867f2498f3fee00407a833408bfaebccded01',
}
MODELS = ('qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b',
          'internvl35_8b', 'onevision', 'phi35', 'qwen3vl')
CONDITIONS_REL = 'outputs/hallusion_blind128_20261006/registration/conditions.json'
CONDITIONS_SHA = '67d24d4b92b4f5e4ad4af1f466f80c0b121a2178f9cec52eac87971a03d5f2a3'
GEN_SOURCES = (
    'workflows/hallusion_reference/generate_independent128.py',
    'workflows/hallusion_reference/test_independent128.py',
    'workflows/hallusion_blind/generate128.py', 'workflows/general_vqa_direct/generate.py',
    'src/kdm/io.py', 'src/kdm/pipeline.py', 'src/kdm/decoding.py',
    'src/kdm/probability.py', 'src/kdm/prompts.py', 'src/kdm/protocol.py',
    'src/kdm/frozen.py', 'src/kdm/execution.py', 'src/kdm/models/hf.py',
    'src/kdm/models/backbone.py', 'src/kdm/models/remote.py', 'src/kdm/models/sid.py',
    'src/kdm/models/internvl_dual.py', 'src/kdm/models/internvl_preprocessing.py',
    'workflows/supplemental/remaining11/execution.py',
    'workflows/supplemental/remaining4/k100_intern_registered_matrix.py',
    'workflows/supplemental/remaining4/internvl_k100_single.py',
)
NATIVE_CONFIG = {'method': 'direct', 'alpha': 1.0, 'beta': 0.1,
    'm3id_lambda': 0.02, 'm3id_threshold': 0.3, 'm3id_offset': 0,
    'deco_alpha': 0.6, 'deco_topk': 20, 'deco_topp': 0.9,
    'max_tokens': 128, 'temperature': 1.0, 'top_p': 1.0}
N_SAMPLES, N_REPLICATES, N_CONDITIONS = 951, 10, 68
EXPECTED_ATTEMPTS = len(MODELS) * N_SAMPLES * N_REPLICATES
ATTEMPT_SUFFIX = '\nGive your best estimate as a specific short answer.'
ITEM_FIELDS = ('method', 'marker', 'reference_marker', 'guided',
               'reference_guided', 'kind', 'replicate', 'attempt', 'source_condition_identity')
BINDING_FIELDS = ('key', 'model', 'sample_id', 'replicate', 'seed',
                  'source_raw', 'source_line', 'source_sha256')
QUALITY_LABELS = {'incorrect': 0, 'correct': 1, 'unclear': 2}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def same(left, right):
    """JSON booleans are distinct from numeric values in all source identities."""
    return digest(left) == digest(right)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Blank JSONL record: {path}:{number}')
            try:
                row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(
                    ValueError('Nonfinite JSON number: ' + s)))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f'Invalid JSONL: {path}:{number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Object required: {path}:{number}')
            yield number, row


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON object required: ' + str(path))
    return value


def bound_path(parent, value):
    parent = Path(parent).resolve()
    candidate = Path(value)
    candidate = (candidate if candidate.is_absolute() else parent / candidate).resolve()
    if not candidate.is_relative_to(parent):
        raise ValueError('Path escapes its registered directory: ' + str(candidate))
    return candidate


def project_path(root, value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else bound_path(root, value)


def quality_key(question, answer, reference):
    if not all(isinstance(value, str) for value in (question, answer, reference)):
        raise ValueError('Full question/answer/reference must be strings')
    return digest(['hallusion_full_reference_quality_v1', question, answer, reference])


def request_identity(request):
    return digest({'schema': 'hallusion_independent_full_quality_input_v1',
        **{field: request[field] for field in
           ('quality_key', 'question', 'answer', 'gt_answer_details')}})


def binding(row):
    return {field: row[field] for field in BINDING_FIELDS}


def check_full_quality(row):
    key = quality_key(row['question'], row['answer'], row['gt_answer_details'])
    q = row.get('quality')
    if (row.get('quality_key') != key or not isinstance(q, dict)
            or q.get('quality_key') != key or q.get('quality_label') not in QUALITY_LABELS
            or type(q.get('score')) is not int or q['score'] not in (0, 1)
            or q.get('official_correctness') != QUALITY_LABELS[q['quality_label']]
            or row.get('score') != q['score']
            or q['score'] != int(q['quality_label'] == 'correct')
            or not isinstance(row.get('quality_source'), dict) or not row['quality_source']):
        raise ValueError('Frozen full-QA quality/source binding differs')
    return key


def load_frozen_quality(path, expected_sha=FROZEN_SHA, expected_rows=N_SAMPLES*N_CONDITIONS,
                        expected_conditions=N_CONDITIONS):
    """The pinned snapshot is authoritative only for its exact full input."""
    if file_sha(path) != expected_sha:
        raise ValueError('Frozen snapshot_074 scores SHA differs')
    index, conditions, identities, count = {}, {}, set(), 0
    for line, row in jsonl(path):
        count += 1
        key = check_full_quality(row)
        if row.get('dataset') != 'hallusionbench' or type(row.get('abstain')) is not bool:
            raise ValueError('Frozen Hallusion behavior/quality is not complete')
        identity = (row['condition_identity'], row['sample_id'])
        if identity in identities:
            raise ValueError('Duplicate frozen condition/sample')
        identities.add(identity)
        conditions.setdefault(row['condition_identity'], set()).add(row['sample_id'])
        candidate = {'quality': row['quality'], 'question': row['question'],
            'answer': row['answer'], 'gt_answer_details': row['gt_answer_details'],
            'quality_source': {'reuse_snapshot': str(Path(path).resolve()),
                'reuse_line': line, 'snapshot_sha256': expected_sha,
                'original_quality_source': row['quality_source']},
            'reuse_eligible': True, 'snapshot_memberships': []}
        previous = index.get(key)
        if previous and any(previous['quality'][field] != candidate['quality'][field]
                for field in ('quality_label', 'score', 'official_correctness')):
            previous['reuse_eligible'] = False
        index.setdefault(key, candidate)
        index[key]['snapshot_memberships'].append({
            'snapshot_path': str(Path(path).resolve()), 'snapshot_sha256': expected_sha,
            'snapshot_line': line, **{field: row.get(field) for field in
                ('model', 'sample_id', 'condition_identity', 'key', 'source_raw', 'source_line', 'source_sha256')},
            'quality_label': row['quality']['quality_label'], 'score': row['quality']['score'],
            'official_correctness': row['quality']['official_correctness'],
            'quality': row['quality'], 'quality_source': row['quality_source']})
    if count != expected_rows or len(conditions) != expected_conditions:
        raise ValueError('Frozen snapshot_074 row/condition census differs')
    if expected_rows == N_SAMPLES*N_CONDITIONS and any(
            len(samples) != N_SAMPLES for samples in conditions.values()):
        raise ValueError('Frozen snapshot_074 whole951 denominator differs')
    return index


def reuse_quality(index, sample, answer):
    question, reference = sample['question'], sample['gt_answer_details']
    key = quality_key(question, answer, reference)
    previous = index.get(key)
    if previous is None:
        return None
    if (previous['question'] != question or previous['answer'] != answer
            or previous['gt_answer_details'] != reference):
        raise ValueError('Exact frozen quality input differs')
    if previous.get('reuse_eligible') is False:
        return None
    return copy.deepcopy({field: previous[field] for field in
        ('quality', 'question', 'answer', 'gt_answer_details', 'quality_source')})


def frozen_conflict_audit(index):
    """Keep every old source/label separately; none is a fresh-review prompt."""
    result = []
    for key, item in sorted(index.items()):
        if item.get('reuse_eligible') is not False:
            continue
        variants = Counter((row['quality_label'], row['score'], row['official_correctness'])
            for row in item['snapshot_memberships'])
        result.append({'schema': 'hallusion_frozen_quality_conflict_audit_v1',
            'quality_key': key, **{field: item[field] for field in ('question', 'answer', 'gt_answer_details')},
            'reuse_eligible': False, 'resolution_policy': 'fresh_full_reference_review_only_for_reference_sufficient_inputs',
            'snapshot_rows': len(item['snapshot_memberships']),
            'variants': [{'quality_label': label, 'score': score, 'official_correctness': code, 'rows': count}
                for (label, score, code), count in sorted(variants.items())],
            'snapshot_memberships': copy.deepcopy(item['snapshot_memberships'])})
    return result


def validate_reliability(path, expected_sha, samples, manifest_sha):
    if file_sha(path) != expected_sha:
        raise ValueError('Registered reference reliability SHA differs')
    registry = read_json(path)
    if (registry.get('schema') != 'kdm_hallusion_reference_sufficiency_v1'
            or registry.get('manifest_sha256') != manifest_sha
            or not isinstance(registry.get('rows'), list)):
        raise ValueError('Reference reliability registry schema/manifest differs')
    index = {}
    for number, row in enumerate(registry['rows'], 1):
        sid = row.get('sample_id')
        if sid not in samples or sid in index:
            raise ValueError('Foreign/duplicate reference reliability sample')
        sample = samples[sid]
        if ('reference_sufficient' not in row
                or row['reference_sufficient'] is not None
                and type(row['reference_sufficient']) is not bool
                or row.get('source_manifest_sha256') != manifest_sha
                or any(row.get(field) != sample[field] for field in
                       ('question', 'gt_answer_details', 'gt_answer'))
                or not isinstance(row.get('reason'), str) or not row['reason'].strip()):
            raise ValueError('Explicit reliability/full-reference binding differs')
        index[sid] = {**row, 'registry_source': {'path': str(Path(path).resolve()),
            'sha256': expected_sha, 'row': number}}
    if set(index) != set(samples):
        raise ValueError('Reliability registry must explicitly cover every frozen item')
    return index


def load_manual(paths, requests, scoring):
    """Read actual explicit decisions; no semantic judgments are generated here."""
    result, sources = {}, []
    for path in paths:
        sha = file_sha(path)
        sources.append({'path': str(Path(path).resolve()), 'sha256': sha})
        for line, row in jsonl(path):
            if row.get('schema') != 'hallusion_independent_quality_decision_v1':
                raise ValueError('Manual quality decision schema differs')
            request = requests.get(row.get('quality_key'))
            if request is None:
                raise ValueError('Manual judgment has no validated current raw input')
            for field in ('quality_key', 'request_sha256', 'question', 'answer', 'gt_answer_details'):
                if row.get(field) != request[field]:
                    raise ValueError('Manual full-input hash binding differs: ' + field)
            source_bindings = row.get('source_bindings')
            if not isinstance(source_bindings, list) or not source_bindings:
                raise ValueError('Manual judgment needs explicit validated source bindings')
            current = {digest(binding(member)) for member in request['memberships']}
            seen_bindings = set()
            for member in source_bindings:
                if not isinstance(member, dict) or set(member) != set(BINDING_FIELDS):
                    raise ValueError('Manual source binding fields differ')
                h = digest(member)
                if h not in current or h in seen_bindings:
                    raise ValueError('Manual source path/SHA/line/key differs or duplicates')
                seen_bindings.add(h)
            review = row.get('quality_review')
            if not isinstance(review, dict):
                raise ValueError('Full-reference quality_review required')
            for field in ('quality_key', 'question', 'answer', 'gt_answer_details'):
                if review.get(field) != request[field]:
                    raise ValueError('Manual quality review full input differs: ' + field)
            sample = request['_sample']
            if review.get('quality_label') == 'unknown':
                if ('score' not in review or review['score'] is not None
                        or 'official_correctness' not in review
                        or review['official_correctness'] is not None):
                    raise ValueError('Explicit unknown must carry null correctness/score')
                validate_explicit_unknown(review, request)
                quality = {'quality_key': request['quality_key'],
                    'quality_label': 'unknown', 'official_correctness': None,
                    'score': None, 'source': 'explicit_full_reference_quality_unknown_v1',
                    'reviewer': {field: review[field] for field in
                                 ('author', 'model', 'effort', 'call_id')},
                    'reason': review['reason']}
            else:
                quality = scoring.validate_quality(sample, request['answer'], review)
            candidate = {'quality': quality, 'question': request['question'],
                'answer': request['answer'], 'gt_answer_details': request['gt_answer_details'],
                'quality_source': {'decision_path': str(Path(path).resolve()),
                    'decision_line': line, 'decision_sha256': sha,
                    'request_sha256': row['request_sha256'],
                    'source_bindings': source_bindings}}
            key = request['quality_key']
            if key in result and any(result[key]['quality'].get(field) != quality.get(field)
                    for field in ('quality_label', 'score', 'official_correctness')):
                raise ValueError('Conflicting explicit manual quality decisions')
            result.setdefault(key, candidate)
    return result, sources


def validate_explicit_unknown(review, request):
    """Validate unknown provenance/evidence without inventing a quality label."""
    if (not isinstance(review.get('author'), str)
            or re.fullmatch(r'/root(?:/[a-z0-9_]+)*', review['author']) is None
            or review.get('model') != 'gpt-5.6-luna' or review.get('effort') != 'medium'
            or not isinstance(review.get('reason'), str) or not review['reason'].strip()
            or 'call_id' not in review or review['call_id'] is not None and
                (not isinstance(review['call_id'], str) or not review['call_id'].strip())):
        raise ValueError('Explicit unknown needs actual Luna/medium reviewer provenance/reason')
    for field, full in (('answer_evidence', request['answer']),
                        ('reference_evidence', request['gt_answer_details'])):
        if not isinstance(review.get(field), str) or review[field] not in full or full and not review[field]:
            raise ValueError('Explicit unknown evidence must be a complete-input original span: ' + field)


def prepare_rows(attempts, samples, reliability, frozen, scoring, manual_paths=()):
    requests, prepared = {}, []
    for row in attempts:
        sample, answer = samples[row['sample_id']], row['answer']
        scoring.source_record(sample)
        request = scoring.quality_request(sample, answer)
        if request['quality_key'] != quality_key(sample['question'], answer, sample['gt_answer_details']):
            raise ValueError('Frozen full-reference quality-key algorithm differs')
        key = request['quality_key']
        if reliability[row['sample_id']]['reference_sufficient'] is True:
            item = requests.setdefault(key, {**request,
                'schema': 'hallusion_independent_quality_request_v1',
                'need_quality': True, 'need_behavior': False, 'memberships': [],
                'request_sha256': request_identity(request), '_sample': sample})
            item['memberships'].append(binding(row))
        prepared.append({**row, 'quality_key': key})
    manual, manual_sources = load_manual(manual_paths, requests, scoring)
    pending = {}
    for row in prepared:
        sample, answer = samples[row['sample_id']], row['answer']
        reused = reuse_quality(frozen, sample, answer)
        conflicted = frozen.get(row['quality_key'], {}).get('reuse_eligible') is False
        if conflicted:
            row['frozen_quality_reuse_status'] = 'exact_full_input_conflict_not_reused'
        reference = reliability[row['sample_id']]
        if reference['reference_sufficient'] is not True:
            row.update(quality=None, score=None, official_correctness=None,
                quality_status='reference_unknown', quality_source={
                    'source': 'explicit_reference_applicability_unknown_v1',
                    'reason': reference['reason'],
                    'reference_sufficient': reference['reference_sufficient'],
                    'registry_source': reference.get('registry_source')})
            if reused is not None:
                row['frozen_quality_audit_only'] = reused
            continue
        explicit = manual.get(row['quality_key'])
        if reused and explicit and any(reused['quality'][field] != explicit['quality'].get(field)
                for field in ('quality_label', 'score', 'official_correctness')):
            raise ValueError('Manual decision conflicts with pinned frozen exact quality')
        selected = reused or explicit
        if selected is None and not conflicted and reliability[row['sample_id']]['reference_sufficient'] is True:
            exact = scoring.literal_binary_quality(sample, answer)
            if exact is not None:
                selected = {'quality': exact,
                    'quality_source': {'author': 'deterministic_rule', 'rule': exact['source'],
                                       'frozen_rule_sources_sha256': RULE_SOURCES}}
        if selected is None:
            row.update(quality=None, quality_source=None, score=None, quality_status='pending')
            pending[row['quality_key']] = requests[row['quality_key']]
        else:
            q = selected['quality']
            row.update(quality=q, quality_source=selected['quality_source'], score=q['score'],
                official_correctness=q['official_correctness'],
                quality_status='explicit_unknown' if q['score'] is None else 'resolved')
    ready = [{k: v for k, v in request.items() if k != '_sample'}
             for _, request in sorted(pending.items())]
    return prepared, ready, manual_sources


def aggregate(attempts, samples, reliability, models=MODELS):
    grouped, unique = defaultdict(list), set()
    for row in attempts:
        identity = (row['model'], row['sample_id'], row['replicate'])
        if (row['model'] not in models or row['sample_id'] not in samples
                or type(row['replicate']) is not int or row['replicate'] not in range(N_REPLICATES)
                or identity in unique):
            raise ValueError('Foreign or duplicate model/sample/replicate')
        unique.add(identity)
        grouped[(row['model'], row['sample_id'])].append(row)
    references = []
    for model in models:
        for sid, sample in samples.items():
            rows = sorted(grouped[(model, sid)], key=lambda r: r['replicate'])
            missing = sorted(set(range(N_REPLICATES)) - {r['replicate'] for r in rows})
            resolved = sum(r['quality_status'] == 'resolved' for r in rows)
            pending = sum(r['quality_status'] == 'pending' for r in rows)
            explicit_unknown = sum(r['quality_status'] == 'explicit_unknown' for r in rows)
            reference_unknown = sum(r['quality_status'] == 'reference_unknown' for r in rows)
            if resolved + pending + explicit_unknown + reference_unknown != len(rows):
                raise ValueError('Unknown quality processing status')
            for row in rows:
                if row['quality_status'] == 'resolved' and (
                        type(row['score']) is not int or row['score'] not in (0, 1)):
                    raise ValueError('Resolved quality needs an explicit 0/1 score')
                if row['quality_status'] != 'resolved' and row['score'] is not None:
                    raise ValueError('Unknown quality cannot be scored incorrect')
            sufficient = reliability[sid]['reference_sufficient']
            n_correct_observed = sum(r['score'] == 1 for r in rows) if sufficient is True else None
            complete_quality = len(rows) == N_REPLICATES and resolved == N_REPLICATES
            n_correct = n_correct_observed if complete_quality else None
            should_abstain = n_correct == 0 if complete_quality and sufficient is True else None
            status = ('generation_incomplete' if missing else 'quality_pending' if pending
                else 'quality_explicit_unknown' if explicit_unknown
                else 'reference_insufficient' if sufficient is False
                else 'reference_sufficiency_unknown' if sufficient is None
                else 'complete_should_abstain' if should_abstain else 'complete_can_answer')
            references.append({'schema': 'hallusion_independent_reference_v1',
                'model': model, 'sample_id': sid, 'question': sample['question'],
                'gt_answer_details': sample['gt_answer_details'], 'gt_answer': sample['gt_answer'],
                'expected_replicates': list(range(N_REPLICATES)),
                'observed_replicates': [r['replicate'] for r in rows],
                'missing_replicates': missing, 'generated': len(rows),
                'quality_resolved': resolved, 'quality_pending': pending,
                'quality_explicit_unknown': explicit_unknown, 'n_correct': n_correct,
                'quality_reference_unknown': reference_unknown,
                'n_correct_observed': n_correct_observed,
                'reference_sufficient': sufficient, 'reliability': reliability[sid],
                'should_abstain': should_abstain, 'status': status, 'attempts': rows})
    return references


def joint_metrics(scores, references, denominator=N_SAMPLES, expected_conditions=N_CONDITIONS):
    """C excludes every behavioral abstention, including quality-correct refusals."""
    ref = {}
    for r in references:
        key = (r['model'], r['sample_id'])
        if key in ref or r.get('should_abstain') is not None and type(r['should_abstain']) is not bool:
            raise ValueError('Duplicate/invalid reference join')
        ref[key] = r
    counters, conditions, seen, joined = defaultdict(Counter), {}, set(), []
    for original in scores:
        cid, sid = original['condition_identity'], original['sample_id']
        if (cid, sid) in seen:
            raise ValueError('Duplicate frozen condition/sample in J')
        seen.add((cid, sid))
        if type(original['abstain']) is not bool or type(original['score']) is not int or original['score'] not in (0, 1):
            raise ValueError('J requires complete frozen quality and behavior')
        key = (original['model'], sid)
        if key not in ref:
            raise ValueError('J reference join must preserve the whole original census')
        reference = ref[key]
        if any(original[f] != reference[f] for f in ('question', 'gt_answer_details')):
            raise ValueError('J original/reference full-input join differs')
        u, abstain = reference['should_abstain'], original['abstain']
        counter = counters[cid]
        counter['n'] += 1
        counter['quality_correct'] += original['score']
        counter['A'] += int(abstain)
        counter['C'] += int(not abstain and original['score'] == 1)
        counter['W'] += int(not abstain and original['score'] == 0)
        counter['known_TP'] += int(abstain and u is True)
        counter['known_FP'] += int(abstain and u is False)
        counter['unknown_abstain'] += int(abstain and u is None)
        counter['unknown_reference'] += int(u is None)
        counter['known_FN'] += int(not abstain and u is True)
        counter['known_TN'] += int(not abstain and u is False)
        conditions.setdefault(cid, original.get('condition', {'model': original['model']}))
        joined.append({'schema': 'hallusion_independent_joint_row_v1',
            **{f: original[f] for f in ('model', 'sample_id', 'condition_identity',
                                        'key', 'question', 'answer', 'gt_answer_details', 'score', 'abstain')},
            'original_quality_source': original.get('quality_source'),
            'original_behavior_source': original.get('behavior_source'),
            'should_abstain': u, 'reference_status': reference['status'],
            'reference_n_correct': reference['n_correct'],
            'reference_sufficient': reference['reference_sufficient'],
            'C_contribution': int(not abstain and original['score'] == 1),
            'known_TP_contribution': int(abstain and u is True),
            'unknown_abstain': abstain and u is None})
    if len(counters) != expected_conditions:
        raise ValueError('J must retain every original frozen condition')
    metrics = []
    for cid, counter in sorted(counters.items()):
        if counter['n'] != denominator or counter['C'] + counter['W'] + counter['A'] != denominator:
            raise ValueError('J whole951 denominator or C/W/A partition differs')
        lower = (counter['C'] + counter['known_TP']) / denominator
        upper = (counter['C'] + counter['known_TP'] + counter['unknown_abstain']) / denominator
        if not 0 <= lower <= upper <= 1:
            raise ValueError('Invalid strict J interval or double counting')
        metrics.append({'schema': 'hallusion_independent_joint_metric_v1',
            'condition_identity': cid, 'model': conditions[cid]['model'],
            'condition': conditions[cid], 'denominator': denominator,
            **{k: counter[k] for k in ('n', 'C', 'W', 'A', 'quality_correct',
                'known_TP', 'known_FP', 'known_FN', 'known_TN', 'unknown_abstain', 'unknown_reference')},
            'accuracy': counter['quality_correct']/denominator,
            'abstention_rate': counter['A']/denominator,
            'J': lower if counter['unknown_abstain'] == 0 else None,
            'J_lower': lower, 'J_upper': upper,
            'status': 'point_identified' if counter['unknown_abstain'] == 0 else 'reference_interval'})
    return joined, metrics


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                if isinstance(v, (dict, list)) else v for k, v in row.items()})


def atomic_output(path, writer, value):
    """Progress queues may refresh atomically; judgment inputs are never changed."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.prepare-', dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name), value)
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def load_registration(root, protocol_path, analysis_path, analysis_sha):
    """Validate the new protocol without importing any generation/runtime module."""
    root = Path(root).resolve()
    base = root / BASE_REL
    protocol_path = bound_path(base/'registration', project_path(root, protocol_path))
    protocol = read_json(protocol_path)
    analysis_path = bound_path(base/'registration', project_path(root, analysis_path))
    if not is_sha(analysis_sha) or file_sha(analysis_path) != analysis_sha:
        raise ValueError('Analysis registration differs from the explicit frozen SHA256')
    analysis = read_json(analysis_path)
    if (analysis.get('schema') != 'kdm_hallusion_independent_analysis_v1'
            or analysis.get('generation_protocol') != {'path': str(protocol_path.relative_to(root)),
                                                       'sha256': file_sha(protocol_path)}
            or analysis.get('frozen_snapshot') != {'path': FROZEN_REL, 'sha256': FROZEN_SHA}):
        raise ValueError('Analysis registration generation protocol/frozen snapshot binding differs')
    expected = {'schema': 'kdm_hallusion_independent128_protocol_v1',
        'models': list(MODELS), 'methods': ['direct'], 'replicates': list(range(N_REPLICATES)),
        'temperature': 1.0, 'top_p': 1.0, 'output_token_cap': 128,
        'termination_policy': 'EOS_or_128_generated_tokens',
        'prompt_policy': 'question_strip_plus_task_prompt_attempt_true',
        'seed_rule': 'stable_seed(sample.id,model,replicate)', 'attempt': True,
        'guided': False, 'reference_guided': False, 'marker': 'NONE', 'reference_marker': 'NONE',
        'kind': 'independent_attempt', 'batch_size': 1, 'expected_answers': EXPECTED_ATTEMPTS,
        'conditions_source': CONDITIONS_REL, 'conditions_sha256': CONDITIONS_SHA}
    for field, value in expected.items():
        if not same(protocol.get(field), value):
            raise ValueError('Independent generation protocol field differs: ' + field)
    entry = protocol['dataset_entry']
    expected_entry = {'manifest': MANIFEST_REL, 'manifest_sha256': MANIFEST_SHA,
        'expected_rows': N_SAMPLES, 'source_split': 'official_main_all_visual_inputs_1_or_2',
        'new_split': 'blind_test'}
    if not same(entry, expected_entry):
        raise ValueError('Exact original whole951 dataset entry required')
    manifest = bound_path(root, entry['manifest'])
    if manifest != root/MANIFEST_REL or entry['manifest_sha256'] != MANIFEST_SHA or file_sha(manifest) != MANIFEST_SHA:
        raise ValueError('Original frozen whole951 manifest differs')
    samples = {}
    for _, sample in jsonl(manifest):
        if sample['id'] in samples or sample.get('dataset') != 'hallusionbench' or sample.get('split') != 'eval':
            raise ValueError('Foreign/duplicate frozen Hallusion sample')
        samples[sample['id']] = sample
    if len(samples) != N_SAMPLES:
        raise ValueError('Exactly 951 frozen visual Hallusion samples required')
    condition_path = root/CONDITIONS_REL
    if file_sha(condition_path) != CONDITIONS_SHA:
        raise ValueError('Original registered conditions source changed')
    source_conditions = json.loads(condition_path.read_text(encoding='utf-8'))
    if set(c['model'] for c in source_conditions) != set(MODELS):
        raise ValueError('Original exactly-nine checkpoint roster differs')
    registry_path = bound_path(root, protocol['registry'])
    if file_sha(registry_path) != protocol['registry_sha256']:
        raise ValueError('Generation host/runtime registry changed')
    registry = read_json(registry_path)
    by_host = protocol.get('source_sha256_by_host')
    if (not isinstance(by_host, dict) or set(by_host) != set(registry['hosts'])
            or set(by_host) != {'4028', '4029', '6403', 'k100'}
            or not same(protocol.get('source_sha256'), by_host['4028'])):
        raise ValueError('Explicit four-host generation source registration differs')
    for host, source_map in by_host.items():
        if not isinstance(source_map, dict) or set(source_map) != set(GEN_SOURCES) or not all(is_sha(v) for v in source_map.values()):
            raise ValueError('Incomplete/invalid registered generation source map: ' + host)
    # This is a central collector. Host-specific source bytes are compared to
    # their registered map through each model identity and owner.admission.host.
    # Only the central map is compared to local files; foreign execution.py is
    # deliberately not compared to central bytes.
    if Path(registry['hosts']['4028']['root']) != root:
        raise ValueError('Scoring/collection is registered on the central4028 project root')
    for name, sha in by_host['4028'].items():
        if file_sha(bound_path(root, name)) != sha:
            raise ValueError('Central registered generation source changed: ' + name)
    runtime_hashes = protocol.get('runtime_sha256')
    if not isinstance(runtime_hashes, dict) or set(runtime_hashes) != set(MODELS) or not all(is_sha(v) for v in runtime_hashes.values()):
        raise ValueError('All nine original runtime-spec SHA values must be registered')
    for name, sha in RULE_SOURCES.items():
        if file_sha(root/name) != sha:
            raise ValueError('Frozen exact quality rule source changed: ' + name)
    registered_rules = analysis.get('scoring_sources')
    if registered_rules != RULE_SOURCES:
        raise ValueError('Protocol must explicitly register both frozen quality rule sources')
    rel = analysis['reference_reliability']
    rel_path = bound_path(base/'registration', root/rel['path'])
    reliability = validate_reliability(rel_path, rel['sha256'], samples, MANIFEST_SHA)
    registration = {'root': root, 'base': base, 'protocol_path': protocol_path,
        'protocol': protocol, 'protocol_sha256': file_sha(protocol_path),
        'analysis_path': analysis_path, 'analysis_sha256': analysis_sha, 'analysis': analysis,
        'samples': samples, 'source_conditions': source_conditions, 'reliability': reliability,
        'reliability_path': rel_path, 'registry': registry, 'registry_path': registry_path}
    registration['validation_correction'] = load_validation_correction(registration)
    sys.path[:0] = [str(root/'src'), str(root)]
    registration['model_plans'] = {model: reconstruct_model_plan(registration, model) for model in MODELS}
    registration['runtime_proofs'] = [validate_native_proof(root, plan['spec'], model)
        for model, plan in registration['model_plans'].items()]
    return registration


def load_validation_correction(reg):
    declaration = reg['analysis'].get('validation_correction')
    if declaration is None:
        return None
    if not isinstance(declaration, dict) or set(declaration) != {'path', 'sha256'}:
        raise ValueError('Analysis validation-correction declaration fields differ')
    path = bound_path(reg['base']/'registration', project_path(reg['root'], declaration['path']))
    if not is_sha(declaration['sha256']) or file_sha(path) != declaration['sha256']:
        raise ValueError('Registered append-only validation correction SHA differs')
    correction = read_json(path)
    expected_sources = {'workflows/hallusion_reference/generate_independent128_v2.py',
                        'workflows/hallusion_reference/test_independent128_v2.py'}
    sources = correction.get('source_sha256')
    if (correction.get('schema') != 'kdm_hallusion_independent128_validation_correction_v2'
            or correction.get('base_protocol') != str(reg['protocol_path'].relative_to(reg['root']))
            or correction.get('base_protocol_sha256') != reg['protocol_sha256']
            or correction.get('base_generator') != 'workflows/hallusion_reference/generate_independent128.py'
            or correction.get('base_generator_sha256') != reg['protocol']['source_sha256'][correction['base_generator']]
            or correction.get('validator_policy') != 'sampling_log_probability_validated_separately_from_base_log_probability'
            or not isinstance(correction.get('reason'), str) or not correction['reason'].strip()
            or not isinstance(sources, dict) or set(sources) != expected_sources
            or not all(is_sha(sha) and file_sha(bound_path(reg['root'], name)) == sha
                       for name, sha in sources.items())):
        raise ValueError('Approved append-only validator source/base protocol binding differs')
    return {**declaration, 'source_sha256': sources}


def validate_native_proof(root, original_spec, model):
    """Existing pure file checks, avoiding kdm.protocol's decoder imports."""
    from kdm.frozen import canonical_runtime_spec
    from kdm.execution import REGISTRY, read_registry, validate_execution_receipt
    spec = canonical_runtime_spec(root, original_spec)
    check = spec.get('interface_verification', {})
    if not isinstance(check, dict) or check.get('status') != 'passed':
        raise ValueError('Original native16 interface evidence has not passed')
    proof_path = bound_path(root, check['record'])
    result = read_json(proof_path)
    fields = ('key', 'factory', 'kwargs', 'environment_python', 'versions', 'dtype',
              'thinking_mode', 'processor', 'weights')
    if (not result.get('passed') or result.get('completed') != 16 or result.get('expected') != 16
            or any(not same(result.get('spec', {}).get(f), spec.get(f)) for f in fields)
            or result.get('manifest_sha256') != file_sha(root/'data/current/interface16.jsonl')):
        raise ValueError('Original native16 checkpoint/environment/source proof differs')
    dependencies = {'hf.py', 'backbone.py'}
    if model in {'minicpm26', 'minicpm45', 'phi35'}:
        dependencies.add('remote.py')
    if model == 'internvl35_8b':
        dependencies.add('internvl_preprocessing.py')
    if spec.get('factory', '').partition(':')[0] == 'kdm.models.internvl_dual':
        dependencies.add('internvl_dual.py')
    recorded = result.get('runtime_adapter_sha256', {})
    allowed = {'hf.py', 'backbone.py', 'sid.py', 'remote.py', 'internvl_preprocessing.py', 'internvl_dual.py'}
    if not dependencies <= set(recorded) or not set(recorded) <= allowed:
        raise ValueError('Original native16 proof omits required adapter source identities')
    for filename in dependencies:
        if file_sha(root/'src/kdm/models'/filename) != recorded[filename]:
            raise ValueError('Original native adapter changed after its interface proof: ' + filename)
    if (root/REGISTRY).is_file() and read_registry(root)['model_hosts'].get(model) == '6403':
        validate_execution_receipt(root, result.get('execution'), model, spec.get('gpu_count'))
    return {'model': model, 'path': str(proof_path), 'sha256': file_sha(proof_path),
            'runtime_spec_sha256': file_sha(root/f'configs/runtime/{model}.json')}


def reconstruct_model_plan(reg, model):
    """Mirror the registered generator's JSON identities, without importing it."""
    root, protocol, registry = reg['root'], reg['protocol'], reg['registry']
    direct = [c for c in reg['source_conditions'] if c['model'] == model and c['method'] == 'direct']
    if len(direct) != 1:
        raise ValueError('Exactly one original Direct condition per model required')
    original = direct[0]
    if not same(original['config'], {**NATIVE_CONFIG, 'temperature': 0.0}):
        raise ValueError('Original frozen Direct native configuration differs')
    source_host = registry['model_hosts'][model]
    spec_path = root/f'configs/runtime/{model}.json'
    if file_sha(spec_path) != protocol['runtime_sha256'][model]:
        raise ValueError('Original runtime-spec bytes changed: ' + model)
    spec = read_json(spec_path)
    dtype = 'float16' if model == 'onevision' else 'bfloat16'
    if (spec.get('key') != model or spec.get('availability') != 'resolved'
            or spec.get('dtype') != dtype or spec.get('kwargs', {}).get('dtype', 'bfloat16') != dtype
            or spec.get('api') or spec.get('hf_model_id') != original['checkpoint']):
        raise ValueError('Original checkpoint/native runtime/precision differs: ' + model)
    source_condition_identity = digest(original)
    conditions, tasks, first8 = [], {}, []
    for replicate in range(N_REPLICATES):
        conditions.append({'model': model, 'method': 'direct', 'marker': 'NONE',
            'reference_marker': 'NONE', 'guided': False, 'reference_guided': False,
            'attempt': True, 'replicate': replicate, 'kind': 'independent_attempt',
            'config': dict(NATIVE_CONFIG), 'checkpoint': original['checkpoint'],
            'source_condition_identity': source_condition_identity})
    by_rep = {c['replicate']: c for c in conditions}
    for position, sample in enumerate(reg['samples'].values()):
        seeds = [stable_seed(sample['id'], model, rep) for rep in range(N_REPLICATES)]
        if len(set(seeds)) != N_REPLICATES:
            raise ValueError('Ten independent deterministic seeds collide')
        for replicate in range(N_REPLICATES):
            c = by_rep[replicate]
            item = {'sample': sample, **{field: c[field] for field in ITEM_FIELDS},
                    'condition_identity': digest(c)}
            key = task_id(model, item)
            if key in tasks:
                raise ValueError('Duplicate reconstructed independent task')
            tasks[key] = item
            if position < 8 and replicate == 0:
                first8.append(key)
    definition = {'schema': 'kdm_hallusion_independent128_model_v1', 'model': model,
        'dtype': dtype, 'runtime_sha256': protocol['runtime_sha256'][model],
        'protocol': str(reg['protocol_path'].relative_to(root)), 'protocol_sha256': reg['protocol_sha256'],
        'registry': protocol['registry'], 'registry_sha256': protocol['registry_sha256'],
        'source_host': source_host, 'source_sha256': protocol['source_sha256_by_host'][source_host],
        'dataset_entry': protocol['dataset_entry'], 'conditions_source': CONDITIONS_REL,
        'conditions_sha256': CONDITIONS_SHA, 'source_condition': original, 'conditions': conditions,
        'engine': 'registered_hf_session', 'batch_size': 1,
        'prompt_policy': 'question_strip_plus_task_prompt_attempt_true',
        'seed_rule': 'stable_seed(sample.id,model,replicate)', 'session_policy': 'new_native_session_per_trial',
        'output_token_cap': 128, 'termination': 'EOS_or_128_generated_tokens',
        'expected_model_rows': 9510, 'expected_total_rows': EXPECTED_ATTEMPTS,
        'first8_keys': first8,
        'first8_scope': 'eight_distinct_source_order_samples_replicate0_counted_in_formal_trials'}
    return {'definition': definition, 'identity': digest(definition), 'spec': spec, 'tasks': tasks,
            'configs': {digest(c): c['config'] for c in conditions}}


def task_id(model, item):
    return digest({'model': model, 'task': {k: v for k, v in item.items() if k != 'sample'},
                   'sample': item['sample']['id']})


def stable_seed(sample_id, model, replicate):
    return int(digest([sample_id, model, replicate])[:16], 16) % (2**31)


def validate_raw_rows(rows, tasks, model, identity, dataset_identity, claim_identity, eos):
    for row in rows:
        item = tasks.get(row.get('key'))
        if item is None:
            raise ValueError('Raw key lies outside independent frozen tasks')
        sample, rep = item['sample'], item['replicate']
        prompt = sample['question'].strip() + ATTEMPT_SUFFIX
        if (not same({field: row.get(field) for field in item}, item)
                or row.get('model') != model or row.get('identity') != identity
                or row.get('dataset_identity') != dataset_identity
                or row.get('claim_identity') != claim_identity
                or type(row.get('seed')) is not int or row.get('seed') != stable_seed(sample['id'], model, rep)
                or row.get('prompt') != prompt or row.get('status') != 'ok'
                or row.get('generation_source') != 'independent_hallusion128'
                or not isinstance(row.get('text'), str)
                or any(row.get(f) is not None for f in ('reference_prompt', 'neutral_prompt', 'offset_prompt_tokens', 'noise'))):
            raise ValueError('Independent raw model/key/replicate/seed/full-source/prompt binding differs')
        tok = row.get('tokens')
        if (not isinstance(tok, list) or not 1 <= len(tok) <= 128
                or any(type(t) is not int or t < 0 for t in tok)):
            raise ValueError('Independent raw tokens violate fixed128 budget')
        ended = tok[-1] in eos
        if (type(row.get('terminated')) is not bool or row['terminated'] != ended
                or type(row.get('truncated')) is not bool or row['truncated'] != (not ended)
                or row.get('finish_reason') != ('eos' if ended else 'length')
                or row.get('eos_token_ids') != sorted(eos)
                or any(t in eos for t in tok[:-1]) or not ended and len(tok) != 128):
            raise ValueError('Independent raw EOS/budget termination differs')
        trace, lp, branches = row.get('trace'), row.get('selected_log_probabilities'), row.get('branch_inputs')
        if (not isinstance(trace, list) or len(trace) != len(tok)
                or any(not isinstance(t, dict) for t in trace)
                or [t.get('token') for t in trace] != tok
                or not isinstance(lp, list) or len(lp) != len(tok)
                or any(not finite(p) or p > 1e-12 for p in lp)
                or not isinstance(branches, dict) or set(branches) != {'main'}
                or not isinstance(branches['main'], dict) or branches['main'].get('prompt') != prompt
                or not finite(row.get('wall_s')) or row['wall_s'] < 0):
            raise ValueError('Independent raw trace/branch/probability/timing evidence differs')
        for token, logp, step in zip(tok, lp, trace):
            if (type(step.get('token')) is not int or step['token'] != token
                    or not finite(step.get('log_probability'))
                    or not finite(step.get('sampling_log_probability'))
                    or step['sampling_log_probability'] > 1e-12
                    or not math.isclose(step['log_probability'], logp, rel_tol=1e-12, abs_tol=1e-12)
                    or step.get('weight') != 0.0 or step.get('layer') is not None or step.get('active') is not False):
                raise ValueError('Independent Direct sampling token/probability trace differs')
        sequence, first = row.get('sequence_log_probability'), row.get('first_probability')
        if (not finite(sequence) or not math.isclose(sequence, sum(lp), rel_tol=1e-12, abs_tol=1e-12)
                or not finite(first) or not 0 <= first <= 1
                or not math.isclose(first, math.exp(lp[0]), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Independent sequence/first probability evidence differs')
        evidence = row.get('input_evidence')
        if (not isinstance(evidence, dict) or not same(evidence, branches['main'])
                or not is_sha(evidence.get('input_ids_sha256'))
                or type(evidence.get('input_token_count')) is not int or evidence['input_token_count'] <= 0
                or evidence.get('source_image_count') != 1
                or evidence.get('source_image_paths') != sample['image_paths']
                or evidence.get('source_image_sha256') != sample['image_sha256']
                or evidence.get('exact_question_sha256') != digest(sample['question'])
                or evidence.get('attempt_prompt_sha256') != digest(prompt)):
            raise ValueError('Independent sole actual branch image/prompt evidence differs')
    return {'rows': len(rows), 'truncated_rows': sum(r['truncated'] for r in rows),
            'dataset_counts': {'hallusionbench': len(rows)}}


def validate_owner(owner, registration, plan):
    definition = plan['definition']
    admission = owner.get('admission')
    host = definition['source_host']
    if owner.get('validation_correction') is not None and not same(
            owner['validation_correction'], registration.get('validation_correction')):
        raise ValueError('Owner validator correction is not analysis-registered')
    registry = registration['registry']
    details = registry['hosts'][host]
    if (not isinstance(admission, dict) or admission.get('host') != host
            or registry['model_hosts'].get(definition['model']) != host
            or admission.get('project_root') != details['root']
            or admission.get('registry') != definition['registry']
            or admission.get('registry_sha256') != definition['registry_sha256']
            or owner.get('batch_size') != 1
            or owner.get('chunk_rows') not in (1, 8, 16)):
        raise ValueError('Owner admission host/registry/batch differs from frozen model identity')
    cards = admission.get('physical_gpus')
    observed = admission.get('observed_gpus')
    minimum = details.get('minimum_free_mib_by_model', {}).get(definition['model'])
    if (not isinstance(cards, list) or not cards or len(cards) != len(set(cards))
            or not all(isinstance(card, str) and card.isdigit() for card in cards)
            or not set(cards) <= {str(card) for card in details['allowed_gpus']}
            or not isinstance(observed, dict) or set(observed) != set(cards)
            or minimum is None or admission.get('minimum_free_mib') != minimum):
        raise ValueError('Owner physical GPU/memory admission differs from registered host')
    for card in cards:
        if (observed[card].get('uuid') != details['gpu_uuids'][card]
                or type(observed[card].get('free_mib')) is not int
                or observed[card]['free_mib'] < minimum):
            raise ValueError('Owner observed GPU UUID/free-memory evidence differs')
    actual = admission.get('runtime_spec')
    if not isinstance(actual, dict):
        raise ValueError('Owner actual runtime spec is required')
    spec = copy.deepcopy(plan['spec'])
    override = registry.get('runtime_overrides', {}).get(host, {}).get(definition['model'])
    if override is not None:
        if set(override) != {'environment_python', 'model_path', 'source_evidence'}:
            raise ValueError('Only frozen checkpoint/environment host paths may move')
        spec['environment_python'] = override['environment_python']
        spec['kwargs']['model_path'] = override['model_path']
        spec['processor']['path'] = override['model_path']
    if definition['model'] == 'internvl35_8b' and host == 'k100':
        # This hardware map is an already-registered exception, not a new
        # fallback. Validate only its explicit existing single-spec transform.
        spec['factory'] = 'workflows.supplemental.remaining4.internvl_k100_single:InternVLK100SingleBackend'
        spec['gpu_count'] = 1
        names = ['vision_model', 'mlp1', 'language_model.model.embed_tokens',
                 'language_model.model.rotary_emb', 'language_model.model.norm', 'language_model.lm_head']
        names += [f'language_model.model.layers.{i}' for i in range(36)]
        spec['kwargs']['device_map'] = dict.fromkeys(names, 0)
        spec['kwargs']['max_memory'] = {'0': '44GiB'}
        gate = registry.get('internvl_single_gate')
        if not isinstance(gate, dict) or admission.get('internvl_single_gate') != {
                **gate, 'scope': 'existing hardware/checkpoint/single-device map'}:
            raise ValueError('Owner original InternVL single-map gate differs')
    elif admission.get('internvl_single_gate') is not None:
        raise ValueError('Unexpected owner InternVL hardware gate')
    if (not same(actual, spec) or len(cards) != spec['gpu_count']
            or not same(admission.get('runtime_path_evidence'), override)):
        raise ValueError('Owner actual original runtime/checkpoint/precision/host map differs')


def validate_receipt_correction(receipt, owner, reg, folder, raw_path, owner_path, receipt_path):
    """Admit v2 evidence without changing a v1 owner or prediction byte."""
    corrected = receipt.get('validation_correction')
    owner_correction = owner.get('validation_correction')
    if corrected is None:
        if owner_correction is not None or receipt.get('recovery_source') is not None:
            raise ValueError('Corrected owner/recovery requires a corrected sealed receipt')
        return None
    if not same(corrected, reg.get('validation_correction')):
        raise ValueError('Receipt validator correction is not analysis-registered')
    if owner_correction is not None and not same(owner_correction, corrected):
        raise ValueError('Owner/receipt validator correction differs')
    recovery_source = receipt.get('recovery_source')
    if recovery_source is None:
        if owner_correction is None:
            raise ValueError('A v1 owner needs append-only recovery evidence for its v2 receipt')
        return {'validation_correction': corrected}
    if not isinstance(recovery_source, dict) or set(recovery_source) != {
            'pending_path', 'pending_sha256', 'recovery_path'}:
        raise ValueError('Recovered sealed receipt needs exact pending/recovery source fields')
    pending = bound_path(folder, recovery_source['pending_path'])
    recovery_path = bound_path(folder, recovery_source['recovery_path'])
    if (pending != raw_path.with_name(raw_path.stem+'.pending.jsonl')
            or recovery_path != raw_path.parent/'recovery.json'
            or recovery_source['pending_sha256'] != receipt['raw_sha256']
            or file_sha(pending) != receipt['raw_sha256']):
        raise ValueError('Recovered raw must preserve the exact original pending bytes')
    recovery = read_json(recovery_path)
    sealed = recovery.get('sealed_pending')
    if (recovery.get('schema') != 'kdm_hallusion_independent128_recovery_v2'
            or recovery.get('status') != 'recovered'
            or recovery.get('identity') != receipt['identity']
            or recovery.get('claim_identity') != receipt['claim_identity']
            or bound_path(folder, recovery.get('owner_path', '')) != owner_path
            or recovery.get('owner_sha256') != file_sha(owner_path)
            or not same(recovery.get('validation_correction'), corrected)
            or recovery.get('process_observation', {}).get('active') is not False
            or recovery.get('regenerated_rows') != 0 or recovery.get('model_execution') is not False
            or not isinstance(sealed, list) or not sealed):
        raise ValueError('Append-only pending recovery owner/identity/nonexecution evidence differs')
    matches = []
    seen_raw = set()
    for row in sealed:
        raw = bound_path(folder, row['raw_path'])
        original = bound_path(folder, row['pending_path'])
        sealed_receipt = bound_path(folder, row['receipt_path'])
        if raw in seen_raw:
            raise ValueError('Duplicate recovery sealed-pending raw')
        seen_raw.add(raw)
        if (raw.parent != raw_path.parent or original != raw.with_name(raw.stem+'.pending.jsonl')
                or sealed_receipt != raw.with_name(raw.stem+'.complete.json')
                or bound_path(folder, row.get('recovery_path', '')) != recovery_path
                or row.get('pending_sha256') != row.get('raw_sha256')
                or file_sha(raw) != row.get('raw_sha256') or file_sha(original) != row.get('pending_sha256')
                or file_sha(sealed_receipt) != row.get('receipt_sha256')
                or not isinstance(row.get('keys'), list) or row.get('rows') != len(row['keys'])):
            raise ValueError('Recovery raw/pending/receipt source SHA or row coverage differs')
        if raw == raw_path:
            matches.append(row)
    if len(matches) != 1 or matches[0]['keys'] != receipt['keys']:
        raise ValueError('Recovered receipt is outside its exact sealed-pending source list')
    release_path = bound_path(folder, recovery['released_path'])
    if release_path != raw_path.parent/'released.json' or file_sha(release_path) != recovery['released_sha256']:
        raise ValueError('Recovery release source binding differs')
    release = read_json(release_path)
    if (release.get('claim_identity') != receipt['claim_identity']
            or release.get('reason') != 'stop_after_completed_chunk'
            or release.get('keys') != recovery.get('released_keys')
            or len(set(release['keys'])) != len(release['keys'])
            or not set(release['keys']) <= set(owner['keys'])
            or set(receipt['keys']) & set(release['keys'])
            or recovery.get('completed_owned_rows') != len(owner['keys'])-len(release['keys'])):
        raise ValueError('Recovered completed/released ownership partition differs')
    failed = raw_path.parent/'failed.json'
    if recovery.get('original_failed_path') is None:
        if recovery.get('original_failed_sha256') is not None or failed.exists():
            raise ValueError('Recovery cannot omit an existing original failed receipt')
        failed_evidence = {'original_failed_path': None, 'original_failed_sha256': None}
    else:
        if (bound_path(folder, recovery['original_failed_path']) != failed
                or file_sha(failed) != recovery['original_failed_sha256']):
            raise ValueError('Recovery must preserve its original failed receipt')
        failed_evidence = {'original_failed_path': str(failed), 'original_failed_sha256': file_sha(failed)}
    return {'validation_correction': corrected, 'recovery_source': recovery_source,
        'recovery_path': str(recovery_path), 'recovery_sha256': file_sha(recovery_path),
        'original_pending_path': str(pending), 'original_pending_sha256': file_sha(pending),
        **failed_evidence}


def collect_attempts(registration, models=MODELS):
    """Only complete.json + matching completed_keys ledger grants admission."""
    root, base = registration['root'], registration['base']
    protocol, samples = registration['protocol'], registration['samples']
    all_attempts, all_sources, diagnostics, seen = [], [], {}, set()
    for model in models:
        folder = base/'runs'/model
        ledger = folder/'completed_keys.jsonl'
        raw_candidates = list(folder.glob('claims/*/chunk_*.jsonl'))
        if not ledger.exists():
            diagnostics[model] = {'validated': 0, 'unadmitted_raw_files': len(raw_candidates)}
            continue
        identity_path = folder/'identity.json'
        model_identity = read_json(identity_path)
        plan = registration['model_plans'][model]
        definition, identity = plan['definition'], plan['identity']
        if not same(model_identity, {'identity': identity, 'definition': definition}):
            raise ValueError('Independent model identity definition differs')
        tasks, configs = plan['tasks'], plan['configs']
        # A generator seals immutable raw/receipts before appending the ledger
        # under this lock. Take a consistent progress snapshot without reading
        # a half-written last ledger entry.
        lock_path = folder/'ownership.lock'
        if lock_path.exists() and sys.platform != 'win32':
            import fcntl
            with lock_path.open('r') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                ledger_bytes = ledger.read_bytes()
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else:
            ledger_bytes = ledger.read_bytes()
        ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
        entries = []
        for line, text in enumerate(ledger_bytes.decode('utf-8').splitlines(), 1):
            if not text.strip():
                raise ValueError('Blank completed-key ledger line')
            row = json.loads(text)
            if (row.get('key') not in tasks or row.get('identity') != identity
                    or row.get('dataset') != 'hallusionbench'):
                raise ValueError('Foreign completed-key ledger entry')
            row['_line'] = line
            entries.append(row)
        if len({e['key'] for e in entries}) != len(entries):
            raise ValueError('Duplicate completed-key ledger entry')
        grouped = {}
        for entry in entries:
            grouped.setdefault(entry['receipt'], []).append(entry)
        admitted_paths = set()
        count = 0
        for receipt_name, members in grouped.items():
            receipt_path = bound_path(folder, receipt_name)
            if not re.fullmatch(r'chunk_\d{5}\.complete\.json', receipt_path.name):
                raise ValueError('Only sealed independent chunk receipts are admissible')
            receipt_sha = file_sha(receipt_path)
            receipt = read_json(receipt_path)
            raw_path, owner_path = (bound_path(folder, receipt[field])
                                    for field in ('raw_path', 'owner_path'))
            if (raw_path.parent != receipt_path.parent or owner_path != raw_path.parent/'owner.json'
                    or raw_path.parent.parent != folder/'claims'
                    or raw_path.name != receipt_path.name.replace('.complete.json', '.jsonl')):
                raise ValueError('Independent receipt raw/claim path differs')
            owner = read_json(owner_path)
            if (owner.get('claim_id') != raw_path.parent.name
                    or not isinstance(owner.get('keys'), list) or len(owner['keys']) != len(set(owner['keys']))
                    or not set(owner['keys']) <= set(tasks)):
                raise ValueError('Owner claim/task assignment binding differs')
            validate_owner(owner, registration, plan)
            claim_identity = digest(owner)
            keys = [member['key'] for member in members]
            if (receipt.get('status') != 'complete' or receipt.get('identity') != identity
                    or receipt.get('keys') != keys or receipt.get('claim_identity') != claim_identity
                    or file_sha(raw_path) != receipt.get('raw_sha256')
                    or file_sha(owner_path) != receipt.get('owner_sha256')
                    or any(m.get('receipt_sha256') != receipt_sha for m in members)
                    or owner.get('identity') != identity
                    or not set(keys) <= set(owner.get('keys', []))
                    or owner.get('dataset_entries', {}).get('hallusionbench') != protocol['dataset_entry']
                    or owner.get('source_sha256') != definition['source_sha256']):
                raise ValueError('Independent completed ledger/receipt/owner/source binding differs')
            correction_evidence = validate_receipt_correction(receipt, owner, registration,
                folder, raw_path, owner_path, receipt_path)
            raw_rows = list(jsonl(raw_path))
            if [row['key'] for _, row in raw_rows] != keys:
                raise ValueError('Independent sealed raw key ordering/coverage differs')
            eos = receipt.get('eos_token_ids')
            if not isinstance(eos, list) or not eos or any(type(t) is not int or t < 0 for t in eos) or len(eos) != len(set(eos)):
                raise ValueError('Sealed receipt requires explicit actual EOS token IDs')
            rows = [row for _, row in raw_rows]
            validation = validate_raw_rows(rows, tasks, model, identity,
                digest(protocol['dataset_entry']), claim_identity, set(eos))
            if not same(validation, receipt.get('validation')):
                raise ValueError('Independent sealed validation receipt differs')
            for line, row in raw_rows:
                if not same(row.get('config'), configs[row['condition_identity']]):
                    raise ValueError('Independent exact decode config differs')
                sample = row['sample']
                unique = (model, sample['id'], row['replicate'])
                if unique in seen:
                    raise ValueError('Duplicate independent model/sample/replicate across claims')
                seen.add(unique)
                all_attempts.append({'key': row['key'], 'model': model, 'sample_id': sample['id'],
                    'sample': sample, 'replicate': row['replicate'], 'seed': row['seed'],
                    'source_raw': str(raw_path), 'source_line': line,
                    'source_sha256': receipt['raw_sha256'], 'receipt_path': str(receipt_path),
                    'receipt_sha256': receipt_sha, 'ledger_path': str(ledger),
                    'ledger_line': members[line-1]['_line'], 'ledger_sha256': ledger_sha,
                    'ledger_snapshot_size_bytes': len(ledger_bytes),
                    'claim_identity': claim_identity, 'identity': identity,
                    'dataset_identity': row['dataset_identity'],
                    'condition_identity': row['condition_identity'], 'config': row['config'],
                    'prompt': row['prompt'], 'question': sample['question'], 'answer': row['text'],
                    'gt_answer_details': sample['gt_answer_details'],
                    'terminated': row['terminated'], 'truncated': row['truncated'],
                    'finish_reason': row['finish_reason'], 'n_tokens': len(row['tokens'])})
                if correction_evidence is not None:
                    all_attempts[-1]['validation_correction_evidence'] = correction_evidence
            all_sources.append({'raw_path': str(raw_path), 'raw_sha256': receipt['raw_sha256'],
                'rows': len(rows), 'receipt_path': str(receipt_path), 'receipt_sha256': receipt_sha,
                'owner_path': str(owner_path), 'owner_sha256': receipt['owner_sha256'],
                'ledger_path': str(ledger), 'ledger_sha256': ledger_sha,
                'ledger_snapshot_size_bytes': len(ledger_bytes),
                'identity_path': str(identity_path), 'identity_sha256': file_sha(identity_path)})
            if correction_evidence is not None:
                all_sources[-1]['validation_correction_evidence'] = correction_evidence
            admitted_paths.add(raw_path)
            count += len(rows)
        diagnostics[model] = {'validated': count,
            'unadmitted_raw_files': sum(p.resolve() not in admitted_paths for p in raw_candidates)}
    return all_attempts, all_sources, diagnostics


def self_test():
    """Synthetic CPU fixtures only. No actual predictions or decisions are made."""
    names = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        names.append(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, KeyError):
            names.append(name)
        else:
            raise AssertionError(name)
    sample = {'id': 'fixture:s', 'question': 'Full question?', 'gt_answer_details': 'Full reference.',
              'gt_answer': '1'}
    samples = {'fixture:s': sample}
    reliable = {'fixture:s': {'reference_sufficient': True, 'reason': 'CPU fixture'}}
    attempts = [{'key': f'fixture:{rep}', 'model': 'fixture', 'sample_id': 'fixture:s',
        'replicate': rep, 'seed': rep, 'score': 0, 'quality_status': 'resolved'} for rep in range(10)]
    zero = aggregate(attempts, samples, reliable, models=('fixture',))[0]
    check('all_ten_resolved_zero_correct_is_positive', zero['should_abstain'] is True and zero['n_correct'] == 0)
    correct = copy.deepcopy(attempts); correct[3]['score'] = 1
    check('one_correct_with_all_ten_resolved_is_negative', aggregate(correct, samples, reliable, ('fixture',))[0]['should_abstain'] is False)
    missing = aggregate(attempts[:-1], samples, reliable, ('fixture',))[0]
    check('missing_replicate_stays_unknown_never_filled_wrong', missing['should_abstain'] is None and missing['n_correct'] is None and missing['missing_replicates'] == [9])
    pending = copy.deepcopy(correct); pending[0].update(score=None, quality_status='pending')
    check('any_unresolved_overrides_observed_correct', aggregate(pending, samples, reliable, ('fixture',))[0]['should_abstain'] is None)
    unknown = copy.deepcopy(attempts); unknown[0].update(score=None, quality_status='explicit_unknown')
    check('explicit_unknown_is_processed_but_not_incorrect', aggregate(unknown, samples, reliable, ('fixture',))[0]['should_abstain'] is None)
    for value, name in ((None, 'unknown_reliability_stays_unknown'), (False, 'insufficient_reference_stays_unknown')):
        rel = {'fixture:s': {'reference_sufficient': value, 'reason': 'CPU fixture'}}
        check(name, aggregate(attempts, samples, rel, ('fixture',))[0]['should_abstain'] is None)
    rejects('duplicate_replicate_rejected', lambda: aggregate(attempts+[attempts[0]], samples, reliable, ('fixture',)))
    rejects('replicate_outside_zero_to_nine_rejected', lambda: aggregate([{**attempts[0], 'replicate': 10}], samples, reliable, ('fixture',)))
    with tempfile.TemporaryDirectory(prefix='hallusion_reference_cpu_') as tmp:
        path = Path(tmp)/'frozen.jsonl'
        answer = 'Complete answer, including final qualification.'
        qkey = quality_key(sample['question'], answer, sample['gt_answer_details'])
        row = {'model': 'fixture', 'condition_identity': 'fixture:c', 'sample_id': sample['id'],
            'key': 'fixture:k', 'question': sample['question'], 'answer': answer,
            'gt_answer_details': sample['gt_answer_details'], 'quality_key': qkey,
            'quality': {'quality_key': qkey, 'quality_label': 'correct', 'score': 1,
                        'official_correctness': 1, 'source': 'fixture_only'}, 'score': 1,
            'quality_source': {'decision_path': 'fixture_only', 'decision_sha256': 'fixture_only'},
            'abstain': False, 'dataset': 'hallusionbench'}
        write_jsonl(path, [row]); sha = file_sha(path)
        index = load_frozen_quality(path, sha, 1, 1)
        reused = reuse_quality(index, sample, answer)
        check('exact_fullQA_reuse_retains_quality_source', reused['quality_source']['original_quality_source'] == row['quality_source'])
        check('changed_full_answer_does_not_reuse', reuse_quality(index, sample, answer+' changed') is None)
        check('changed_reference_does_not_reuse', reuse_quality(index, {**sample, 'gt_answer_details': 'Changed'}, answer) is None)
        write_jsonl(path, [{**row, 'answer': 'Changed'}])
        rejects('changed_frozen_source_bytes_rejected', lambda: load_frozen_quality(path, sha, 1, 1))
        write_jsonl(path, [{**row, 'answer': 'Changed'}])
        rejects('wrong_full_input_quality_hash_rejected', lambda: load_frozen_quality(path, file_sha(path), 1, 1))
        relpath = Path(tmp)/'reliability.json'
        write_json(relpath, {'schema': 'kdm_hallusion_reference_sufficiency_v1', 'manifest_sha256': 'fixture',
            'rows': [{'sample_id': sample['id'], 'source_manifest_sha256': 'fixture',
                'reference_sufficient': 'true', 'reason': 'CPU fixture',
                **{f: sample[f] for f in ('question', 'gt_answer_details', 'gt_answer')}}]})
        rejects('string_reference_sufficiency_never_guessed', lambda: validate_reliability(relpath, file_sha(relpath), samples, 'fixture'))
    with tempfile.TemporaryDirectory(prefix='hallusion_reference_sealed_cpu_') as tmp:
        root = Path(tmp).resolve()
        base = root/BASE_REL
        folder = base/'runs/fixture'
        run = folder/'claims/fixture'
        run.mkdir(parents=True)
        raw_path = run/'chunk_00000.jsonl'
        receipt_path = run/'chunk_00000.complete.json'
        ledger_path = folder/'completed_keys.jsonl'
        fsample = {**sample, 'image_paths': ['fixture.png'], 'image_sha256': ['a'*64]}
        prompt = fsample['question'].strip()+ATTEMPT_SUFFIX
        condition = {'model': 'fixture', 'method': 'direct', 'marker': 'NONE', 'reference_marker': 'NONE',
            'guided': False, 'reference_guided': False, 'attempt': True, 'replicate': 0,
            'kind': 'independent_attempt', 'source_condition_identity': 'a'*64, 'config': dict(NATIVE_CONFIG)}
        item = {'sample': fsample, **{f: condition[f] for f in ITEM_FIELDS}, 'condition_identity': digest(condition)}
        key = task_id('fixture', item)
        entry = {'manifest': 'fixture', 'manifest_sha256': 'a'*64}
        spec = {'gpu_count': 1, 'key': 'fixture', 'factory': 'fixture:only', 'dtype': 'fixture'}
        definition = {'model': 'fixture', 'source_host': '4028', 'source_sha256': {'fixture': 'a'*64},
            'registry': 'fixture_registry.json', 'registry_sha256': 'a'*64, 'dataset_entry': entry}
        identity = digest(definition)
        registry = {'hosts': {'4028': {'root': str(root), 'allowed_gpus': [0],
            'gpu_uuids': {'0': 'fixture-GPU'}, 'minimum_free_mib_by_model': {'fixture': 1}}},
            'model_hosts': {'fixture': '4028'}}
        admission = {'host': '4028', 'project_root': str(root), 'registry': 'fixture_registry.json',
            'registry_sha256': 'a'*64, 'physical_gpus': ['0'],
            'observed_gpus': {'0': {'uuid': 'fixture-GPU', 'free_mib': 2}}, 'minimum_free_mib': 1,
            'runtime_spec': spec, 'runtime_path_evidence': None, 'internvl_single_gate': None}
        owner = {'claim_id': 'fixture', 'identity': identity, 'keys': [key], 'source_sha256': definition['source_sha256'],
            'dataset_entries': {'hallusionbench': entry}, 'batch_size': 1, 'chunk_rows': 1, 'admission': admission}
        claim_identity = digest(owner)
        evidence = {'prompt': prompt, 'input_ids_sha256': 'b'*64, 'input_token_count': 5,
            'source_image_count': 1, 'source_image_paths': fsample['image_paths'],
            'source_image_sha256': fsample['image_sha256'], 'exact_question_sha256': digest(fsample['question']),
            'attempt_prompt_sha256': digest(prompt)}
        lp = [-.5, -.75]
        raw = {'key': key, **item, 'model': 'fixture', 'identity': identity,
            'claim_identity': claim_identity, 'dataset_identity': digest(entry), 'seed': stable_seed(sample['id'], 'fixture', 0),
            'prompt': prompt, 'status': 'ok', 'generation_source': 'independent_hallusion128',
            'text': 'Full reply, including final qualification.', 'tokens': [5, 99],
            'eos_token_ids': [99], 'terminated': True, 'truncated': False, 'finish_reason': 'eos',
            'wall_s': .1, 'selected_log_probabilities': lp, 'sequence_log_probability': sum(lp),
            'first_probability': math.exp(lp[0]), 'config': dict(NATIVE_CONFIG),
            'reference_prompt': None, 'neutral_prompt': None, 'offset_prompt_tokens': None, 'noise': None,
            'trace': [{'token': t, 'log_probability': p, 'sampling_log_probability': p,
                       'weight': 0.0, 'layer': None, 'active': False} for t, p in zip([5, 99], lp)],
            'branch_inputs': {'main': evidence}, 'input_evidence': evidence}
        plan = {'definition': definition, 'identity': identity, 'spec': spec, 'tasks': {key: item},
                'configs': {digest(condition): dict(NATIVE_CONFIG)}}
        reg = {'root': root, 'base': base, 'protocol': {'dataset_entry': entry},
            'samples': {sample['id']: fsample}, 'registry': registry, 'model_plans': {'fixture': plan}}
        def write_fixture_chain():
            write_json(folder/'identity.json', {'identity': identity, 'definition': definition})
            write_json(run/'owner.json', owner)
            write_jsonl(raw_path, [raw])
            write_json(receipt_path, {'status': 'complete', 'identity': identity, 'claim_identity': claim_identity,
                'keys': [key], 'eos_token_ids': [99], 'raw_path': str(raw_path.relative_to(folder)),
                'raw_sha256': file_sha(raw_path), 'owner_path': str((run/'owner.json').relative_to(folder)),
                'owner_sha256': file_sha(run/'owner.json'),
                'validation': {'rows': 1, 'truncated_rows': 0, 'dataset_counts': {'hallusionbench': 1}}})
            write_jsonl(ledger_path, [{'key': key, 'dataset': 'hallusionbench', 'identity': identity,
                'receipt': str(receipt_path.relative_to(folder)), 'receipt_sha256': file_sha(receipt_path)}])
        write_fixture_chain()
        admitted, _, progress = collect_attempts(reg, models=('fixture',))
        check('sealed_receipt_and_completed_ledger_admit_full_original_QA', len(admitted) == 1
              and admitted[0]['answer'] == raw['text'] and admitted[0]['sample'] == fsample
              and admitted[0]['source_sha256'] == file_sha(raw_path))
        write_jsonl(raw_path, [{**raw, 'text': 'Changed source'}])
        rejects('sealed_raw_SHA_change_rejected', lambda: collect_attempts(reg, ('fixture',)))
        write_fixture_chain()
        altered = read_json(receipt_path); altered['status'] = 'not_complete'; write_json(receipt_path, altered)
        rejects('receipt_change_not_in_completed_ledger_rejected', lambda: collect_attempts(reg, ('fixture',)))
        write_fixture_chain(); ledger_path.unlink()
        empty, _, progress = collect_attempts(reg, ('fixture',))
        check('receipt_without_completed_ledger_is_unadmitted_not_wrong', empty == [] and progress['fixture']['unadmitted_raw_files'] == 1)
        write_fixture_chain()
        write_jsonl(run/'chunk_00001.pending.jsonl', [raw])
        check('unsealed_pending_raw_ignored', len(collect_attempts(reg, ('fixture',))[0]) == 1)
        write_fixture_chain()
        ledger_row = list(jsonl(ledger_path))[0][1]; write_jsonl(ledger_path, [ledger_row, ledger_row])
        rejects('duplicate_completed_ledger_key_rejected', lambda: collect_attempts(reg, ('fixture',)))
        write_fixture_chain()
        rejects('changed_seed_rejected', lambda: validate_raw_rows([{**raw, 'seed': raw['seed']+1}],
            plan['tasks'], 'fixture', identity, digest(entry), claim_identity, {99}))
        rejects('numeric_true_raw_attempt_rejected', lambda: validate_raw_rows([{**raw, 'attempt': 1}],
            plan['tasks'], 'fixture', identity, digest(entry), claim_identity, {99}))
        renormalized = copy.deepcopy(raw)
        renormalized['trace'][0]['sampling_log_probability'] += 1.2e-12
        check('sampling_renormalization_is_not_base_log_probability_equality',
            validate_raw_rows([renormalized], plan['tasks'], 'fixture', identity,
                digest(entry), claim_identity, {99})['rows'] == 1)
        correction = {'path': BASE_REL+'/registration/validation_correction.json',
            'sha256': 'c'*64, 'source_sha256': {'fixture_v2_only.py': 'd'*64}}
        reg['validation_correction'] = correction
        write_fixture_chain()
        pending_path = run/'chunk_00000.pending.jsonl'
        pending_path.write_bytes(raw_path.read_bytes())
        write_json(run/'failed.json', {'status': 'failed', 'error': 'CPU fixture original validator failure'})
        write_json(run/'released.json', {'claim_identity': claim_identity,
            'reason': 'stop_after_completed_chunk', 'keys': []})
        recovered_receipt = read_json(receipt_path)
        recovered_receipt['validation_correction'] = correction
        recovered_receipt['recovery_source'] = {'pending_path': str(pending_path.relative_to(folder)),
            'pending_sha256': file_sha(pending_path), 'recovery_path': str((run/'recovery.json').relative_to(folder))}
        write_json(receipt_path, recovered_receipt)
        write_jsonl(ledger_path, [{'key': key, 'dataset': 'hallusionbench', 'identity': identity,
            'receipt': str(receipt_path.relative_to(folder)), 'receipt_sha256': file_sha(receipt_path)}])
        recovery = {'schema': 'kdm_hallusion_independent128_recovery_v2', 'status': 'recovered',
            'identity': identity, 'claim_identity': claim_identity, 'owner_path': str((run/'owner.json').relative_to(folder)),
            'owner_sha256': file_sha(run/'owner.json'), 'validation_correction': correction,
            'process_observation': {'active': False}, 'regenerated_rows': 0, 'model_execution': False,
            'sealed_pending': [{'pending_path': str(pending_path.relative_to(folder)),
                'pending_sha256': file_sha(pending_path), 'raw_path': str(raw_path.relative_to(folder)),
                'raw_sha256': file_sha(raw_path), 'receipt_path': str(receipt_path.relative_to(folder)),
                'receipt_sha256': file_sha(receipt_path), 'keys': [key], 'rows': 1}],
            'completed_owned_rows': 1, 'released_keys': [], 'released_path': str((run/'released.json').relative_to(folder)),
            'released_sha256': file_sha(run/'released.json'), 'original_failed_path': str((run/'failed.json').relative_to(folder)),
            'original_failed_sha256': file_sha(run/'failed.json')}
        recovery['sealed_pending'][0]['recovery_path'] = str((run/'recovery.json').relative_to(folder))
        write_json(run/'recovery.json', recovery)
        recovered, _, _ = collect_attempts(reg, ('fixture',))
        check('registered_v2_recovery_keeps_exact_pending_raw_owner_failed_sources', len(recovered) == 1
            and recovered[0]['validation_correction_evidence']['original_pending_sha256'] == file_sha(raw_path)
            and recovered[0]['validation_correction_evidence']['original_failed_sha256'] == file_sha(run/'failed.json'))
        pending_path.write_text('Changed original pending bytes', encoding='utf-8')
        rejects('v2_recovery_original_pending_change_rejected', lambda: collect_attempts(reg, ('fixture',)))
        pending_path.write_bytes(raw_path.read_bytes())
        write_json(run/'recovery.json', {**recovery, 'regenerated_rows': 1})
        rejects('v2_recovery_regeneration_is_not_source_preservation', lambda: collect_attempts(reg, ('fixture',)))
        write_json(run/'recovery.json', recovery)
        bad_receipt = {**recovered_receipt, 'validation_correction': {**correction, 'sha256': 'e'*64}}
        rejects('unregistered_v2_correction_rejected', lambda: validate_receipt_correction(
            bad_receipt, owner, reg, folder, raw_path, run/'owner.json', receipt_path))
        write_fixture_chain()
        class FixtureScoring:
            def source_record(self, s):
                if s != fsample:
                    raise ValueError('Fixture source changed')
                return s
            def quality_request(self, s, answer):
                return {'quality_key': quality_key(s['question'], answer, s['gt_answer_details']),
                    'question': s['question'], 'answer': answer, 'gt_answer_details': s['gt_answer_details'],
                    'dataset': 'fixture', 'source': {'fixture_only': True}}
            def literal_binary_quality(self, s, answer):
                return None
            def validate_quality(self, s, answer, review):
                request = self.quality_request(s, answer)
                if any(request[f] != review.get(f) for f in ('quality_key', 'question', 'answer', 'gt_answer_details')):
                    raise ValueError('Fixture fullQA binding changed')
                validate_explicit_unknown(review, request)
                label = review.get('quality_label')
                if label not in QUALITY_LABELS:
                    raise ValueError('Fixture unresolved label')
                return {'quality_key': request['quality_key'], 'quality_label': label,
                        'score': int(label == 'correct'), 'official_correctness': QUALITY_LABELS[label],
                        'source': 'CPU_FIXTURE_ONLY'}
        scoring = FixtureScoring()
        fsamples = {fsample['id']: fsample}
        frel = {fsample['id']: {'reference_sufficient': True, 'reason': 'CPU fixture'}}
        _, queue, _ = prepare_rows(admitted, fsamples, frel, {}, scoring)
        request = queue[0]
        check('queue_binds_complete_QA_and_requests_quality_only', request['need_quality'] is True
            and request['need_behavior'] is False and request['answer'] == raw['text']
            and request['request_sha256'] == request_identity(request))
        review = {**{f: request[f] for f in ('quality_key', 'question', 'answer', 'gt_answer_details')},
            'quality_label': 'correct', 'answer_evidence': request['answer'],
            'reference_evidence': request['gt_answer_details'], 'reason': 'Synthetic CPU fixture only',
            'author': '/root/cpu_fixture', 'model': 'gpt-5.6-luna', 'effort': 'medium', 'call_id': 'CPU_FIXTURE_ONLY'}
        decision = {'schema': 'hallusion_independent_quality_decision_v1',
            **{f: request[f] for f in ('quality_key', 'request_sha256', 'question', 'answer', 'gt_answer_details')},
            'source_bindings': request['memberships'], 'quality_review': review}
        manual_path = root/'manual_fixture.jsonl'
        write_jsonl(manual_path, [decision])
        resolved, queue, _ = prepare_rows(admitted, fsamples, frel, {}, scoring, [manual_path])
        check('explicit_manual_quality_retains_actual_source_SHA', resolved[0]['score'] == 1 and not queue
            and resolved[0]['quality_source']['decision_sha256'] == file_sha(manual_path))
        altered = copy.deepcopy(decision); altered['source_bindings'][0]['source_line'] += 1
        write_jsonl(manual_path, [altered])
        rejects('manual_source_line_binding_change_rejected', lambda: prepare_rows(admitted, fsamples, frel, {}, scoring, [manual_path]))
        altered = copy.deepcopy(decision); altered['answer'] += ' changed'
        write_jsonl(manual_path, [altered])
        rejects('manual_full_answer_change_rejected', lambda: prepare_rows(admitted, fsamples, frel, {}, scoring, [manual_path]))
        altered = copy.deepcopy(decision); altered['quality_review'].update(
            quality_label='unknown', score=None, official_correctness=None)
        write_jsonl(manual_path, [altered])
        explicit, queue, _ = prepare_rows(admitted, fsamples, frel, {}, scoring, [manual_path])
        check('explicit_manual_unknown_has_null_score_and_no_pending_queue', explicit[0]['score'] is None
              and explicit[0]['quality_status'] == 'explicit_unknown' and not queue)
        unknown_ref = {fsample['id']: {'reference_sufficient': None, 'reason': 'Explicit CPU unknown',
            'registry_source': {'path': 'CPU_FIXTURE_ONLY', 'sha256': 'a'*64, 'row': 1}}}
        excluded, queue, _ = prepare_rows(admitted, fsamples, unknown_ref, {}, scoring)
        check('reference_unknown_is_outside_Luna_queue_with_null_quality', not queue
            and excluded[0]['quality_status'] == 'reference_unknown' and excluded[0]['quality'] is None
            and excluded[0]['score'] is None and excluded[0]['official_correctness'] is None)
        ten_unknown = [{**excluded[0], 'replicate': rep, 'key': f'fixture:{rep}'} for rep in range(10)]
        unknown_record = aggregate(ten_unknown, fsamples, unknown_ref, ('fixture',))[0]
        check('ten_reference_unknown_attempts_never_become_zero_correct_positive',
            unknown_record['should_abstain'] is None and unknown_record['n_correct'] is None
            and unknown_record['n_correct_observed'] is None and unknown_record['quality_reference_unknown'] == 10)
        frozen_conflicts_path = root/'conflicting_frozen_fixture.jsonl'
        conflict_key = quality_key(fsample['question'], raw['text'], fsample['gt_answer_details'])
        first = {'dataset': 'hallusionbench', 'model': 'fixture', 'condition_identity': 'conflict:c0',
            'sample_id': fsample['id'], 'key': 'conflict:k0', 'question': fsample['question'],
            'answer': raw['text'], 'gt_answer_details': fsample['gt_answer_details'], 'quality_key': conflict_key,
            'quality': {'quality_key': conflict_key, 'quality_label': 'correct', 'score': 1,
                        'official_correctness': 1, 'source': 'CPU_FIXTURE_ONLY'}, 'score': 1, 'abstain': False,
            'quality_source': {'decision_path': 'CPU_FIXTURE_SOURCE_A', 'decision_sha256': 'a'*64, 'decision_line': 1},
            'source_raw': str(raw_path), 'source_line': 1, 'source_sha256': file_sha(raw_path)}
        second = {**first, 'condition_identity': 'conflict:c1', 'key': 'conflict:k1', 'score': 0,
            'quality': {**first['quality'], 'quality_label': 'incorrect', 'score': 0, 'official_correctness': 0},
            'quality_source': {'decision_path': 'CPU_FIXTURE_SOURCE_B', 'decision_sha256': 'b'*64, 'decision_line': 2}}
        write_jsonl(frozen_conflicts_path, [first, second])
        frozen_conflicts_sha = file_sha(frozen_conflicts_path)
        conflict_index = load_frozen_quality(frozen_conflicts_path, frozen_conflicts_sha, 2, 2)
        audit = frozen_conflict_audit(conflict_index)
        check('conflicting_frozen_fullQA_is_never_reused_and_retains_every_old_source',
            reuse_quality(conflict_index, fsample, raw['text']) is None and len(audit) == 1
            and audit[0]['snapshot_rows'] == 2
            and [r['snapshot_line'] for r in audit[0]['snapshot_memberships']] == [1, 2]
            and {r['quality_label'] for r in audit[0]['snapshot_memberships']} == {'correct', 'incorrect'}
            and all(r['snapshot_sha256'] == frozen_conflicts_sha for r in audit[0]['snapshot_memberships']))
        class LiteralFixtureScoring(FixtureScoring):
            calls = 0
            def literal_binary_quality(self, s, answer):
                self.calls += 1
                return {'quality_key': quality_key(s['question'], answer, s['gt_answer_details']),
                    'quality_label': 'correct', 'score': 1, 'official_correctness': 1,
                    'source': 'CPU_LITERAL_FIXTURE_ONLY'}
        literal_scoring = LiteralFixtureScoring()
        conflicted_rows, queue, _ = prepare_rows(admitted, fsamples, frel, conflict_index, literal_scoring)
        check('conflicting_frozen_fullQA_goes_to_fresh_quality_queue_without_old_labels_or_literal_shortcut',
            conflicted_rows[0]['quality_status'] == 'pending' and len(queue) == 1
            and literal_scoring.calls == 0 and 'quality_label' not in json.dumps(queue[0])
            and queue[0]['answer'] == raw['text'])
        write_jsonl(manual_path, [decision])
        fresh, queue, _ = prepare_rows(admitted, fsamples, frel, conflict_index, literal_scoring, [manual_path])
        check('fresh_explicit_fullQA_quality_resolves_frozen_conflict_independently',
            fresh[0]['score'] == 1 and fresh[0]['quality_status'] == 'resolved' and not queue
            and fresh[0]['quality_source']['decision_sha256'] == file_sha(manual_path)
            and literal_scoring.calls == 0)
        write_jsonl(frozen_conflicts_path, [first])
        rejects('conflicting_snapshot_source_hash_gate_still_rejects_changes',
            lambda: load_frozen_quality(frozen_conflicts_path, frozen_conflicts_sha, 2, 2))
    refs = [{'model': 'fixture', 'sample_id': f'fixture:{n}', 'question': f'Q{n}',
        'gt_answer_details': 'R', 'should_abstain': u, 'status': 'fixture',
        'n_correct': 0 if u is True else 1 if u is False else None,
        'reference_sufficient': None if u is None else True}
        for n, u in enumerate((True, None, False, False))]
    scores = [{'model': 'fixture', 'sample_id': f'fixture:{n}', 'question': f'Q{n}',
        'gt_answer_details': 'R', 'condition_identity': 'fixture:c', 'key': f'fixture:{n}',
        'answer': 'A', 'abstain': abstain, 'score': score}
        for n, (abstain, score) in enumerate(((True, 1), (True, 0), (False, 1), (False, 0)))]
    _, metric = joint_metrics(scores, refs, denominator=4, expected_conditions=1)
    m = metric[0]
    check('unknown_abstention_produces_strict_J_interval', m['J'] is None and m['J_lower'] == .5 and m['J_upper'] == .75 and m['unknown_abstain'] == 1)
    check('quality_correct_abstention_is_not_double_counted_as_C', m['C'] == 1 and m['known_TP'] == 1 and m['quality_correct'] == 2)
    check('original_accuracy_AR_preserved', m['accuracy'] == .5 and m['abstention_rate'] == .5)
    fixed = copy.deepcopy(refs); fixed[1]['should_abstain'] = False
    _, metric = joint_metrics(scores, fixed, denominator=4, expected_conditions=1)
    check('no_unknown_abstention_identifies_point_J', metric[0]['J'] == .5 and metric[0]['J_lower'] == metric[0]['J_upper'])
    rejects('J_missing_reference_row_rejected', lambda: joint_metrics(scores, refs[:-1], 4, 1))
    rejects('J_duplicate_frozen_sample_rejected', lambda: joint_metrics(scores+[scores[0]], refs, 4, 1))
    return {'schema': 'hallusion_independent_cpu_fixture_v1', 'status': 'passed',
            'gpu_initialized': False, 'semantic_models_called': 0, 'count': len(names), 'checks': names}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--prepare-quality', action='store_true')
    mode.add_argument('--aggregate-reference', action='store_true')
    mode.add_argument('--joint', action='store_true', help='Recompute the source-bound reference then join frozen snapshot_074')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--protocol', type=Path, help='New registered protocol; default BASE/registration/protocol.json')
    parser.add_argument('--analysis-registration', type=Path, help='Separately frozen analysis registration path (required except self-test)')
    parser.add_argument('--analysis-registration-sha256', help='Explicit frozen SHA256 of analysis registration (required except self-test)')
    parser.add_argument('--manual-quality', type=Path, action='append', default=[], help='Explicit source-bound manual decision JSONL; repeatable')
    parser.add_argument('--out', type=Path, help='Fresh output folder; prepare-quality defaults to BASE/quality')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        if args.out:
            args.out.mkdir(parents=True, exist_ok=False)
            write_json(args.out/'cpu_fixture_receipt.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    root = args.root.resolve()
    base = root/BASE_REL
    protocol_path = args.protocol or base/'registration/protocol.json'
    if args.analysis_registration is None or args.analysis_registration_sha256 is None:
        parser.error('--analysis-registration and --analysis-registration-sha256 are required')
    reg = load_registration(root, protocol_path, args.analysis_registration, args.analysis_registration_sha256)
    # This module and the frozen exact-quality helper use standard Python only.
    # Do not import generate.py or any backend/runtime/model construction path.
    sys.path[:0] = [str(root/'src'), str(root)]
    from workflows.hallusion_blind.complete_response_quality import CompleteResponseScoring
    scoring = CompleteResponseScoring(root)
    frozen_path = root/FROZEN_REL
    frozen = load_frozen_quality(frozen_path)
    frozen_conflicts = frozen_conflict_audit(frozen)
    frozen_conflict_keys = {r['quality_key'] for r in frozen_conflicts}
    attempts, sources, diagnostics = collect_attempts(reg)
    manual_paths = [bound_path(base, project_path(root, path)) for path in args.manual_quality]
    quality, pending, manual_sources = prepare_rows(attempts, reg['samples'],
        reg['reliability'], frozen, scoring, manual_paths)
    references = aggregate(quality, reg['samples'], reg['reliability'])
    out = args.out or (base/'quality' if args.prepare_quality else None)
    if out is None:
        parser.error('--out is required for aggregate-reference and joint (fresh folder)')
    out = bound_path(base, project_path(root, out))
    if args.prepare_quality:
        if out.exists() and (out/'summary.json').exists():
            previous = read_json(out/'summary.json')
            if (previous.get('mode') != 'prepare_quality'
                    or previous.get('protocol_sha256') != reg['protocol_sha256']
                    or previous.get('analysis_registration_sha256') != reg['analysis_sha256']):
                raise ValueError('Progress queue belongs to another registered generation/analysis identity')
        out.mkdir(parents=True, exist_ok=True)
        atomic_output(out/'pending_unique.jsonl', write_jsonl, pending)
        atomic_output(out/'attempt_quality.jsonl', write_jsonl, quality)
        atomic_output(out/'frozen_quality_conflicts.jsonl', write_jsonl, frozen_conflicts)
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_jsonl(out/'pending_unique.jsonl', pending)
        write_jsonl(out/'attempt_quality.jsonl', quality)
        write_jsonl(out/'frozen_quality_conflicts.jsonl', frozen_conflicts)
    if not args.prepare_quality:
        write_jsonl(out/'reference.jsonl', references)
        write_csv(out/'reference.csv', [{k: v for k, v in r.items() if k != 'attempts'} for r in references])
    if args.joint:
        joined, metrics = joint_metrics((r for _, r in jsonl(frozen_path)), references)
        for r in joined:
            r['original_snapshot_sha256'] = FROZEN_SHA
        for r in metrics:
            r['original_snapshot_path'], r['original_snapshot_sha256'] = str(frozen_path), FROZEN_SHA
        write_jsonl(out/'joint_rows.jsonl', joined)
        write_jsonl(out/'joint_metrics.jsonl', metrics)
        write_csv(out/'joint_metrics.csv', metrics)
    counts = Counter(r['quality_status'] for r in quality)
    conflict_summary = {'schema': 'hallusion_frozen_quality_conflict_summary_v1',
        'snapshot_path': str(frozen_path), 'snapshot_sha256': FROZEN_SHA,
        'snapshot_rows': sum(len(item['snapshot_memberships']) for item in frozen.values()),
        'unique_quality_keys': len(frozen), 'reuse_eligible_quality_keys': len(frozen)-len(frozen_conflicts),
        'conflict_quality_keys': len(frozen_conflicts),
        'conflict_snapshot_rows': sum(item['snapshot_rows'] for item in frozen_conflicts),
        'new_attempts_matching_conflict': sum(r['quality_key'] in frozen_conflict_keys for r in quality),
        'new_pending_attempts_matching_conflict': sum(r['quality_key'] in frozen_conflict_keys and
            r['quality_status'] == 'pending' for r in quality),
        'new_pending_unique_matching_conflict': sum(r['quality_key'] in frozen_conflict_keys for r in pending),
        'new_reference_unknown_attempts_matching_conflict': sum(r['quality_key'] in frozen_conflict_keys and
            r['quality_status'] == 'reference_unknown' for r in quality),
        'new_explicit_decisions_matching_conflict': sum(r['quality_key'] in frozen_conflict_keys and
            r['quality_status'] in {'resolved', 'explicit_unknown'} for r in quality),
        'policy': 'All full-QA frozen label/score/code values must agree to reuse. Conflicts bypass literal rules and require fresh full-reference review on sufficient references. Old judgments are audit-only.'}
    atomic_output(out/'frozen_quality_conflicts_summary.json', write_json, conflict_summary)
    ref_counts = Counter('unknown' if r['should_abstain'] is None else str(r['should_abstain']).lower() for r in references)
    generation_complete = len(quality) == EXPECTED_ATTEMPTS
    processing_complete = generation_complete and counts['pending'] == 0
    summary = {'schema': 'hallusion_independent_scoring_receipt_v1',
        'mode': 'prepare_quality' if args.prepare_quality else 'joint' if args.joint else 'aggregate_reference',
        'gpu_initialized': False, 'semantic_models_called': 0,
        'expected_models': len(MODELS), 'expected_samples_per_model': N_SAMPLES,
        'expected_replicates': list(range(N_REPLICATES)), 'expected_attempts': EXPECTED_ATTEMPTS,
        'validated_attempts': len(quality), 'missing_attempts': EXPECTED_ATTEMPTS-len(quality),
        'generation_complete': generation_complete, 'quality_status_counts': dict(counts),
        'pending_unique_quality': len(pending), 'reference_counts': dict(ref_counts),
        'reference_processing_complete': processing_complete,
        'reference_processing_complete_with_explicit_unknowns': processing_complete and ref_counts['unknown'] > 0,
        'status': 'complete_with_explicit_unknowns' if processing_complete and ref_counts['unknown']
                  else 'complete' if processing_complete else 'generation_incomplete' if not generation_complete
                  else 'quality_pending',
        'protocol_path': str(reg['protocol_path']), 'protocol_sha256': reg['protocol_sha256'],
        'frozen_snapshot_path': str(frozen_path), 'frozen_snapshot_sha256': FROZEN_SHA,
        'frozen_quality_conflict_summary': conflict_summary,
        'frozen_manifest_sha256': MANIFEST_SHA, 'quality_rule_sources_sha256': RULE_SOURCES,
        'runtime_proof_sources': reg['runtime_proofs'],
        'analysis_registration_path': str(reg['analysis_path']),
        'analysis_registration_sha256': reg['analysis_sha256'],
        'reference_reliability': reg['analysis']['reference_reliability'],
        'validation_correction': reg.get('validation_correction'),
        'manual_quality_sources': manual_sources, 'generation_sources': sources,
        'per_model_progress': diagnostics, 'scorer_sha256': file_sha(Path(__file__)),
        'output_hashes': {name: file_sha(out/name) for name in
            ('pending_unique.jsonl', 'attempt_quality.jsonl', 'reference.jsonl', 'reference.csv',
             'joint_rows.jsonl', 'joint_metrics.jsonl', 'joint_metrics.csv',
             'frozen_quality_conflicts.jsonl', 'frozen_quality_conflicts_summary.json') if (out/name).is_file()}}
    atomic_output(out/'summary.json', write_json, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'generation_sources'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
