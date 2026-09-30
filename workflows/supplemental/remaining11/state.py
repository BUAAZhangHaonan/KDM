"""Refresh a compact checkpoint from this run's actual host receipts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import atomic_json, file_hash, read_jsonl
from kdm.pipeline import experiment_tasks
from workflows.supplemental.remaining11.generate import MODELS
from workflows.supplemental.remaining11.score import COND
from workflows.supplemental.remaining11.snapshot import build_snapshot, run_folder


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshots(folder, run):
    local = build_snapshot(run)
    latest = {local['host']: local}
    for path in sorted((folder / 'remote_completed').glob('*/snapshots/*/host_snapshot.json')):
        snapshot = read(path)
        if snapshot['run'] != run:
            raise ValueError('Snapshot belongs to another supplemental run')
        host = snapshot['host']
        if host not in latest or snapshot['captured_utc'] > latest[host]['captured_utc']:
            latest[host] = snapshot
    return latest


def latest_summary(paths, schema):
    candidates = [(path, read(path)) for path in paths]
    candidates = [(path, value) for path, value in candidates if value['schema'] == schema]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: datetime.fromisoformat(entry[1]['updated_utc']))


def food_score_checkpoint(folder, path, summary):
    source = path.parent
    files = [source / name for name in ('condition_counts.csv', 'boundary_queue.jsonl', 'reference_G.jsonl')]
    identity = {'summary_sha256': file_hash(path),
                'files': {str(item.relative_to(ROOT)): {'bytes': item.stat().st_size,
                                                       'mtime_ns': item.stat().st_mtime_ns} for item in files},
                'registered_condition_source_sha256': file_hash(ROOT / 'configs/kdm/method_plan.json')}
    epoch = identity['summary_sha256'][:20] + '_' + identity['registered_condition_source_sha256'][:20]
    cache = folder / 'state_checkpoints' / (epoch + '.json')
    if cache.is_file():
        checkpoint = read(cache)
        if checkpoint['identity'] != identity:
            raise ValueError('Immutable score checkpoint inputs changed')
        return checkpoint
    samples = [row for row in read_jsonl(ROOT / 'data/current/all.jsonl') if row['dataset'] == 'food101']
    evaluation = [row for row in samples if row['split'] == 'eval']
    classes = {row['class'] for row in samples}
    if (len(samples) != 4848 or len(evaluation) != 2424 or len(classes) != 101
            or any(Counter(row['class'] for row in samples if row['split'] == split) !=
                   Counter({name: 24 for name in classes}) for split in ('dev', 'eval'))):
        raise ValueError('Actual Food score denominator differs from registered quotas')
    methods = read(ROOT / 'configs/kdm/method_plan.json')
    expected = {}
    for model in MODELS:
        for task in experiment_tasks((evaluation[0],), methods=methods[model]['food101']):
            condition = {'model': model, 'dataset': 'food101', 'split': 'eval', **task}
            key = tuple(condition[field] for field in COND)
            if key in expected:
                raise ValueError('Duplicate condition in registered Food denominator')
            expected[key] = condition
    if len(expected) != 832:
        raise ValueError('Food registered condition count differs from832')
    by_model = defaultdict(lambda: {'stages': {}, 'reference_expected': 0, 'reference_complete': 0})
    for entry in summary['selected_sources']:
        stage = by_model[entry['model']]['stages'].setdefault(entry['stage'], Counter())
        stage['scored_rows'] += entry['selected_rows']
    seen_pending = set()
    for question in read_jsonl(source / 'boundary_queue.jsonl'):
        for member in question['memberships']:
            key = (member['model'], member['stage'], member['key'])
            if key in seen_pending:
                raise ValueError('Duplicate boundary membership in immutable score checkpoint')
            seen_pending.add(key)
            by_model[member['model']]['stages'][member['stage']]['label_pending_rows'] += 1
    observed, condition_counts = {}, Counter()
    with (source / 'condition_counts.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            row['replicate'] = int(row['replicate'])
            for field in ('guided', 'reference_guided'):
                if row[field] not in ('True', 'False'):
                    raise ValueError('Invalid condition boolean in actual score table')
                row[field] = row[field] == 'True'
            key = tuple(row[field] for field in COND)
            if key not in expected or key in observed or row['main_marker'] != row['marker']:
                raise ValueError('Score condition differs from the complete registered denominator')
            observed[key] = {field: int(row[field]) for field in ('n', 'canonical_unknown', 'abstain_unknown')}
            counts = observed[key]
            if not 0 <= counts['n'] <= len(evaluation) or any(not 0 <= counts[field] <= counts['n']
                    for field in ('canonical_unknown', 'abstain_unknown')):
                raise ValueError('Condition score coverage exceeds its registered denominator')
            stage = by_model[row['model']]['stages']['formal']
            stage['canonical_pending_rows'] += counts['canonical_unknown']
            stage['abstain_pending_rows'] += counts['abstain_unknown']
    reference_seen = set()
    for row in read_jsonl(source / 'reference_G.jsonl'):
        key = row['model'], row['sample_id']
        if row['dataset'] != 'food101' or key in reference_seen or not isinstance(row['reference_complete'], bool):
            raise ValueError('Reference count source has a duplicate key or foreign dataset')
        reference_seen.add(key)
        model = by_model[row['model']]
        model['reference_expected'] += 1
        model['reference_complete'] += row['reference_complete']
        stage = model['stages'].setdefault('independent', Counter())
        stage['canonical_pending_rows'] += sum(item['canonical_name_in_primary_score'] is None for item in row['attempts'])
        stage['abstain_pending_rows'] += sum(item['abstain'] is None for item in row['attempts'])
    if (len(reference_seen) != summary['reference_expected_rows']
            or sum(value['reference_complete'] for value in by_model.values()) != summary['reference_complete_rows']
            or sum(entry['scored_rows'] for model in by_model.values() for stage, entry in model['stages'].items()
                   if stage != 'candidate') != summary['rows']):
        raise ValueError('Actual score/reference counts differ from immutable summary')
    progress_path = folder / 'state_checkpoints' / (epoch + '_condition_progress.csv')
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(COND) + ['main_marker', 'expected_rows', 'scored_rows', 'missing_scored_rows',
                           'canonical_pending_rows', 'abstain_pending_rows', 'full_decided_condition']
    with progress_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(expected, key=lambda value: tuple(map(str, value))):
            counts = observed.get(key, {'n': 0, 'canonical_unknown': 0, 'abstain_unknown': 0})
            full = counts['n'] == len(evaluation) and not counts['canonical_unknown'] and not counts['abstain_unknown']
            writer.writerow({**dict(zip(COND, key)), 'main_marker': key[5], 'expected_rows': len(evaluation),
                             'scored_rows': counts['n'], 'missing_scored_rows': len(evaluation) - counts['n'],
                             'canonical_pending_rows': counts['canonical_unknown'],
                             'abstain_pending_rows': counts['abstain_unknown'], 'full_decided_condition': full})
            condition_counts['expected_conditions'] += 1
            condition_counts['observed_conditions'] += counts['n'] > 0
            condition_counts['complete_decided_conditions'] += full
            stage = by_model[key[0]]['stages']['formal']
            stage['expected_conditions'] += 1
            stage['complete_decided_conditions'] += full
    checkpoint = {'identity': identity, 'models': dict(by_model), 'condition_counts': dict(condition_counts),
                  'condition_progress_path': str(progress_path.relative_to(ROOT)),
                  'condition_progress_sha256': file_hash(progress_path)}
    atomic_json(cache, checkpoint)
    return checkpoint


def scoring_checkpoints(folder):
    paths = sorted((folder / 'scores').glob('*/summary.json'))
    food = latest_summary(paths, 'kdm_remaining11_main_score_v3')
    viz = latest_summary(paths, 'kdm_remaining11_vizwiz_baseline_score_v1')
    compact, detail = {}, {}
    if food is not None:
        path, summary = food
        detail['food101'] = food_score_checkpoint(folder, path, summary)
        fields = ('updated_utc', 'formal_rows', 'independent_rows', 'candidate_rows', 'canonical_pending_rows',
                  'abstain_pending_rows', 'pending_boundary_questions', 'reference_expected_rows',
                  'reference_complete_rows', 'reference_pending_rows', 'reference_rule', 'source_manifest',
                  'source_manifest_sha256', 'scorer_sha256', 'complete')
        compact['food101'] = {field: summary[field] for field in fields}
        compact['food101'].update(directory=str(path.parent.relative_to(ROOT)), summary_sha256=file_hash(path),
                                 **{key: value for key, value in detail['food101'].items() if key in
                                    ('condition_counts', 'condition_progress_path', 'condition_progress_sha256')})
    if viz is not None:
        path, summary = viz
        reference = read(path.parent / 'reference_status.json')
        fields = ('updated_utc', 'scored_rows', 'registered_formal_rows', 'registered_conditions',
                  'complete_decided_conditions', 'pending_score_rows', 'missing_scored_formal_rows',
                  'asset_manifest_path', 'scorer_sha256', 'complete_registered_methods')
        compact['vizwiz'] = {field: summary[field] for field in fields}
        compact['vizwiz'].update(directory=str(path.parent.relative_to(ROOT)), summary_sha256=file_hash(path),
                                 reference_status=reference,
                                 condition_progress_path=str((path.parent / 'condition_coverage.csv').relative_to(ROOT)),
                                 verification_path=str((path.parent / 'verification.json').relative_to(ROOT)),
                                 validation_path=str((path.parent / 'validation.json').relative_to(ROOT)))
        with (path.parent / 'condition_coverage.csv').open(encoding='utf-8') as source:
            models = defaultdict(Counter)
            for row in csv.DictReader(source):
                if row['dataset'] != 'vizwiz':
                    raise ValueError('VizWiz condition coverage contains another dataset')
                counts = models[row['model']]
                counts['expected_conditions'] += 1
                counts['complete_decided_conditions'] += row['condition_complete'] == 'True'
                scored = int(row['scored_n'])
                if row['score_unresolved_n'] == '' and scored != 0:
                    raise ValueError('Observed VizWiz score condition lacks its unresolved count')
                counts['scored_rows'] += scored
                counts['label_pending_rows'] += int(row['score_unresolved_n']) if scored else 0
            if sum(value['scored_rows'] for value in models.values()) != summary['scored_rows']:
                raise ValueError('Actual VizWiz condition counts differ from score summary')
            detail['vizwiz'] = {'models': dict(models)}
    return compact, detail


def latest_analysis(folder, scoring):
    item = latest_summary(sorted((folder / 'analysis').glob('*/summary.json')), 'kdm_remaining11_analysis_v1')
    if item is None:
        return None
    path, summary = item
    result = {key: value for key, value in summary.items() if isinstance(value, (int, bool))}
    result.update(directory=str(path.parent.relative_to(ROOT)), updated_utc=summary['updated_utc'],
                  score_directory=summary['score_directory'], analysis_sha256=summary['analysis_sha256'],
                  summary_sha256=file_hash(path),
                  latest_food_score_input_matches=summary['input_sha256']['score_summary'] ==
                      scoring.get('food101', {}).get('summary_sha256'),
                  condition_coverage_path=str((path.parent / 'condition_coverage.csv').relative_to(ROOT)))
    for name in ('validation.json', 'verification.json'):
        if (path.parent / name).is_file():
            result[name.removesuffix('.json') + '_path'] = str((path.parent / name).relative_to(ROOT))
    return result


def progress_eta(folder, state, hosts):
    """Derive finite observed-rate projections from real immutable part receipts."""
    captured = datetime.now(timezone.utc)
    claims = []
    for host, snapshot in hosts.items():
        for job in snapshot['jobs']:
            progress = job['progress']
            remaining = progress['expected'] - progress['completed']
            if remaining < 0:
                raise ValueError('Observed generation rows exceed the original claim')
            recent = job.get('recent_part_measurements', [])
            intervals = []
            for previous, current in zip(recent, recent[1:]):
                seconds = (datetime.fromisoformat(current['finished_utc']) -
                           datetime.fromisoformat(previous['finished_utc'])).total_seconds()
                if seconds <= 0:
                    raise ValueError('Immutable part completion times are not increasing')
                intervals.append({'part': current['part'], 'rows': current['rows'],
                                  'seconds': seconds, 'rows_per_second': current['rows'] / seconds})
            elapsed = (datetime.fromisoformat(progress.get('updated_utc', progress['started_utc'])) -
                       datetime.fromisoformat(progress['started_utc'])).total_seconds()
            active = progress['status'] == 'running' and job['process']['present']
            if active and intervals:
                rate = sum(item['rows'] for item in intervals) / sum(item['seconds'] for item in intervals)
                rate_source = 'last_three_immutable_part_completion_receipts'
            elif active and elapsed > 0 and progress['completed']:
                rate = progress['completed'] / elapsed
                rate_source = 'since_start_including_loading_insufficient_completed_parts'
            else:
                rate, rate_source = None, None
            eta = remaining / rate if rate else None
            claims.append({
                'host': host, 'model': job['model'], 'dataset': job['dataset'], 'stage': job['stage'],
                'claim_id': job['claim_id'], 'pid': progress['pid'],
                'process_present': job['process']['present'], 'status': progress['status'],
                'ownership_released': job.get('ownership_release') is not None,
                'expected_rows': progress['expected'], 'written_rows': progress['completed'],
                'remaining_claim_rows': remaining, 'started_utc': progress['started_utc'],
                'progress_updated_utc': progress.get('updated_utc', progress['started_utc']),
                'snapshot_captured_utc': snapshot['captured_utc'],
                'last_progress_age_seconds_at_snapshot':
                    (datetime.fromisoformat(snapshot['captured_utc']) -
                     datetime.fromisoformat(progress.get('updated_utc', progress['started_utc']))).total_seconds(),
                'recent_parts': recent, 'observed_intervals': intervals,
                'rows_per_second': rate, 'rate_source': rate_source,
                'projected_remaining_hours_at_observed_mix': eta / 3600 if eta is not None else None,
                'projected_finish_utc_at_observed_mix':
                    (captured + timedelta(seconds=eta)).isoformat() if eta is not None else None,
                'progress_source': job['progress_path'], 'admission_source': job['admission_path'],
                'failure_source': job['failure_path'],
                'failure_error': job['failure']['error'] if job['failure'] else None,
            })
    stages = []
    for row in state['model_stages']:
        active = [claim for claim in claims if (claim['model'], claim['dataset'], claim['stage']) ==
                  (row['model'], row['dataset'], row['stage']) and claim['status'] == 'running'
                  and claim['process_present']]
        rates = [claim['rows_per_second'] for claim in active if claim['rows_per_second']]
        remaining = (row['running_claimed_remaining_rows'] + row['failed_held_remaining_rows'] +
                     row['not_yet_claimed_rows'])
        labels = row['scoring_progress']
        stages.append({
            'model': row['model'], 'dataset': row['dataset'], 'stage': row['stage'],
            'expected_rows': row['expected_rows'], 'reused_rows': row['original_reused_rows'],
            'new_completed_immutable_rows': row['new_completed_immutable_rows'],
            'running_partial_written_rows': row['running_partial_written_rows'],
            'remaining_generation_rows': remaining,
            'running_claimed_remaining_rows': row['running_claimed_remaining_rows'],
            'failed_held_remaining_rows': row['failed_held_remaining_rows'],
            'not_yet_claimed_rows': row['not_yet_claimed_rows'],
            'active_claims': [claim['claim_id'] for claim in active],
            'current_parallel_rows_per_second': sum(rates) if rates else None,
            'projected_running_claim_completion_hours_at_observed_mix':
                max(claim['projected_remaining_hours_at_observed_mix'] for claim in active)
                if active and len(rates) == len(active) else None,
            'projection_all_remaining_rows_hours_at_current_parallel_mix':
                remaining / sum(rates) / 3600 if rates else None,
            'completed_rows_scored': labels.get('scored_rows', 0) if labels is not None else None,
            'current_scored_boundary_pending_rows': row['labels_review_remaining'],
            'unscored_immutable_available_rows': max(0, row['original_reused_rows'] +
                row['new_completed_immutable_rows'] - labels.get('scored_rows', 0)) if labels is not None else None,
            'reference_pending_rows': row['reference_remaining'],
            'full_stage_generation_complete': row['generation_rows_covered'],
        })
    annotation = {}
    for dataset in state['dataset_totals']:
        independent = [row for row in stages if row['dataset'] == dataset and row['stage'] == 'independent']
        annotation[dataset] = {
            'registered_independent_rows': sum(row['expected_rows'] for row in independent),
            'immutable_available_rows': sum(row['reused_rows'] + row['new_completed_immutable_rows'] for row in independent),
            'completed_rows_scored': sum(row['completed_rows_scored'] or 0 for row in independent),
            'current_scored_boundary_pending_rows': sum(row['current_scored_boundary_pending_rows'] or 0
                                                       for row in independent),
            'unscored_immutable_available_rows': sum(row['unscored_immutable_available_rows'] or 0
                                                    for row in independent),
            'generation_remaining_rows': sum(row['remaining_generation_rows'] for row in independent),
            'semantic_boundary_annotation_eta_hours': None,
            'annotation_eta_reason': 'Actual Luna boundary-batch and root-review throughput must be measured separately',
        }
    output = {'schema': 'kdm_remaining11_observed_progress_eta_v1', 'updated_utc': captured.isoformat(),
              'run': state['run'], 'host_snapshot_times': {host: snapshot['captured_utc'] for host, snapshot in hosts.items()},
              'claims': claims, 'model_stages': stages, 'independent_annotation': annotation,
              'scoring_source': state['scoring_checkpoints'],
              'eta_limits': [
                  'Rates use the latest two completion intervals of at most three immutable512-row parts.',
                  'Formal method, kind, image and token-length mixtures can change after the observed parts; later methods may run slower.',
                  'Since-start rates include model loading and are provisional when fewer than two immutable parts exist.',
                  'All-remaining projection holds current parallelism and observed method mix fixed; held and unassigned work has no admitted completion schedule.',
                  'GPU extraction projections do not measure Luna annotation, root review, scoring, merging or future five-model experiments.',
                  'Cross-validation of whether abstention is appropriate follows core answer extraction; registered reference and ten-attempt raw definitions are retained.',
              ], 'gpu_queries': 0, 'jobs_interrupted': 0, 'completion_claim': False}
    path = folder / 'PROGRESS_ETA.json'
    atomic_json(path, output)
    fields = [key for key in stages[0] if key != 'active_claims']
    with (folder / 'PROGRESS_ETA.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in stages)
    return str(path.relative_to(ROOT))


def update(run):
    folder = run_folder(run)
    with (folder / 'assets/asset_gaps.csv').open(encoding='utf-8') as source:
        gaps = list(csv.DictReader(source))
    for gap in gaps:
        gap['dataset'] = 'food101'
    viz_manifest = folder / 'assets/vizwiz/audit_20260930/asset_manifest.json'
    if viz_manifest.is_file():
        gaps.extend(read(viz_manifest)['gaps'])
    scoring, score_detail = scoring_checkpoints(folder)
    hosts = snapshots(folder, run)
    jobs = [job for snapshot in hosts.values() for job in snapshot['jobs']]
    parts = {part['part_id']: part for snapshot in hosts.values()
             for part in snapshot['completed_parts']}
    manifest_path = folder / 'remote_completed/received_manifest.json'
    received = read(manifest_path)['parts'] if manifest_path.is_file() else []
    complete_by_claim = defaultdict(int)
    complete_by_stage = defaultdict(int)
    central_by_stage = defaultdict(int)
    for part in parts.values():
        complete_by_claim[(part['host'], part['model'], part['dataset'], part['stage'], part['claim_id'])] += part['rows']
        complete_by_stage[(part['model'], part['dataset'], part['stage'])] += part['rows']
        if part['host'] == hosts[next(iter(hosts))]['host']:
            central_by_stage[(part['model'], part['dataset'], part['stage'])] += part['rows']
    for part in received:
        central_by_stage[(part['model'], part['dataset'], part['stage'])] += part['rows']
    stage_jobs = defaultdict(list)
    for job in jobs:
        stage_jobs[(job['model'], job['dataset'], job['stage'])].append(job)
    rows = []
    for gap in gaps:
        model, dataset, stage = gap['model'], gap['dataset'], gap['stage']
        immutable = complete_by_stage[(model, dataset, stage)]
        running_remaining = failed_held = partial_running = released_zero_load = 0
        job_states = []
        for job in stage_jobs[(model, dataset, stage)]:
            progress = job['progress']
            expected = int(progress['expected'])
            written = int(progress['completed'])
            completed = complete_by_claim[(job['host'], model, dataset, stage, job['claim_id'])]
            status = progress['status']
            present = job['process']['present']
            release = job.get('ownership_release')
            if release is not None:
                if (status != 'failed' or present or written or completed
                        or release['source_generated_rows'] != 0 or release['expected_rows'] != expected
                        or release['scientific_parameters_changed']):
                    raise ValueError('Ownership-release snapshot has generation rows or mismatched coverage')
                remaining = 0
                released_zero_load += expected
                effective_status = 'zero_load_failure_released'
            elif status in ('loading', 'running') and present:
                remaining = max(expected - written, 0)
                running_remaining += remaining
                partial_running += max(written - completed, 0)
                effective_status = 'running'
            elif status == 'failed' or (status in ('loading', 'running') and not present):
                remaining = max(expected - completed, 0)
                failed_held += remaining
                effective_status = 'failed_held' if status == 'failed' else 'process_missing_held'
            else:
                remaining = max(expected - completed, 0)
                effective_status = status
            job_states.append({'host': job['host'], 'claim_id': job['claim_id'],
                               'status': effective_status, 'remaining': remaining,
                               'immutable_rows': completed, 'observed_rows': written,
                               'pid': job['process']['pid'], 'process_present': present,
                               'progress_path': job['progress_path'],
                               'failure_path': job.get('failure_path'),
                               'ownership_release': release})
        initial_missing = int(gap['remaining_rows'])
        not_claimed = initial_missing - immutable - partial_running - running_remaining - failed_held
        if not_claimed < 0 or immutable > initial_missing:
            raise ValueError(f'Overlapping or excessive ownership counts for {model}/{dataset}/{stage}')
        if failed_held:
            next_step = 'Preserve failed source and held keys; score validated immutable rows; continue independent claims'
        elif running_remaining:
            next_step = 'Continue existing claim range; score immutable completed parts'
        elif not_claimed and released_zero_load:
            next_step = 'Start the explicitly bound replacement claim with unchanged full keys and runtime; preserve original zero-load failure'
        elif not_claimed:
            next_step = 'Dispatch explicit unclaimed keys using registered runtime and unchanged scientific parameters'
        else:
            next_step = 'Validate complete key coverage and quotas, then resolve scoring and uniform references'
        labels = references = None
        if dataset == 'food101' and dataset in score_detail:
            scored = score_detail[dataset]['models'][model]
            labels = dict(scored['stages'].get(stage, {}))
            references = {'expected': scored['reference_expected'], 'complete': scored['reference_complete'],
                          'pending': scored['reference_expected'] - scored['reference_complete']}
        elif dataset == 'vizwiz' and dataset in scoring:
            labels = dict(score_detail[dataset]['models'][model]) if stage == 'formal' else {
                'scored_rows': scoring[dataset]['reference_status']['independent_attempt_scores_available_in_this_baseline_checkpoint'],
                'label_pending_rows': 0}
            labels['score_protocol'] = 'official_VQA'
            official = scoring[dataset]['reference_status']
            independent_gap = next(entry for entry in gaps
                                   if (entry['model'], entry['dataset'], entry['stage']) == (model, dataset, 'independent'))
            references = {field: official[field] for field in ('official_reference_source', 'official_reference_eval_rows',
                          'human_answerable_eval_rows', 'human_unanswerable_eval_rows', 'knowledge_deficit_GT')}
            references.update(independent_expected_rows=independent_gap['expected_rows'],
                              independent_scored_rows=labels['scored_rows'] if stage == 'independent' else 0,
                              independent_pending_rows=independent_gap['expected_rows'])
        label_pending = labels.get('label_pending_rows', 0) if labels is not None else None
        rows.append({'model': model, 'dataset': dataset, 'stage': stage,
                     'expected_rows': int(gap['expected_rows']),
                     'original_reused_rows': int(gap['reusable_rows']),
                     'initial_missing_rows': initial_missing,
                     'new_completed_immutable_rows': immutable,
                     'central_available_completed_rows': central_by_stage[(model, dataset, stage)],
                     'running_partial_written_rows': partial_running,
                     'running_claimed_remaining_rows': running_remaining,
                     'failed_held_remaining_rows': failed_held,
                     'released_zero_load_source_rows': released_zero_load,
                     'not_yet_claimed_rows': not_claimed,
                     'scoring_progress': labels, 'reference_progress': references,
                     'labels_review_remaining': label_pending,
                     'reference_remaining': references['pending'] if references is not None and dataset == 'food101' else
                         references['independent_pending_rows'] if references is not None else None,
                     'labels_complete': labels is not None and labels.get('scored_rows', 0) == int(gap['expected_rows'])
                         and label_pending == 0,
                     'references_complete': references is not None and dataset == 'food101' and references['pending'] == 0,
                     'generation_rows_covered': immutable == initial_missing,
                     'jobs': job_states, 'next_step': next_step})
    count_fields = ('expected_rows', 'original_reused_rows', 'initial_missing_rows',
                    'new_completed_immutable_rows', 'central_available_completed_rows',
                    'running_partial_written_rows', 'running_claimed_remaining_rows',
                    'failed_held_remaining_rows', 'released_zero_load_source_rows', 'not_yet_claimed_rows')
    dataset_totals = {dataset: {stage: {field: sum(row[field] for row in rows
                      if row['dataset'] == dataset and row['stage'] == stage) for field in count_fields}
                      for stage in ('formal', 'independent', 'candidate')}
                      for dataset in sorted({row['dataset'] for row in rows})}
    totals = dataset_totals['food101']
    compact_jobs = []
    for job in jobs:
        progress = job['progress']
        compact_jobs.append({
            'host': job['host'], 'model': job['model'], 'dataset': job['dataset'],
            'stage': job['stage'], 'claim_id': job['claim_id'],
            'status_at_snapshot': progress['status'], 'expected_rows': progress['expected'],
            'observed_written_rows': progress['completed'],
            'pid': job['process']['pid'], 'process_present_at_snapshot': job['process']['present'],
            'progress_updated_utc': progress.get('updated_utc', progress['started_utc']),
            'progress_path': job['progress_path'], 'admission_path': job['admission_path'],
            'admission_sha256': job['admission_sha256'],
            'registered_backend_sha256': job['admission']['registered_backend_sha256'],
            'runtime_backend_sha256': job['admission']['backend_sha256'],
            'runner_sha256': job['admission']['runner_sha256'],
            'failure_path': job['failure_path'],
            'failure_error': job['failure']['error'] if job['failure'] else None,
            'ownership_release': job.get('ownership_release'),
        })
    state = {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'run': run, 'task': 'KDM remaining11 supplemental',
        'internal_submission_date': '2026-10-11',
        'paper_target': 'NAACL 2027 October ARR (user-provided target)',
        'frozen_five_models': {'rows': 853248, 'conditions': 352, 'status': 'retained'},
        'asset_gap_table': str((folder / 'assets/asset_gaps.csv').relative_to(ROOT)),
        'totals': totals, 'totals_dataset': 'food101', 'dataset_totals': dataset_totals, 'model_stages': rows,
        'host_snapshots': {host: {'captured_utc': snapshot['captured_utc'],
                                 'root': snapshot['project_root'], 'jobs': len(snapshot['jobs']),
                                 'completed_parts': len(snapshot['completed_parts'])}
                           for host, snapshot in hosts.items()},
        'jobs': compact_jobs, 'scoring_checkpoints': scoring, 'latest_analysis': latest_analysis(folder, scoring),
        'received_manifest': str(manifest_path.relative_to(ROOT)) if manifest_path.is_file() else None,
        'received_verified_parts': len(received),
        'label_review_pending': {dataset: {field: checkpoint[field] for field in
                                 ('canonical_pending_rows', 'abstain_pending_rows', 'pending_boundary_questions')}
                                 if dataset == 'food101' else {'pending_score_rows': checkpoint['pending_score_rows']}
                                 for dataset, checkpoint in scoring.items()},
        'reference_pending': {dataset: checkpoint['reference_pending_rows'] if dataset == 'food101' else
                              checkpoint['reference_status'] for dataset, checkpoint in scoring.items()},
        'next_steps': ['continue existing safe Food-101 claims and preserve scientific failures',
                       'score and review each verified immutable part as it becomes available',
                       'merge uniform references and complete registered method conditions',
                       'derive controls, paired statistics, supplemental figures and appendix'],
        'vizwiz_registration': str((folder / 'assets/registered_inclusion.json').relative_to(ROOT)),
        'gpu_queries_per_update': 0, 'completion_claim': False,
    }
    state['progress_eta_path'] = progress_eta(folder, state, hosts)
    atomic_json(folder / 'CURRENT_STATE.json', state)
    atomic_json(ROOT / 'CURRENT_STATE.json', state)
    checklist = ['# KDM remaining11 运行与验收', '',
                 '- [x] 阅读最终协议和历史定位索引。',
                 '- [x] 一次资源、GPU UUID、真实进程与锁检查。',
                 '- [x] 逐模型资产覆盖、身份和剩余键核对。',
                 f'- [{"x" if received else " "}] 跨机完成分片原SHA、身份与逐行key/config核验后接收。',
                 '- [ ] 缺失生成完成；每条件和类别配额完整。',
                 '- [ ] 主评分、弃权与统一参考GT全部已决。',
                 '- [ ] 必要Luna标注与root复核合并。',
                 '- [ ] 完整方法比较、控制、配对统计与真实图表。',
                 '- [ ] 补充文档、论文附录与细粒度提交交付。', '']
    (folder / 'RUN_CHECKLIST.md').write_text('\n'.join(checklist), encoding='utf-8')
    print(json.dumps({'totals': totals, 'jobs': len(jobs), 'received_parts': len(received),
                      'state': str(folder / 'CURRENT_STATE.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run_20260930_140337')
    update(parser.parse_args().run)
