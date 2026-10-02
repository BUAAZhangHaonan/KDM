"""Audit frozen CDA traces by exact source locators; no model initialization."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ('model', 'method', 'kind', 'marker', 'reference_marker', 'guided', 'reference_guided', 'replicate')
CALIBRATION = ('h_prior', 'h_context', 'h_null_prior', 'h_null_context', 'rp', 'rc', 'wp', 'wc', 'wa', 'weights')
SCALARS = ('h_prior', 'h_context', 'h_null_prior', 'h_null_context', 'rp', 'rc', 'wp', 'wc', 'wa')
WEIGHTS = ('wp', 'wc', 'wa')
STEP_FIELDS = (
    'panel', 'dataset', 'split', 'model', 'condition_id', 'marker', 'reference_marker', 'sample_id',
    'state', 'uniform_reference', 'source_file_id', 'source_line', 'score_source_line', 'step_index',
    'token', 'h_prior', 'h_context', 'h_null_prior', 'h_null_context', 'rp', 'rc', 'wp', 'wc', 'wa',
    'zero_sum_extension', 'negative_wa', 'calibration_status', 'log_probability',
    'recomputed_rp', 'recomputed_rc', 'recomputed_wp', 'recomputed_wc', 'recomputed_wa',
    'equations_6_7_max_error', 'weights_vector_max_error', 'weights_sum_error',
    'zero_sum_flag_matches', 'negative_wa_flag_matches', 'largest_weight_branch',
    'h_abstention_available', 'branch_logits_saved',
)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def moment() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_totals(group: dict, stats: dict) -> None:
    for key, value in stats.items():
        if key.endswith('_max_error') or key.endswith('_max'):
            group[key] = max(group.get(key, -math.inf), value)
        elif key.endswith('_min'):
            group[key] = min(group.get(key, math.inf), value)
        else:
            group[key] = group.get(key, 0) + value


def finalize_group(key: tuple, stats: dict) -> dict:
    panel, aggregation, model, marker, state, reference = key
    result = {
        'panel': panel, 'aggregation': aggregation, 'dataset': 'food101', 'split': 'eval',
        'model': model, 'marker': marker, 'state': state, 'uniform_reference': reference, **stats,
    }
    for name in SCALARS:
        count = stats.get(name + '_observed_steps', 0)
        result[name + '_step_mean'] = stats.get(name + '_sum', 0.0) / count if count else None
    for name in WEIGHTS:
        count = stats.get('responses_with_weights', 0)
        result[name + '_response_mean'] = stats.get(name + '_response_mean_sum', 0.0) / count if count else None
        if not stats.get(name + '_observed_steps', 0):
            result[name + '_min'] = None
            result[name + '_max'] = None
        first_count = stats.get('responses_with_first_weights', 0)
        result[name + '_first_position_mean'] = stats.get(name + '_first_sum', 0.0) / first_count if first_count else None
    for name in ('zero_sum_steps', 'negative_wa_steps', 'abstention_largest_steps', 'prior_largest_steps', 'context_largest_steps', 'largest_weight_tie_steps'):
        denominator = stats.get('steps_with_weights', 0)
        result[name + '_fraction'] = stats.get(name, 0) / denominator if denominator else None
    for name in ('responses_any_zero_sum', 'responses_all_zero_sum', 'responses_any_negative_wa', 'responses_abstention_largest_all_steps'):
        denominator = stats['responses']
        result[name + '_fraction'] = stats.get(name, 0) / denominator if denominator else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/g203-4028/projects/knowledge-deficit-mitigation'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment must be explicit'
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    export = args.root / 'outputs/paper_20260929'
    old_audit = args.root / 'outputs/paper_core_20260930/run_20260930_core_p0/cda_trace'
    conditions = pd.read_csv(export / 'conditions.csv')
    conditions = conditions.loc[conditions.method.eq('cda_visual')].copy()
    assert len(conditions) == 20 and conditions.n.eq(2424).all()
    condition_map = conditions.set_index('condition_id').to_dict('index')
    score_columns = ('condition_id', 'sample_id', 'correct_canonical', 'correct_literal', 'abstain',
                     'uniform_reference', 'source_file_id', 'source_line', 'seed', 'score_source_line')
    scores = pd.read_parquet(export / 'scores.parquet', columns=list(score_columns))
    scores = scores.loc[scores.condition_id.isin(condition_map)].copy()
    assert len(scores) == 48480 and not scores.duplicated(['condition_id', 'sample_id']).any()
    assert scores[['correct_canonical', 'abstain', 'uniform_reference']].notna().all().all()
    assert not (scores.correct_canonical & scores.abstain).any()
    scores['model'] = scores.condition_id.map(lambda cid: condition_map[cid]['model'])
    references = pd.read_parquet(export / 'references.parquet', columns=['model', 'sample_id', 'split', 'uniform_reference'])
    references = references.loc[references.split.eq('eval')].rename(columns={'uniform_reference': 'reference_from_frozen_reference_table'})
    joined = scores.merge(references, on=['model', 'sample_id'], how='left', validate='many_to_one')
    assert joined.reference_from_frozen_reference_table.notna().all()
    assert joined.uniform_reference.eq(joined.reference_from_frozen_reference_table).all()
    sources = pd.read_csv(export / 'sources.csv').set_index('source_file_id')
    representatives = {}
    with gzip.open(old_audit / 'source_bound_trace_rows.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            key = (record['condition_id'], record['sample_id'])
            assert key not in representatives
            representatives[key] = record
    assert len(representatives) == 2020
    representative_sample_ids = {key[1] for key in representatives}
    assert len(representative_sample_ids) == 101
    locators = defaultdict(dict)
    for score in scores.to_dict('records'):
        line = int(score['source_line'])
        wanted = locators[int(score['source_file_id'])]
        assert line not in wanted
        wanted[line] = score

    groups = defaultdict(dict)
    response_rows, configuration_rows, discrepancies = [], {}, []
    missing_step_fields = Counter()
    observed_step_fields = Counter()
    source_receipts = []
    verified_representative_lines = 0
    whole_step_count = 0
    with gzip.open(args.out / 'cda_trace_steps.csv.gz', 'wt', encoding='utf-8', newline='') as full_stream, gzip.open(args.out / 'cda_trace_representative_steps.csv.gz', 'wt', encoding='utf-8', newline='') as representative_stream:
        full_writer = csv.DictWriter(full_stream, fieldnames=STEP_FIELDS)
        representative_writer = csv.DictWriter(representative_stream, fieldnames=STEP_FIELDS)
        full_writer.writeheader()
        representative_writer.writeheader()
        for source_id, wanted in sorted(locators.items()):
            path = Path(sources.loc[source_id, 'real_path'])
            assert path.is_file() and sources.loc[source_id, 'role'] == 'formal_response'
            remaining = set(wanted)
            stop = max(wanted)
            located = 0
            with gzip.open(path, 'rb') as stream:
                for number, line in enumerate(stream, 1):
                    if number > stop:
                        break
                    if number not in wanted:
                        continue
                    score = wanted[number]
                    cid = int(score['condition_id'])
                    condition = condition_map[cid]
                    raw = json.loads(line)
                    assert raw['status'] == 'ok' and raw['sample']['id'] == score['sample_id']
                    assert all(raw[name] == condition[name] for name in FIELDS)
                    assert raw['sample']['dataset'] == 'food101' and raw['sample']['split'] == 'eval'
                    assert int(raw['seed']) == int(score['seed'])
                    assert raw['config']['method'] == 'cda_visual'
                    assert raw['implementation'] == 'CDA_visual_context_adaptation'
                    assert raw['cda_equations'] == 'ACL2025_main_text_4_6_7_no_momentum'
                    key = (cid, score['sample_id'])
                    representative = key in representatives
                    state = 'A' if score['abstain'] else ('C' if score['correct_canonical'] else 'E')
                    if representative:
                        existing = representatives[key]
                        assert existing['raw_line_sha256'] == hashlib.sha256(line).hexdigest()
                        assert existing['state'] == state and existing['uniform_reference'] == bool(score['uniform_reference'])
                        assert existing['tokens'] == raw['tokens'] and existing['trace'] == raw['trace']
                        verified_representative_lines += 1
                    assert len(raw['trace']) == len(raw['tokens']) and len(raw['trace']) > 0
                    identity = {
                        **condition, 'condition_id': cid, 'dataset': 'food101', 'implementation': raw['implementation'],
                        'cda_equations': raw['cda_equations'], 'null_convention': raw['null_convention'],
                        'config_json': json.dumps(raw['config'], ensure_ascii=False, sort_keys=True),
                        'source_identity': raw['identity'], 'first_source_path': str(path), 'first_source_line': number,
                    }
                    identity_key = (cid, identity['source_identity'], identity['config_json'])
                    if identity_key not in configuration_rows:
                        configuration_rows[identity_key] = {**identity, 'responses': 0}
                    configuration_rows[identity_key]['responses'] += 1
                    stats = {
                        'responses': 1, 'steps': len(raw['trace']), 'steps_with_weights': 0,
                        'recomputed_steps': 0, 'missing_calibration_steps': 0, 'zero_sum_steps': 0,
                        'negative_wa_steps': 0, 'prior_largest_steps': 0, 'context_largest_steps': 0,
                        'abstention_largest_steps': 0, 'largest_weight_tie_steps': 0,
                        'equations_6_7_max_error': 0.0, 'weights_vector_max_error': 0.0,
                        'weights_sum_max_error': 0.0, 'equation_or_flag_discrepancy_steps': 0,
                        'h_abstention_saved_steps': 0, 'branch_logits_saved_steps': 0,
                    }
                    for name in SCALARS:
                        stats[name + '_sum'] = 0.0
                        stats[name + '_observed_steps'] = 0
                    for name in WEIGHTS:
                        stats[name + '_min'] = math.inf
                        stats[name + '_max'] = -math.inf
                    stats['responses_with_first_weights'] = 0
                    for index, step in enumerate(raw['trace']):
                        observed_step_fields.update(step.keys())
                        missing = [name for name in CALIBRATION if name not in step]
                        missing_step_fields.update(missing)
                        branch_logits_saved = any(name in step for name in ('prior_logits', 'context_logits', 'abstention_logits', 'branch_logits', 'states'))
                        event = {
                            'panel': 'full_eval', 'dataset': 'food101', 'split': 'eval', 'model': condition['model'],
                            'condition_id': cid, 'marker': condition['marker'], 'reference_marker': condition['reference_marker'],
                            'sample_id': score['sample_id'], 'state': state, 'uniform_reference': bool(score['uniform_reference']),
                            'source_file_id': source_id, 'source_line': number, 'score_source_line': int(score['score_source_line']),
                            'step_index': index, 'h_abstention_available': 'h_abstention' in step,
                            'branch_logits_saved': branch_logits_saved,
                            **{name: step.get(name) for name in ('token', *SCALARS, 'zero_sum_extension', 'negative_wa', 'calibration_status', 'log_probability')},
                        }
                        assert step['token'] == raw['tokens'][index]
                        for name in SCALARS:
                            if name in step:
                                assert math.isfinite(step[name]), (cid, score['sample_id'], index, name)
                                stats[name + '_sum'] += step[name]
                                stats[name + '_observed_steps'] += 1
                        if all(name in step for name in WEIGHTS):
                            weights = [step[name] for name in WEIGHTS]
                            stats['steps_with_weights'] += 1
                            for name, value in zip(WEIGHTS, weights):
                                stats[name + '_min'] = min(stats[name + '_min'], value)
                                stats[name + '_max'] = max(stats[name + '_max'], value)
                            largest = [branch for branch, value in zip(('prior', 'context', 'abstention'), weights) if value == max(weights)]
                            winner = largest[0] if len(largest) == 1 else 'tie'
                            stats['largest_weight_tie_steps' if winner == 'tie' else winner + '_largest_steps'] += 1
                            event['largest_weight_branch'] = winner
                            if index == 0:
                                stats['responses_with_first_weights'] = 1
                                for name in WEIGHTS:
                                    stats[name + '_first_sum'] = step[name]
                        stats['zero_sum_steps'] += step.get('zero_sum_extension') is True
                        stats['negative_wa_steps'] += step.get('negative_wa') is True
                        stats['h_abstention_saved_steps'] += 'h_abstention' in step
                        stats['branch_logits_saved_steps'] += branch_logits_saved
                        if missing:
                            stats['missing_calibration_steps'] += 1
                        else:
                            hp, hc, hnp, hnc = (step[name] for name in CALIBRATION[:4])
                            assert hnp > 0 and hnc > 0
                            rp = max(hp - hnp, 0.0) / hnp
                            rc = max(hc - hnc, 0.0) / hnc
                            total = rp + rc
                            wp, wc = (rp * rp / total, rc * rc / total) if total > 0 else (0.0, 0.0)
                            wa = 1.0 - wp - wc
                            recalculated = (rp, rc, wp, wc, wa)
                            error = max(abs(step[name] - value) for name, value in zip(('rp', 'rc', 'wp', 'wc', 'wa'), recalculated))
                            assert len(step['weights']) == 3
                            vector_error = max(abs(saved - value) for saved, value in zip(step['weights'], (wp, wc, wa)))
                            sum_error = abs(sum(step['weights']) - 1.0)
                            zero_matches = step.get('zero_sum_extension') == (total == 0.0)
                            negative_matches = step.get('negative_wa') == (wa < 0.0)
                            event.update(dict(zip(('recomputed_rp', 'recomputed_rc', 'recomputed_wp', 'recomputed_wc', 'recomputed_wa'), recalculated)))
                            event.update({
                                'equations_6_7_max_error': error, 'weights_vector_max_error': vector_error,
                                'weights_sum_error': sum_error, 'zero_sum_flag_matches': zero_matches,
                                'negative_wa_flag_matches': negative_matches,
                            })
                            stats['recomputed_steps'] += 1
                            stats['equations_6_7_max_error'] = max(stats['equations_6_7_max_error'], error)
                            stats['weights_vector_max_error'] = max(stats['weights_vector_max_error'], vector_error)
                            stats['weights_sum_max_error'] = max(stats['weights_sum_max_error'], sum_error)
                            if error > 1e-10 or vector_error > 1e-10 or not zero_matches or not negative_matches:
                                discrepancies.append({**event, 'difference_type': 'saved_equations_or_flags'})
                                stats['equation_or_flag_discrepancy_steps'] += 1
                        full_writer.writerow(event)
                        if representative:
                            representative_writer.writerow({**event, 'panel': 'representative101_per_condition'})
                    stats.update({
                        'responses_with_weights': int(stats['steps_with_weights'] > 0),
                        'responses_any_zero_sum': int(stats['zero_sum_steps'] > 0),
                        'responses_all_zero_sum': int(stats['zero_sum_steps'] == stats['steps']),
                        'responses_any_negative_wa': int(stats['negative_wa_steps'] > 0),
                        'responses_abstention_largest_all_steps': int(stats['abstention_largest_steps'] == stats['steps']),
                        'terminated_responses': int(raw['terminated']),
                    })
                    for name in WEIGHTS:
                        count = stats[name + '_observed_steps']
                        stats[name + '_response_mean_sum'] = stats[name + '_sum'] / count if count else 0.0
                    reference_group = 'positive' if score['uniform_reference'] else 'negative'
                    panels = ['full_eval'] + (['representative101_per_condition'] if representative else [])
                    for panel in panels:
                        keys = (
                            (panel, 'condition_state_reference', condition['model'], condition['marker'], state, reference_group),
                            (panel, 'model_state_reference', condition['model'], 'ALL', state, reference_group),
                            (panel, 'global_state_reference', 'ALL', 'ALL', state, reference_group),
                            (panel, 'condition_all', condition['model'], condition['marker'], 'ALL', 'ALL'),
                            (panel, 'model_all', condition['model'], 'ALL', 'ALL', 'ALL'),
                            (panel, 'global_all', 'ALL', 'ALL', 'ALL', 'ALL'),
                        )
                        for group_key in keys:
                            add_totals(groups[group_key], stats)
                    response_rows.append({
                        'dataset': 'food101', 'split': 'eval', 'model': condition['model'], 'condition_id': cid,
                        'marker': condition['marker'], 'reference_marker': condition['reference_marker'],
                        'sample_id': score['sample_id'], 'state': state, 'correct_canonical': bool(score['correct_canonical']),
                        'correct_literal': bool(score['correct_literal']), 'abstain': bool(score['abstain']),
                        'uniform_reference': bool(score['uniform_reference']), 'representative101': representative,
                        'source_file_id': source_id, 'source_line': number, 'score_source_line': int(score['score_source_line']),
                        'seed': int(score['seed']), 'source_identity': raw['identity'], 'terminated': bool(raw['terminated']),
                        'steps': stats['steps'], 'recomputed_steps': stats['recomputed_steps'],
                        'missing_calibration_steps': stats['missing_calibration_steps'],
                        'zero_sum_steps': stats['zero_sum_steps'], 'negative_wa_steps': stats['negative_wa_steps'],
                        'abstention_largest_steps': stats['abstention_largest_steps'],
                        'equations_6_7_max_error': stats['equations_6_7_max_error'],
                        'weights_vector_max_error': stats['weights_vector_max_error'],
                        'weights_sum_max_error': stats['weights_sum_max_error'],
                        **{name + '_response_mean': stats[name + '_response_mean_sum'] for name in WEIGHTS},
                        **{name + '_first': stats.get(name + '_first_sum') for name in WEIGHTS},
                    })
                    whole_step_count += stats['steps']
                    located += 1
                    remaining.remove(number)
            assert not remaining, (source_id, len(remaining))
            source_receipts.append({'source_file_id': source_id, 'source_path': str(path), 'requested_response_rows': len(wanted), 'located_response_rows': located, 'first_source_line': min(wanted), 'last_source_line': max(wanted), 'scanned_lines': stop})
            save_json(args.out / 'CURRENT_STATE.json', {'status': 'running', 'updated_utc': moment(), 'completed_sources': source_receipts, 'located_response_rows': len(response_rows), 'trace_steps': whole_step_count, 'expected_response_rows': 48480, 'GPU_initialized': False})
            print(json.dumps({'source_file_id': source_id, 'responses': located, 'cumulative_rows': len(response_rows), 'cumulative_steps': whole_step_count, 'elapsed_s': time.monotonic() - started}), flush=True)

    responses = pd.DataFrame(response_rows).sort_values(['condition_id', 'sample_id'])
    assert len(responses) == 48480 and not responses.duplicated(['condition_id', 'sample_id']).any()
    assert responses.groupby('condition_id').size().eq(2424).all()
    assert verified_representative_lines == 2020
    assert responses.loc[responses.representative101, 'steps'].sum() == 18545
    summaries = pd.DataFrame([finalize_group(key, value) for key, value in sorted(groups.items())])
    assert summaries.loc[summaries.panel.eq('full_eval') & summaries.aggregation.eq('global_all'), 'responses'].item() == 48480
    assert summaries.loc[summaries.panel.eq('representative101_per_condition') & summaries.aggregation.eq('global_all'), 'responses'].item() == 2020
    responses.to_csv(args.out / 'cda_trace_audit.csv', index=False)
    responses.to_parquet(args.out / 'cda_trace_audit.parquet', index=False)
    summaries.to_csv(args.out / 'cda_trace_summary.csv', index=False)
    pd.DataFrame(configuration_rows.values()).sort_values(['condition_id', 'source_identity']).to_csv(args.out / 'cda_config_identity.csv', index=False)
    pd.DataFrame(source_receipts).to_csv(args.out / 'sources.csv', index=False)
    pd.DataFrame(discrepancies, columns=[*STEP_FIELDS, 'difference_type']).to_csv(args.out / 'cda_execution_differences.csv', index=False)
    full = groups[('full_eval', 'global_all', 'ALL', 'ALL', 'ALL', 'ALL')]
    receipt = {
        'created_utc': moment(), 'elapsed_s': time.monotonic() - started,
        'status': 'complete', 'scope': 'existing frozen core-five Food eval CDA visual adaptation; twenty registered conditions',
        'response_rows': len(responses), 'conditions': len(conditions), 'inputs_per_condition': 2424,
        'unique_model_inputs': int(responses[['model', 'sample_id']].drop_duplicates().shape[0]),
        'trace_steps': whole_step_count, 'representative_response_rows': verified_representative_lines,
        'representative_trace_steps': int(responses.loc[responses.representative101, 'steps'].sum()),
        'recomputed_steps': full['recomputed_steps'], 'missing_calibration_steps': full['missing_calibration_steps'],
        'missing_step_fields': dict(missing_step_fields), 'observed_step_fields': dict(observed_step_fields),
        'equations_6_7_max_error': full['equations_6_7_max_error'],
        'weights_vector_max_error': full['weights_vector_max_error'],
        'weights_sum_max_error': full['weights_sum_max_error'],
        'equation_or_flag_discrepancy_steps': full['equation_or_flag_discrepancy_steps'],
        'zero_sum_steps': full['zero_sum_steps'], 'negative_wa_steps': full['negative_wa_steps'],
        'h_abstention_saved_steps': full['h_abstention_saved_steps'],
        'branch_logits_saved_steps': full['branch_logits_saved_steps'],
        'equation4_vector_recomputation': 'unavailable: per-step prior/context/abstention branch logit vectors were not saved',
        'weight_sum_is_equation4_check': False,
        'entropy_difference_direction': 'max(H_branch - H_null_branch, 0) / H_null_branch',
        'zero_sum_rule': 'wp=wc=0, wa=1 continuous extension; source flags retained',
        'null_convention': 'same_prefix_text_placeholder_and_uniform_image',
        'state_definition': 'A if frozen abstain; C if frozen canonical correct and not abstain; E otherwise',
        'reference_definition': 'frozen uniform_reference, verified against references.parquet by model/sample_id',
        'aggregation_note': 'Four phrase conditions are distinct runs; token-weighted and equal-response-weighted means are separate.',
        'new_generations': 0, 'new_API_calls': 0, 'GPU_initialized': False,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'git_head_at_execution': subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip(),
        'cda_source_sha256': digest(args.root / 'src/kdm/cda.py'),
        'pipeline_source_sha256': digest(args.root / 'src/kdm/pipeline.py'),
        'audit_script_sha256': digest(Path(__file__)),
        'inputs': {
            'conditions': str(export / 'conditions.csv'), 'scores': str(export / 'scores.parquet'),
            'references': str(export / 'references.parquet'), 'sources': str(export / 'sources.csv'),
            'registered_representative_trace': str(old_audit / 'source_bound_trace_rows.jsonl.gz'),
        },
    }
    save_json(args.out / 'validation.json', receipt)
    save_json(args.out / 'CURRENT_STATE.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
