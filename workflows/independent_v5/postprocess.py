#!/usr/bin/env python3
"""Verify a completed Food independent cohort and produce free, preliminary labels.

No model loading, GPU calls, paid APIs or semantic/human-judgment claims. Inputs
remain immutable. A fresh output directory is mandatory; failures retain partial
evidence. Unrecognized text never becomes an incorrect answer by default.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kdm.decoding import DecodeConfig
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from kdm.scoring import lexical_label, normalize
from workflows.quick_match_v1.match import Matcher

CONFIG = asdict(DecodeConfig(temperature=1.0, top_p=1.0))
LABELS = ('correct', 'incorrect', 'abstain', 'invalid', 'unresolved')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def task(sample, replicate):
    return dict(sample=sample, method='direct', marker='UNKNOWN', reference_marker='UNKNOWN',
                guided=False, reference_guided=False, attempt=True, replicate=replicate,
                kind='independent_attempt')


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect_attempt(row, model, samples, identity, eos):
    """Structural validation separate from text scoring; malformed records fail closed."""
    sid = row['sample']['id']
    require(sid in samples and row['sample'] == samples[sid], 'Sample differs from frozen manifest')
    rep = row['replicate']
    require(type(rep) is int and 0 <= rep < 10, 'Replicate must be an integer in 0..9')
    expected = task(samples[sid], rep)
    require(all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in expected.items()),
            'Wrong independent task fields')
    require(row['model'] == model and row['identity'] == identity and row['status'] == 'ok',
            'Wrong model, row identity or generation status')
    require(row['key'] == task_id(model, expected), 'Incorrect task key')
    require(row['config'] == CONFIG and row['seed'] == stable_seed(sid, model, rep), 'Config or seed mismatch')
    require(row['prompt'] == task_prompt(samples[sid]['question'], guided=False, attempt=True), 'Prompt mismatch')
    require(isinstance(row['text'], str), 'Answer must be text')
    tokens, logps = row['tokens'], row['selected_log_probabilities']
    require(isinstance(tokens, list) and 1 <= len(tokens) <= 32 and all(type(t) is int and t >= 0 for t in tokens), 'Invalid token sequence')
    require(len(logps) == len(tokens) and all(finite_number(p) and p <= 1e-6 for p in logps), 'Invalid selected log probabilities')
    require(not any(t in eos for t in tokens[:-1]), 'EOS before sequence end')
    terminated = tokens[-1] in eos
    require(type(row['terminated']) is bool and row['terminated'] == terminated and (terminated or len(tokens) == 32), 'EOS/termination mismatch')
    require(finite_number(row['sequence_log_probability']) and math.isclose(row['sequence_log_probability'], sum(logps), rel_tol=1e-8, abs_tol=1e-8), 'Sequence probability mismatch')
    require(finite_number(row['first_probability']) and math.isclose(row['first_probability'], math.exp(logps[0]), rel_tol=1e-8, abs_tol=1e-10), 'First probability mismatch')
    return sid, rep


def label_attempt(matcher, row):
    result = matcher.classify(row)
    # Frozen lexical_label/analysis explicitly gives empty invalid output 0 credit.
    # This is narrower than semantic invalidity; nonempty unmatched text stays unknown.
    if lexical_label(row['text']) == 'invalid':
        result.update(screening_label='invalid', preliminary_score=0.0, reason='frozen_empty_invalid')
    if result['screening_label'] == 'needs_confirmation':
        result.update(screening_label='unresolved', preliminary_score=None)
    require(result['screening_label'] in LABELS, 'Unexpected Food screening label')
    return result


def question_summary(model, sample, attempts, gold_rank):
    """Apply frozen conjunction with three-valued correctness, never impute unknowns."""
    require(set(attempts) == set(range(10)), 'Question lacks all ten unique attempts')
    counts = Counter(attempts[r]['screening_label'] for r in range(10))
    correct, unresolved = counts['correct'], counts['unresolved']
    if correct:
        state, warrant = 'independent_correct_answer_observed', False
    elif gold_rank == 1:
        state, warrant = 'closed_rank_one_no_joint_support', False
    elif unresolved:
        state, warrant = 'unresolved_independent_answers', None
    else:
        state, warrant = 'joint_deficit_supported_by_free_screening', True
    return dict(schema='kdm_independent_food_free_probe_v1', model=model,
                sample_id=sample['id'], split=sample['split'], cluster=sample['cluster'],
                attempts=10, counts={k: counts[k] for k in LABELS},
                independent_any_correct=True if correct else (None if unresolved else False),
                mean_correctness=None if unresolved else correct / 10,
                mean_correctness_bounds=[correct / 10, (correct + unresolved) / 10],
                gold_rank=gold_rank, preliminary_behavioral_warrant=warrant,
                evidence_state=state, automatic_preliminary_only=True,
                final_gt=False, human_reviewed=False)


def checked_closed(record, model, samples, matcher):
    """Require existing completed verification receipt, then independently rescan all scores."""
    done = load(record / 'complete.json')
    progress = load(record / 'progress.json')
    receipt_path = record / 'completed_verification_20260923.json'
    receipt = load(receipt_path)
    raw = within(ROOT, progress['output'])
    side_path = raw.with_suffix('.identity.json')
    side = load(side_path)
    definition = side['definition']
    digest = file_hash(raw)
    require(done.get('complete') is True and done.get('closed') == 4848 and progress.get('status') == 'complete', 'Closed cohort incomplete')
    require(definition['model'] == model and definition['schema'] == 'kdm_vllm_tree_closed_v4' and definition['manifest_sha256'] == file_hash(ROOT / 'data/current/all.jsonl'), 'Closed cohort model/manifest/schema mismatch')
    require(done['identity'] == definition and side['identity'] == stable_hash(definition), 'Closed sidecar mismatch')
    require(done['output_sha256'] == digest == receipt['raw_sha256'] and receipt.get('verified') is True
            and receipt['model'] == model and receipt['rows'] == 4848 and receipt['source_identity'] == side['identity']
            and receipt['sidecar_sha256'] == file_hash(side_path), 'Closed completion/verification receipt mismatch')
    ranks = {}
    for row in read_jsonl(raw):
        sid = row['sample']['id']
        require(sid not in ranks and row['sample'] == samples.get(sid) and row['model'] == model and row['identity'] == side['identity'] and row['status'] == 'ok', 'Closed row identity/coverage mismatch')
        require(row['key'] == stable_hash([model, sid, 'closed']), 'Closed key mismatch')
        scored = matcher.classify(row)  # Verifies all 101 classes, finite scores and exact frozen rank.
        for score in row['candidate_scores']:
            n = score['n_tokens']
            require(type(n) is int and n > 0 and math.isclose(score['sum_logp'] / n, score['mean_logp'], rel_tol=1e-8, abs_tol=1e-8), 'Closed length normalization mismatch')
        ranks[sid] = scored['gold_rank']
    require(set(ranks) == set(samples) and file_hash(raw) == digest, 'Closed coverage incomplete or raw changed')
    return ranks, dict(record=str(record.relative_to(ROOT)), raw=str(raw.relative_to(ROOT)), raw_sha256=digest,
                       sidecar_sha256=file_hash(side_path), receipt_sha256=file_hash(receipt_path), identity=side['identity'])


def run(record, closed_record, out):
    record, closed_record, out = (within(ROOT, p) for p in (record, closed_record, out))
    require(out.is_relative_to(ROOT / 'outputs'), 'Output must be inside project outputs/')
    out.mkdir(parents=True, exist_ok=False)
    try:
        samples = {s['id']: s for s in read_jsonl(ROOT / 'data/current/all.jsonl') if s['dataset'] == 'food101'}
        require(len(samples) == 4848 and Counter(s['split'] for s in samples.values()) == {'dev': 2424, 'eval': 2424}, 'Full original Food manifest required')
        done, progress = load(record / 'complete.json'), load(record / 'progress.json')
        raw = within(ROOT, progress['output'])
        side_path = raw.with_suffix('.identity.json')
        side = load(side_path)
        definition, identity = side['definition'], side['identity']
        model = definition['model']
        require(done.get('generation_complete') is True and done['independent'] == 48480
                and progress['status'] == 'generation_complete' and progress['independent'] == 48480, 'Generation not complete')
        require(done['identity'] == identity == stable_hash(definition), 'Independent sidecar identity mismatch')
        require(definition['schema'] == 'kdm_vllm_independent_food_v1' and definition['repeats'] == 10
                and definition['base_config'] == CONFIG and definition['manifest_sha256'] == file_hash(ROOT / 'data/current/all.jsonl'), 'Independent protocol mismatch')
        eos = definition['eos_token_ids']
        require(isinstance(eos, list) and eos and all(type(t) is int and t >= 0 for t in eos), 'Native EOS IDs required')
        matcher = Matcher()
        ranks, closed_proof = checked_closed(closed_record, model, samples, matcher)
        bound_closed = definition['closed_completion']
        require(bound_closed['raw_sha256'] == closed_proof['raw_sha256'] and bound_closed['identity'] == load(within(ROOT, load(closed_record / 'progress.json')['output']).with_suffix('.identity.json'))['definition'], 'Independent run bound a different closed cohort')
        attempts, seen, seeds = {sid: {} for sid in samples}, set(), {sid: set() for sid in samples}
        counts, digest = Counter(), hashlib.sha256()
        with raw.open('rb') as source, (out / 'labels.partial.jsonl').open('x') as sink:
            for line, content in enumerate(source, 1):
                require(content.endswith(b'\n'), 'Partial raw record')
                row = json.loads(content)
                sid, rep = inspect_attempt(row, model, samples, identity, eos)
                require(row['key'] not in seen and rep not in attempts[sid] and row['seed'] not in seeds[sid], 'Duplicate key/replicate/seed')
                seen.add(row['key']); seeds[sid].add(row['seed'])
                label = label_attempt(matcher, row)
                attempts[sid][rep] = label
                counts[label['screening_label']] += 1
                value = dict(schema='kdm_independent_food_free_label_v1', model=model, sample_id=sid,
                             split=samples[sid]['split'], replicate=rep, seed=row['seed'], key=row['key'],
                             text=row['text'], source_identity=identity, source_line=line,
                             source_row_sha256=hashlib.sha256(content).hexdigest(), **label)
                sink.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                digest.update(content)
        require(len(seen) == 48480 and digest.hexdigest() == done['output_sha256'] == file_hash(raw), 'Incomplete coverage or changed independent raw')
        probes = [question_summary(model, s, attempts[sid], ranks[sid]) for sid, s in samples.items()]
        require(side == load(side_path) and done == load(record / 'complete.json'), 'Completion/sidecar changed during processing')
        with (out / 'questions.jsonl').open('x') as sink:
            for probe in probes:
                sink.write(json.dumps(probe, ensure_ascii=False, allow_nan=False) + '\n')
        (out / 'labels.partial.jsonl').rename(out / 'labels.jsonl')
        rules = ['workflows/independent_v5/postprocess.py', 'workflows/quick_match_v1/match.py',
                 'configs/kdm/food_aliases.json', 'src/kdm/scoring.py', 'src/kdm/analysis.py', 'src/kdm/reports.py']
        summary = dict(schema='kdm_independent_food_free_summary_v1', model=model,
                       generated_answers=48480, screened_answers=48480, counts={k: counts[k] for k in LABELS},
                       questions=4848, question_states=dict(Counter(p['evidence_state'] for p in probes)),
                       per_split={split: dict(Counter(p['evidence_state'] for p in probes if p['split'] == split)) for split in ('dev', 'eval')},
                       source=dict(record=str(record.relative_to(ROOT)), raw=str(raw.relative_to(ROOT)),
                                   raw_sha256=digest.hexdigest(), identity=identity, sidecar_sha256=file_hash(side_path),
                                   complete_sha256=file_hash(record / 'complete.json')), closed_source=closed_proof,
                       rules_sha256={p: file_hash(ROOT / p) for p in rules},
                       outputs_sha256={p: file_hash(out / p) for p in ('labels.jsonl', 'questions.jsonl')},
                       rule='Food: ten attempts mean correctness = 0 AND original mean-logprob gold rank > 1',
                       final_gt=False, automatic_preliminary_only=True, human_reviewed=False, api_requests=0, gpu_used=False,
                       limits='Whole-answer aliases and exact lexical markers only. Nonmatches stay unresolved. Empty invalid answers receive frozen zero credit. Rank-one ties use the frozen strict-greater rank rule. A negative joint warrant is not a claim of general answerability. Separate vLLM cohort; no NumPy random-draw equivalence. No semantic or human review claimed.')
        atomic_json(out / 'summary.json', summary)
        print(json.dumps({'out': str(out), 'model': model, 'answers': 48480, 'counts': summary['counts'], 'question_states': summary['question_states']}))
        return summary
    except BaseException as exc:
        atomic_json(out / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc), 'accepted': False, 'final_gt': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--closed-record', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.record, args.closed_record, args.out)
