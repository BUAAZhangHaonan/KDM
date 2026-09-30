#!/usr/bin/env python3
"""Score the sealed 505 natural noisy-image answers using accepted QA scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, within
from workflows.main_results import score as frozen
from workflows.paper_core.score_native import load_references, save_rows
from workflows.supplemental.remaining11.score import load_authority, infer_qa, rows, score_target

MODELS = ('qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b')
AUTHORITY = 'outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--authority-manifest', default=AUTHORITY)
    parser.add_argument('--decision-file', action='append', default=[])
    args = parser.parse_args()
    source = within(ROOT, args.input_dir)
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    samples = {row['id']: row for _, row, _ in rows(ROOT / 'data/current/all.jsonl')
               if row['dataset'] == 'food101' and row['split'] == 'eval'}
    patterns = frozen.compile_classes(sorted({row['class'] for row in samples.values()}))
    _, references = load_references()
    manifest_path = within(ROOT, args.authority_manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest = {'census_final_labels': manifest['census_final_labels'], 'historical_labels': []}
    decision_paths = [within(ROOT, path) for path in args.decision_file]
    reviews, behavior, behavior_sources, _, decisions = load_authority(source, manifest, decision_paths)
    scores, pending, cache, source_receipts = [], {}, {}, []
    counts = Counter()
    for model in MODELS:
        raw = source / model / 'natural_reference.events.jsonl'
        meta = json.loads(raw.with_suffix('.identity.json').read_text(encoding='utf-8'))
        if stable_hash(meta['definition']) != meta['identity']:
            raise ValueError('Natural-reference ledger identity changed')
        seen, classes = set(), Counter()
        source_receipts.append({'model': model, 'path': str(raw.relative_to(ROOT)),
                                'sha256': file_hash(raw),
                                'identity_sha256': file_hash(raw.with_suffix('.identity.json'))})
        for line, row, line_sha in rows(raw):
            if (row['identity'] != meta['identity'] or row['status'] != 'ok'
                    or row['model'] != model or row['sample_id'] in seen):
                raise ValueError('Natural-reference source identity/status/key differs')
            sid = row['sample_id']
            sample = samples[sid]
            if row['clean_source']['target_class'] != sample['class']:
                raise ValueError('Natural-reference original target identity differs')
            seen.add(sid)
            classes[sample['class']] += 1
            generated = row['noise_generated_r']
            if generated['status'] != 'ok':
                raise ValueError('Natural-reference actual generation failed')
            answer = generated['text']
            qkey = frozen.qah(sample['question'], answer)
            if qkey not in cache:
                cache[qkey] = infer_qa(sample['question'], answer, patterns, reviews, behavior, decisions)
            inferred = cache[qkey]
            canonical, literal, reason = score_target(answer, sample['class'], inferred, patterns)
            record = {'model': model, 'dataset': 'food101', 'split': 'eval', 'sample_id': sid,
                      'method': 'direct', 'kind': 'natural_noisy_reference', 'main_marker': 'NONE',
                      'reference_marker': 'NONE', 'guided': False, 'reference_guided': False,
                      'noise_step': 500, 'seed': row['seed'], 'qa_key': qkey,
                      'question': sample['question'], 'answer': answer, 'target_class': sample['class'],
                      'canonical_name_in_primary_score': canonical, 'literal_extracted_name_score': literal,
                      'abstain': inferred['abstain'], 'uniform_reference': references[model, sid],
                      'score_reason': reason, 'behavior_source': inferred['behavior_source'],
                      'tokens': generated['tokens'], 'terminated': generated['terminated'],
                      'source': {'path': str(raw.relative_to(ROOT)), 'line': line, 'line_sha256': line_sha,
                                 'identity': row['identity'], 'key': row['key']},
                      'accepted_decision_path': inferred['decision'].get('decision_source_path')
                                                if inferred['decision'] else None}
            scores.append(record)
            counts[model] += 1
            if canonical is None or literal is None or inferred['abstain'] is None or inferred['name_boundary']:
                item = pending.setdefault(qkey, {'qa_key': qkey, 'question': sample['question'],
                                                'answer': answer, 'source_members': []})
                item['source_members'].append({'model': model, 'sample_id': sid, 'source': record['source']})
        if len(seen) != 101 or len(classes) != 101 or set(classes.values()) != {1}:
            raise ValueError('Natural-reference representative panel differs from 101-class quota')
    save_rows(output / 'score_rows.jsonl.gz', scores)
    save_rows(output / 'pending_QA.jsonl', list(pending.values()))
    import pandas as pd
    pd.DataFrame(scores).drop(columns=['source', 'tokens']).to_parquet(output / 'new_scores.parquet', index=False)
    metrics = []
    for model in MODELS:
        model_rows = [row for row in scores if row['model'] == model]
        a = sum(row['abstain'] is True for row in model_rows)
        r = sum(row['uniform_reference'] for row in model_rows)
        tp = sum(row['abstain'] is True and row['uniform_reference'] for row in model_rows)
        primary_complete = all(row['canonical_name_in_primary_score'] is not None for row in model_rows)
        behavior_complete = all(row['abstain'] is not None for row in model_rows)
        c = sum(row['canonical_name_in_primary_score'] == 1 for row in model_rows)
        metrics.append({'model': model, 'n': 101, 'scope': 'representative101_natural_reference',
                        'correct': c, 'abstentions': a, 'tp': tp, 'fp': a - tp, 'reference_positive': r,
                        'accuracy': c / 101 if primary_complete else None,
                        'precision': tp / a if behavior_complete and a else None,
                        'precision_null_reason': 'no_abstentions' if not a else None,
                        'recall': tp / r if behavior_complete and r else None,
                        'canonical_pending': sum(row['canonical_name_in_primary_score'] is None for row in model_rows),
                        'literal_pending': sum(row['literal_extracted_name_score'] is None for row in model_rows),
                        'abstain_pending': sum(row['abstain'] is None for row in model_rows)})
    pd.DataFrame(metrics).to_csv(output / 'natural_reference_metrics.csv', index=False)
    receipt = {'schema': 'kdm_attribution_reference_scoring_v1', 'rows': len(scores),
               'models': dict(counts), 'boundary_QA': len(pending),
               'canonical_pending': sum(row['canonical_name_in_primary_score'] is None for row in scores),
               'literal_pending': sum(row['literal_extracted_name_score'] is None for row in scores),
               'abstain_pending': sum(row['abstain'] is None for row in scores),
               'sources': source_receipts, 'decision_files': [{'path': str(p.relative_to(ROOT)),
                                                             'sha256': file_hash(p)} for p in decision_paths],
               'authority_manifest_sha256': file_hash(manifest_path),
               'scorer_sha256': file_hash(Path(__file__)), 'GPU_initialized': False,
               'new_generations': 0, 'new_API_calls': 0}
    atomic_json(output / 'scoring_receipt.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {'sources', 'decision_files'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
