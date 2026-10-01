"""Prepare disjoint, complete-QA Luna payloads; generate no semantic decisions.

The caller does not expose the annotation model's tokenizer or a context-size
switch. UTF-8 payload bytes are recorded as a conservative content bound, not
misreported as measured model tokens or as a changed model context setting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import file_hash, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows


def write_rows(path, records):
    with path.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-dir', required=True)
    parser.add_argument('--exclude-queue', action='append', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-rows', type=int, default=250)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--content-byte-budget', type=int, default=28672)
    parser.add_argument('--answer-population', choices=('short', 'long'), default='short')
    args = parser.parse_args()
    if not 1 <= args.max_rows <= 250 or not 1 <= args.batches <= 3 or not 4096 <= args.content_byte_budget <= 28672:
        raise ValueError('The finite row/parallel/content budget differs')
    score_dir, output = (within(ROOT, value) for value in (args.score_dir, args.output))
    output.relative_to(ROOT / 'outputs/supplemental/remaining4')
    source = score_dir / 'viz_pending_complete_QA.jsonl'
    receipt_path = score_dir / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    source_sha = file_hash(source)
    if not receipt['passed'] or source_sha != receipt['outputs'][source.name]:
        raise ValueError('The immutable pending QA source differs from its receipt')
    excluded, exclusions = set(), []
    for relative in args.exclude_queue:
        path = within(ROOT, relative)
        actual = list(rows(path))
        for _, row, _ in actual:
            if frozen.qah(row['question'], row['answer']) != row['qa_key']:
                raise ValueError('An owned QA key differs from its complete input')
            excluded.add(row['qa_key'])
        exclusions.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path), 'rows': len(actual)})
    candidates = []
    for line, row, line_sha in rows(source):
        if frozen.qah(row['question'], row['answer']) != row['qa_key']:
            raise ValueError('A pending QA key differs from its complete input')
        if row['qa_key'] in excluded:
            continue
        # Length selection changes no text, label, score or GT. Long responses
        # use their own measured batches; they are never truncated to fit.
        is_short = len(row['answer']) <= 64 and '\n' not in row['answer']
        if is_short != (args.answer_population == 'short'):
            continue
        candidates.append((row['qa_key'], line, line_sha, row))
    candidates.sort(key=lambda item: item[0])
    output.mkdir(parents=True, exist_ok=False)
    cursor, assigned, batches = 0, set(), []
    for batch in range(args.batches):
        payload, owner_rows, byte_count = [], [], 0
        while cursor < len(candidates) and len(payload) < args.max_rows:
            qa, source_line, line_sha, row = candidates[cursor]
            message = {'i': len(payload), 'q': row['question'], 'r': row['answer']}
            cost = len((json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8'))
            if byte_count + cost > args.content_byte_budget:
                break
            if qa in assigned:
                raise ValueError('Parallel batches repeat a QA key')
            assigned.add(qa)
            payload.append(message)
            owner_rows.append({'candidate_index': message['i'], 'qa_key': qa,
                'question': row['question'], 'answer': row['answer'],
                'source_queue_path': str(source.relative_to(ROOT)), 'source_queue_sha256': source_sha,
                'source_queue_line': source_line, 'source_queue_line_sha256': line_sha,
                'source_memberships': row['source_memberships']})
            cursor += 1
            byte_count += cost
        if not payload:
            raise ValueError('No complete QA fits the remaining finite content budget')
        directory = output / ('batch_' + str(batch))
        directory.mkdir()
        message_path, owners_path = directory / 'full_QA_payload.jsonl', directory / 'owned_complete_QA.jsonl'
        write_rows(message_path, payload)
        write_rows(owners_path, owner_rows)
        if message_path.stat().st_size != byte_count:
            raise ValueError('The written full-QA content differs from its budget')
        batches.append({'batch': batch, 'rows': len(payload), 'payload_utf8_bytes': byte_count,
            'payload': str(message_path.relative_to(ROOT)), 'payload_sha256': file_hash(message_path),
            'owner_queue': str(owners_path.relative_to(ROOT)), 'owner_queue_sha256': file_hash(owners_path)})
    result = {'passed': True, 'input_pending_unique_QA': receipt['pending_unique_Viz_QA'],
        'source': str(source.relative_to(ROOT)), 'source_sha256': source_sha,
        'source_receipt_sha256': file_hash(receipt_path), 'batches': batches,
        'assigned_unique_QA': len(assigned), 'exclusions': exclusions,
        'answer_population': args.answer_population,
        'unowned_population_QA_available': len(candidates),
        'input_content_byte_budget': args.content_byte_budget,
        'exact_annotation_model_tokens_measured': False,
        'annotation_context_size_changed': False, 'questions_or_answers_truncated': 0,
        'semantic_decisions_generated': 0, 'GT_exposed_in_model_payload': False,
        'GPU_initialized': False, 'preparation_elapsed_seconds': perf_counter() - started,
        'runner_sha256': file_hash(Path(__file__)),
        'created_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
