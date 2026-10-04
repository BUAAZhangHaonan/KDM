"""Display a bounded review chunk in full and record exactly what was emitted."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--start', required=True, type=int, help='One-based first row')
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    assert args.start >= 1 and 1 <= args.count <= 4
    source = args.batch/'pending_review.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line]
    chosen = rows[args.start-1:args.start-1+args.count]
    assert chosen, 'Start is beyond this assigned batch'
    fields = ('qa_key', 'quality_key', 'question', 'options', 'answer', 'gt_answer_details')
    rendered = '\n'.join(json.dumps({k: row[k] for k in fields if k in row}, ensure_ascii=False)
                         for row in chosen)
    assert len(rendered) <= 24000, 'Reduce count; never truncate source text'
    print(rendered, flush=True)
    receipt = {'at_utc': datetime.now(timezone.utc).isoformat(), 'start': args.start,
               'rows': len(chosen), 'keys': [row.get('qa_key', row.get('quality_key')) for row in chosen],
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'emitted_characters': len(rendered),
               'emitted_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
               'meaning': 'Complete fields emitted; this does not attest semantic judgment quality'}
    with (args.batch/'full_display_receipts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt)+'\n')


if __name__ == '__main__':
    main()
