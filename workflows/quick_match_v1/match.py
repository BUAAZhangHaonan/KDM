#!/usr/bin/env python3
"""Conservative CPU-only lexical screening. Never a semantic judge or final GT."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.scoring import normalize, lexical_label, vqa_score
from kdm.models.official_vqa_normalizer import VQAEval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Matcher:
    def __init__(self):
        self.aliases = json.loads((ROOT / 'configs/kdm/food_aliases.json').read_text())
        self.names = set(self.aliases)
        self.by_text = {}
        for label, names in self.aliases.items():
            for name in names:
                self.by_text.setdefault(normalize(name), set()).add(label)
        evaluator = VQAEval(None, None)
        self.vqa_normalize = lambda text: evaluator.processDigitArticle(evaluator.processPunctuation(text))

    def classify(self, row):
        sample = row['sample']
        result = {'screening_label': 'needs_confirmation', 'preliminary_score': None,
                  'reason': 'no_exact_match', 'final_gt': False, 'human_reviewed': False}
        if 'candidate_scores' in row:
            scores = row['candidate_scores']
            if sample['dataset'] != 'food101' or len(scores) != 101 or {s['label'] for s in scores} != self.names:
                raise ValueError('Closed record must contain each of the 101 Food classes exactly once')
            if row.get('ranking_rule') != 'mean_log_probability':
                raise ValueError('Unknown closed ranking rule')
            if any(not math.isfinite(s['mean_logp']) or not math.isfinite(s['sum_logp']) for s in scores):
                raise ValueError('Nonfinite score')
            highest = max(s['mean_logp'] for s in scores)
            top = [s['label'] for s in scores if s['mean_logp'] == highest]
            gold = sample['class']
            target = next(s for s in scores if s['label'] == gold)
            rank = 1 + sum(s['mean_logp'] > target['mean_logp'] for s in scores)
            if row.get('gold_rank') != rank or row.get('target') != gold:
                raise ValueError('Stored closed rank or target inconsistent with scores')
            result.update(top_labels=top, gold_rank=rank, reason='closed_unique_top1' if len(top) == 1 else 'closed_tied_top1')
            if len(top) == 1:
                correct = top[0] == gold
                result.update(screening_label='correct' if correct else 'incorrect', preliminary_score=float(correct))
            return result
        text = row['text']
        literal = lexical_label(text)
        if literal == 'abstain':
            result.update(screening_label='abstain', preliminary_score=0.0, reason='frozen_exact_abstention')
            return result
        if literal == 'invalid':
            result['reason'] = 'empty_answer'
            return result
        if sample['dataset'] == 'food101':
            labels = self.by_text.get(normalize(text), set())
            if len(labels) == 1:
                correct = sample['class'] in labels
                result.update(screening_label='correct' if correct else 'incorrect',
                              preliminary_score=float(correct), matched_class=next(iter(labels)), reason='frozen_full_answer_alias')
        elif sample['dataset'] == 'vizwiz':
            score = float(vqa_score(text, sample['gold'], self.vqa_normalize))
            if score > 0:
                result.update(screening_label='correct' if score == 1 else 'partial_match', preliminary_score=score,
                              reason='official_normalized_full_answer_match')
        else:
            raise ValueError('Unknown dataset')
        return result


def run(inputs, out, limit=0):
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be inside project outputs/')
    out.mkdir(parents=True, exist_ok=False)
    matcher = Matcher()
    counts, groups, sources, seen = Counter(), {}, [], set()
    with (out / 'labels.partial.jsonl').open('x') as sink:
        for path in inputs:
            path = path.resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Input must be within project root')
            before = path.stat()
            digest, n = hashlib.sha256(), 0
            with path.open('rb') as source:
                for raw in source:
                    if limit and n >= limit:
                        break
                    row = json.loads(raw)
                    key = (row['model'], row['key'])
                    if key in seen:
                        raise ValueError('Duplicate source key')
                    seen.add(key)
                    if not row.get('identity'):
                        raise ValueError('Missing original record identity')
                    result = matcher.classify(row)
                    kind = 'closed' if 'candidate_scores' in row else row.get('kind', 'response')
                    value = {'schema': 'quick_match_v1', 'model': row['model'], 'key': row['key'],
                             'sample_id': row['sample']['id'], 'dataset': row['sample']['dataset'],
                             'split': row['sample']['split'], 'kind': kind, 'guided': row.get('guided'),
                             'replicate': row.get('replicate'), 'source_identity': row['identity'],
                             'source_path': str(path.relative_to(ROOT)), 'source_line': n + 1,
                             'source_row_sha256': hashlib.sha256(raw).hexdigest(), **result}
                    sink.write(json.dumps(value, ensure_ascii=False) + '\n')
                    counts[result['screening_label']] += 1
                    group = '|'.join(map(str, [row['model'], value['dataset'], value['split'], kind, value['guided']]))
                    groups.setdefault(group, Counter())[result['screening_label']] += 1
                    digest.update(raw)
                    n += 1
            after = path.stat()
            if not limit and (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Input changed while screening; partial output is not accepted')
            sources.append({'path': str(path.relative_to(ROOT)), 'screened_rows': n,
                            'screened_bytes_sha256': digest.hexdigest(), 'limited_snapshot': bool(limit),
                            'source_size_at_start': before.st_size})
    (out / 'labels.partial.jsonl').rename(out / 'labels.jsonl')
    report = {'schema': 'quick_match_v1', 'automatic_preliminary_only': True, 'final_gt': False,
              'api_requests': 0, 'gpu_used': False, 'human_reviewed': False, 'rows': sum(counts.values()),
              'counts': counts, 'groups': groups, 'sources': sources,
              'rule_files_sha256': {p: sha(ROOT / p) for p in ['workflows/quick_match_v1/match.py',
                  'configs/kdm/food_aliases.json', 'src/kdm/scoring.py', 'src/kdm/models/official_vqa_normalizer.py']},
              'limits': 'Full-string lexical matches only. Nonmatches remain unresolved, not wrong. Closed unique top1 accuracy is not an abstention GT rule. Ties unresolved. No semantic extraction, no human review, no paid API.'}
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'out': str(out), 'rows': report['rows'], 'counts': counts}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='At most N rows per input for schema smoke checks; 0 means full files')
    a = p.parse_args()
    if a.limit < 0:
        p.error('--limit must be nonnegative')
    run(a.input, a.out, a.limit)
