"""Verify complete Qwen35 shards; reuse existing stage labels; match matrix only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workflows/acceleration_v4'))
import formal_postprocess as post
from kdm.io import atomic_json, file_hash

def main():
    sources = [Path(f'outputs/records/acceleration_v4/panel5_food_formal_4029_v1/qwen35_4b/shard_{i:03d}_of_004') for i in range(4)]
    accepted, coverage = post.validate_completed_shards(sources)
    assert coverage['qwen35_4b']['full_model_generation_present']
    reused, reuse_evidence = {}, []
    for stage in ('unknown_main', 'unknown_controls'):
        base = ROOT / f'outputs/annotations/acceleration_v4/qwen35_4029_{stage}_20260923_v1'
        summary = json.loads((base / 'summary.json').read_text())
        assert summary['rows'] == 12120 and summary['completed_stage'] == stage
        assert summary['stage_complete_for_input_shards'] is True
        count = 0
        for line in (base / 'labels.jsonl').open():
            row = json.loads(line)
            assert row['model'] == 'qwen35_4b' and row['stage'] == stage
            assert row['key'] not in reused and not row['final_gt'] and not row['human_reviewed']
            reused[row['key']] = row
            count += 1
        assert count == 12120
        reuse_evidence.append({'stage': stage, 'rows': count, 'labels_path': str((base / 'labels.jsonl').relative_to(ROOT)), 'labels_sha256': file_hash(base / 'labels.jsonl'), 'summary_sha256': file_hash(base / 'summary.json')})
    out = ROOT / 'outputs/annotations/acceleration_v4/qwen35_4029_complete_20260923_v1'
    out.mkdir(exist_ok=False)
    matcher = post.quick.Matcher()
    counts, groups, stages = Counter(), defaultdict(Counter), defaultdict(Counter)
    seen, used, new_rows = set(), set(), 0
    try:
        with (out / 'labels.partial.jsonl').open('x') as sink:
            for source in accepted:
                raw = ROOT / source['raw_path']
                digest = hashlib.sha256()
                for index, line in enumerate(raw.open('rb'), 1):
                    row = json.loads(line)
                    key = row['key']
                    assert key not in seen
                    seen.add(key)
                    group = post.condition(row)
                    linehash = hashlib.sha256(line).hexdigest()
                    if group[1] in ('unknown_main', 'unknown_controls'):
                        value = reused[key]
                        assert value['source_path'] == source['raw_path'] and value['source_line'] == index
                        assert value['source_row_sha256'] == linehash and value['source_identity'] == row['identity']
                        assert value['sample_id'] == row['sample']['id']
                        assert all(value[field] == row[field] for field in post.formal.FIELDS)
                        used.add(key)
                    else:
                        assert group[1] == 'prompt_matrix'
                        result = matcher.classify(row)
                        assert not result['final_gt'] and not result['human_reviewed']
                        value = {'schema': 'formal_v4_quick_match_v1', 'model': row['model'], 'key': key, 'sample_id': row['sample']['id'], 'dataset': 'food101', 'split': row['sample']['split'], 'stage': group[1], **{field: row[field] for field in post.formal.FIELDS}, 'source_identity': row['identity'], 'source_path': source['raw_path'], 'source_line': index, 'source_row_sha256': linehash, **result}
                        new_rows += 1
                    label = value['screening_label']
                    counts[label] += 1
                    groups[group][label] += 1
                    stages[group[:2]][label] += 1
                    sink.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                    digest.update(line)
                assert digest.hexdigest() == source['raw_sha256']
        assert len(seen) == 155136 and new_rows == 130896 and used == set(reused)
        conditions = [{**dict(zip(('model', 'stage', 'kind', 'method', 'marker', 'reference_marker', 'reference_guided'), key)), **post.counts_record(c)} for key, c in sorted(groups.items())]
        stage_rows = [{'model': model, 'stage': stage, **post.counts_record(c)} for (model, stage), c in sorted(stages.items())]
        report = {'schema': 'formal_v4_completed_shards_reused_stages_preliminary_screening', 'automatic_preliminary_only': True, 'api_requests': 0, 'gpu_used': False, 'human_reviewed': False, 'final_gt': False, 'semantic_labeling_complete': False, 'research_complete': False, **post.counts_record(counts), 'conditions': conditions, 'stages': stage_rows, 'model_coverage': coverage, 'full_panel_generation_present': False, 'sources': accepted, 'existing_labels_reused': len(used), 'new_matrix_rows_matched': new_rows, 'reuse_evidence': reuse_evidence, 'script_sha256': file_hash(Path(__file__)), 'rule_files_sha256': {p: file_hash(ROOT / p) for p in (post.RULE, 'configs/kdm/food_aliases.json', 'src/kdm/scoring.py', 'src/kdm/models/official_vqa_normalizer.py')}, 'limits': 'Exact lexical preliminary labels only. Existing main/control labels reused with exact row hash binding; unresolved is not incorrect. No human or final GT claim.'}
        (out / 'labels.partial.jsonl').rename(out / 'labels.jsonl')
        atomic_json(out / 'summary.json', report)
        (out / 'SUMMARY.md').write_text(post.markdown_summary(report), encoding='utf-8')
        print(json.dumps({'out': str(out.relative_to(ROOT)), **post.counts_record(counts), 'stages': stage_rows, 'existing_labels_reused': len(used), 'new_matrix_rows_matched': new_rows}))
    except BaseException as exc:
        atomic_json(out / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc), 'partial_outputs_not_accepted': True})
        raise

if __name__ == '__main__':
    main()
