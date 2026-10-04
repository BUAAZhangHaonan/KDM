"""Export the fully accepted nine-model Direct panel as compact Markdown/CSV."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('mmmu', 'scienceqa', 'pope', 'hallusionbench')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT / 'data/general_vqa_direct_20261004/frozen/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    acceptance = json.loads((args.tables/'CURRENT_STATE.json').read_text())
    assert acceptance['accepted_cells'] == acceptance['total_cells'] == 36, 'The complete package requires all 36 resolved cells'
    assert acceptance['generated'] == acceptance['scored'] == acceptance['expected'] == 35559
    assert acceptance['missing_generation'] == acceptance['pending_semantic'] == 0
    assert acceptance['source_scores_sha256'] == sha(args.scores)
    with (args.tables/'metrics_all_36.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line]
    assert len(rows) == 35559
    sample_rows, expected, configs = [], {}, []
    checkpoints = {}
    for model in protocol['models']:
        checkpoints[model] = json.loads((ROOT/f'configs/runtime/{model}.json').read_text())
    for dataset in DATASETS:
        entry = protocol['datasets'][dataset]
        path = ROOT/entry['manifest']
        assert sha(path) == entry['manifest_sha256']
        samples = [json.loads(line) for line in path.read_text().splitlines() if line]
        expected[dataset] = {sample['id'] for sample in samples}
        assert len(samples) == len(expected[dataset]) == entry['expected_rows']
        for sample in samples:
            sample_rows.append({'dataset': dataset, 'sample_id': sample['id'],
                'source_id': sample.get('source_id', ''), 'source_split': sample.get('source_split', ''),
                'question': sample['question'], 'options': json.dumps(sample.get('options', []), ensure_ascii=False),
                'gold': json.dumps(sample['gold'], ensure_ascii=False),
                'image_paths': json.dumps(sample.get('image_paths', []), ensure_ascii=False),
                'gt_answer_details': sample.get('gt_answer_details', ''),
                'manifest_path': entry['manifest'], 'manifest_sha256': entry['manifest_sha256']})
        for model in protocol['models']:
            runtime = checkpoints[model]
            configs.append({'model': model, 'checkpoint': runtime['hf_model_id'],
                'dtype': runtime['dtype'], 'dataset': dataset, 'n': entry['expected_rows'],
                'method': 'direct', 'guided': False, 'temperature': protocol['temperature'],
                'top_p': protocol['top_p'], 'max_new_tokens': entry['max_tokens'],
                'budget_source': entry['budget_source'], 'sampling_seed': protocol['sampling_seed'],
                'prompt_source': entry['prompt_source']})
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['model'], row['dataset'], row['sample_id'])
        assert key not in seen and key[0] in checkpoints and key[2] in expected[key[1]]
        assert row['score'] in (0, 1) and type(row['abstain']) is bool
        seen.add(key)
        groups[key[:2]].append(row)
    for cell in metrics:
        group = groups[(cell['model'], cell['dataset'])]
        assert {row['sample_id'] for row in group} == expected[cell['dataset']]
        n = len(group)
        assert cell['status'] == 'complete' and int(cell['generated']) == int(cell['scored']) == n
        assert int(cell['correct']) == sum(row['score'] for row in group)
        assert int(cell['abstentions']) == sum(row['abstain'] for row in group)
    # All completeness and source checks precede creation of the output directory.
    args.out.mkdir(parents=True, exist_ok=False)
    csv_write(args.out/'metrics.csv', metrics)
    pope_comparison = []
    for model in protocol['models']:
        group = groups[(model, 'pope')]
        assert all(row.get('official_parser_score') in (0, 1) for row in group)
        n = len(group)
        pope_comparison.append({'model': model, 'n': n,
            'semantic_answer_accuracy': sum(row['score'] for row in group) / n,
            'official_parser_accuracy': sum(row['official_parser_score'] for row in group) / n,
            'score_disagreements': sum(row['score'] != row['official_parser_score'] for row in group),
            'abstention_rate': sum(row['abstain'] for row in group) / n})
    csv_write(args.out/'pope_parser_comparison.csv', pope_comparison)
    csv_write(args.out/'sample_manifest.csv', sample_rows)
    csv_write(args.out/'configurations.csv', configs)
    flat, sources = [], Counter()
    fields = ('model', 'dataset', 'sample_id', 'key', 'qa_key', 'question', 'answer',
              'gold', 'score', 'abstain', 'label', 'predicted_answer', 'answer_text',
              'evidence_span', 'terminated', 'n_tokens', 'wall_s', 'identity',
              'dataset_identity', 'claim_identity', 'engine', 'source_raw', 'source_line',
              'official_parser_prediction', 'official_parser_score', 'gt_answer_details')
    for row in rows:
        item = {field: row.get(field, '') for field in fields}
        for field in ('gold', 'predicted_answer'):
            if isinstance(item[field], (list, dict)):
                item[field] = json.dumps(item[field], ensure_ascii=False)
        item['behavior_provenance'] = json.dumps(row.get('decision_source'), ensure_ascii=False)
        item['quality_provenance'] = json.dumps(row.get('quality'), ensure_ascii=False)
        flat.append(item)
        sources[(row['source_raw'], row['model'], row['dataset'], row['identity'])] += 1
    csv_write(args.out/'per_sample_scores.csv', flat)
    csv_write(args.out/'sources.csv', [dict(zip(('raw_path', 'model', 'dataset', 'identity'), key), rows=n)
                                      for key, n in sorted(sources.items())])
    lookup = {(r['model'], r['dataset']): r for r in metrics}
    report = ['# Nine-model native Direct supplementary results', '',
        'All 35,559 responses have resolved correctness and full-response abstention decisions. '
        'Each cell below is accuracy / abstention rate (%). Nine models share the same frozen inputs.', '',
        '| Checkpoint | MMMU (1,000) | ScienceQA image test (1,000) | POPE (1,000) | HallusionBench visual (951) |',
        '|---|---:|---:|---:|---:|']
    for model in protocol['models']:
        values = [f"{100*float(lookup[(model,d)]['accuracy']):.2f} / {100*float(lookup[(model,d)]['abstention_rate']):.2f}" for d in DATASETS]
        report.append('| ' + checkpoints[model]['hf_model_id'] + ' | ' + ' | '.join(values) + ' |')
    report += ['', 'MMMU and ScienceQA use fixed public-test subsets; POPE uses the fixed balanced original-COCO subset. '
        'HallusionBench covers all 951 image-bearing entries. Subset IDs, image paths, original questions and references are in sample_manifest.csv.', '',
        'MMMU and ScienceQA multiple-choice accuracy compares the final extracted original option with the released answer. '
        'MMMU open answers use the pinned official open-answer parser. Response interpretation uses deterministic rules, '
        'exact full-QA reuse and actual GPT-5.6 Luna medium review, with bounded root adjudication. '
        'Unfinished responses without an answer remain invalid; invalid and abstaining responses are counted separately.', '',
        'HallusionBench question accuracy uses its original full reference and score mapping, with actual GPT-5.6 Luna medium adjudication for semantic cases. '
        'It is a Luna-adjudicated visual-question score. Appropriate inability responses to an explicitly indeterminate reference can receive credit; '
        'abstention is also counted independently. POPE official text-parser predictions and scores are preserved alongside semantic answer extraction; '
        'pope_parser_comparison.csv provides both aggregate accuracies and their disagreement counts. The main table uses semantic answer accuracy.', '',
        'Generation is native greedy Direct without abstention instructions. MMMU has a 128-token ceiling based on an author example; '
        'ScienceQA uses the author-example 512-token ceiling. POPE and HallusionBench use the registered 512-token ceiling. EOS ends generation early. '
        'Nonterminated output and invalid-answer counts are reported separately in metrics.csv. A retained answer before a truncated ending is still scored; '
        'unfinished reasoning without an answer does not count as abstention. Original model precision and preprocessing are retained. '
        'Some supported workers use native generation after an eight-input token/input/EOS equivalence check; per-response engine and source identity are retained.', '',
        'Files: metrics.csv (36 complete cells), per_sample_scores.csv (all responses and score provenance), '
        'sample_manifest.csv (3,951 common inputs), configurations.csv (36 model/dataset settings), sources.csv (immutable raw locations). '
        'Full token records and original annotation/correction files remain at the indexed server paths.', '',
        f"Score source: {args.scores.resolve()}", f"Score SHA256: {acceptance['source_scores_sha256']}",
        f"Protocol SHA256: {sha(protocol_path)}", f"Exported UTC: {datetime.now(timezone.utc).isoformat()}", '']
    (args.out/'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    checks = [{'file': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(args.out.iterdir()) if p.is_file()]
    csv_write(args.out/'file_manifest.csv', checks)
    archive = shutil.make_archive(str(args.out), 'zip', root_dir=args.out)
    print(json.dumps({'accepted_cells': 36, 'responses': len(rows), 'out': str(args.out),
                      'archive': archive, 'archive_bytes': Path(archive).stat().st_size}))


if __name__ == '__main__':
    main()
