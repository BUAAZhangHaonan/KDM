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
    review_root = ROOT / 'outputs/general_vqa_direct'
    holds = [path for folder in ('annotations', 'quality_annotations')
             for path in (review_root/folder).glob('*/REVIEW_HOLD.json')]
    assert not holds, f'Unresolved semantic review holds prevent final export: {len(holds)}'
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
    report_zh = ['# 九模型 Direct 四基准补充结果', '',
        '35,559 条回答已完成生成、正确性与全文语义弃权标注，36 个模型—数据集组合均通过完整键覆盖检查。'
        '九个模型使用相同的固定样本。下表每格为正确率 / 弃权率，单位为百分比。', '',
        '| 完整检查点 | MMMU（1,000） | ScienceQA 图像测试集（1,000） | POPE（1,000） | HallusionBench 图像题（951） |',
        '|---|---:|---:|---:|---:|']
    for model in protocol['models']:
        values = [f"{100*float(lookup[(model,d)]['accuracy']):.2f} / {100*float(lookup[(model,d)]['abstention_rate']):.2f}" for d in DATASETS]
        report_zh.append('| ' + checkpoints[model]['hf_model_id'] + ' | ' + ' | '.join(values) + ' |')
    report_zh += ['',
        '分母为各数据集全部固定输入。MMMU 与 ScienceQA 按原选项或官方开放回答规则评分；'
        'POPE 主表采用完整回复的语义答案，另在 pope_parser_comparison.csv 保留官方文本解析器的对照结果。'
        'HallusionBench 采用官方完整参考答案和题级得分映射，由实际 GPT-5.6 Luna medium 裁定语义边界；'
        '该列是 Luna 裁定的题级正确率。官方参考明确允许无法确定时，恰当弃权可以得分，得分与弃权分别记账。', '',
        '弃权率依据模型完整回复中实际拒绝或无法作答的行为。仅有不确定语气仍保留明确答案的回答继续评分。'
        '输出被截断但已有保留答案时仍提取答案；尚未得到答案的未完成推理记为无有效答案，不计作弃权。', '',
        '## 生成预算与输出结束状态', '',
        '本轮使用原生无弃权引导 Direct、greedy 解码，保留登记的检查点、精度与图像处理。'
        'MMMU 上限 128 token，来源为作者示例；ScienceQA 上限 512 token，来源为作者示例；'
        'POPE 与 HallusionBench 上限为本轮登记的 512 token。具体来源见 configurations.csv。'
        'EOS 提前结束。下表单列 MMMU 未正常结束和无有效答案数量，便于区分生成预算影响与语义弃权。', '',
        '| 完整检查点 | MMMU 输入数 | 未正常结束 | 无有效答案 | 语义弃权 |',
        '|---|---:|---:|---:|---:|']
    for model in protocol['models']:
        cell = lookup[(model, 'mmmu')]
        report_zh.append(f"| {checkpoints[model]['hf_model_id']} | {cell['expected']} | {cell['truncated']} | {cell['invalid']} | {cell['abstentions']} |")
    report_zh += ['',
        'MMMU 存在较高的未结束输出比例时，正确率反映本轮生成预算下的表现，不能直接解释为该模型不受预算限制的作答能力。'
        '所有已生成回复及标注来源完整保留。其他数据集的对应计数见 metrics.csv。', '',
        '## 数据文件', '',
        '- metrics.csv：36 个完整组合的分子、分母、正确率、弃权率、无有效答案和未正常结束计数。',
        '- per_sample_scores.csv：35,559 条完整问题、回答、得分、行为判断及来源。',
        '- sample_manifest.csv：3,951 个固定输入、图像路径、原始问题与参考答案。',
        '- configurations.csv：36 个模型—数据集配置及生成预算来源。',
        '- sources.csv：不可变原始输出路径与各来源行数；完整 token 记录留在服务器对应路径。',
        '- pope_parser_comparison.csv：POPE 官方解析与完整回复语义解析的差异。', '',
        f"最终评分来源：{args.scores.resolve()}",
        f"评分 SHA256：{acceptance['source_scores_sha256']}",
        f"协议 SHA256：{sha(protocol_path)}", '']
    (args.out/'RESULTS_ZH.md').write_text('\n'.join(report_zh), encoding='utf-8')
    checks = [{'file': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(args.out.iterdir()) if p.is_file()]
    csv_write(args.out/'file_manifest.csv', checks)
    archive = shutil.make_archive(str(args.out), 'zip', root_dir=args.out)
    print(json.dumps({'accepted_cells': 36, 'responses': len(rows), 'out': str(args.out),
                      'archive': archive, 'archive_bytes': Path(archive).stat().st_size}))


if __name__ == '__main__':
    main()
