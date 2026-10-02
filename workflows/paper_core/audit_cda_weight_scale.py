"""Describe stored CDA weight magnitudes and their positions without inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(group: pd.DataFrame) -> dict:
    result = {
        'steps': len(group), 'zero_sum_steps': int(group.zero_sum_extension.sum()),
        'negative_wa_steps': int(group.negative_wa.sum()),
        'h_null_prior_min': float(group.h_null_prior.min()),
        'h_null_context_min': float(group.h_null_context.min()),
        'max_abs_weight': float(group.max_abs_weight.max()),
    }
    for name in ('wp', 'wc', 'wa'):
        result[name + '_mean'] = float(group[name].mean())
        for label, quantile in (('minimum', 0), ('median', .5), ('p90', .9), ('p99', .99), ('maximum', 1)):
            result[name + '_' + label] = float(group[name].quantile(quantile))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/g203-4028/projects/knowledge-deficit-mitigation'))
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    needed = [
        'model', 'condition_id', 'marker', 'state', 'uniform_reference', 'sample_id',
        'source_file_id', 'source_line', 'step_index', 'token', 'h_prior', 'h_context',
        'h_null_prior', 'h_null_context', 'rp', 'rc', 'wp', 'wc', 'wa',
        'zero_sum_extension', 'negative_wa', 'log_probability',
    ]
    events = pd.read_csv(args.results / 'cda_trace_steps.csv.gz', usecols=needed, float_precision='round_trip')
    responses = pd.read_parquet(args.results / 'cda_trace_audit.parquet', columns=['source_file_id', 'source_line', 'steps', 'terminated'])
    responses = responses.rename(columns={'steps': 'response_steps', 'terminated': 'response_terminated'})
    events = events.merge(responses, on=['source_file_id', 'source_line'], how='left', validate='many_to_one')
    assert events.response_steps.notna().all()
    assert events.groupby(['source_file_id', 'source_line']).size().to_numpy().sum() == 426903
    events['is_recorded_eos'] = events.response_terminated & events.step_index.eq(events.response_steps - 1)
    events['position'] = np.where(events.is_recorded_eos, 'recorded_eos', 'before_eos_or_unterminated')
    events['max_abs_weight'] = events[['wp', 'wc', 'wa']].abs().max(axis=1)
    rows = []
    for (model, position), group in events.groupby(['model', 'position'], sort=True):
        rows.append({'aggregation': 'model_position', 'model': model, 'marker': 'ALL', 'state': 'ALL', 'uniform_reference': 'ALL', 'position': position, **describe(group)})
    for (model, state, reference, position), group in events.groupby(['model', 'state', 'uniform_reference', 'position'], sort=True):
        rows.append({'aggregation': 'model_state_reference_position', 'model': model, 'marker': 'ALL', 'state': state, 'uniform_reference': bool(reference), 'position': position, **describe(group)})
    for model, group in events.loc[events.step_index.eq(0)].groupby('model', sort=True):
        rows.append({'aggregation': 'model_first_position', 'model': model, 'marker': 'ALL', 'state': 'ALL', 'uniform_reference': 'ALL', 'position': 'first_position_including_possible_eos', **describe(group)})
    for (model, marker, position), group in events.groupby(['model', 'marker', 'position'], sort=True):
        rows.append({'aggregation': 'condition_position', 'model': model, 'marker': marker, 'state': 'ALL', 'uniform_reference': 'ALL', 'position': position, **describe(group)})
    pd.DataFrame(rows).to_csv(args.results / 'cda_weight_position_summary.csv', index=False)
    extremes = events.nlargest(12, 'max_abs_weight')
    extremes.to_csv(args.results / 'cda_weight_extreme_steps12.csv', index=False)

    examples = []
    bound_path = args.root / 'outputs/paper_core_20260930/run_20260930_core_p0/cda_trace/source_bound_trace_rows.jsonl.gz'
    used_states = set()
    with gzip.open(bound_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            content_steps = record['trace'][:-1] if record['terminated'] else record['trace']
            zero_content_steps = [index for index, step in enumerate(content_steps) if step['zero_sum_extension']]
            desired = record['model'] == 'minicpm26' and record['state'] in ('C', 'E') and bool(zero_content_steps)
            if desired and record['state'] not in used_states:
                examples.append({'selection_reason': 'MiniCPM has a stored non-EOS wp=wc=0, wa=1 step with frozen final state ' + record['state'], 'zero_sum_non_eos_step_indices': zero_content_steps, **record})
                used_states.add(record['state'])
            if used_states == {'C', 'E'}:
                break
    source_examples = {}
    for record in examples:
        source_examples.setdefault(record['source_path'], {})[record['source_line']] = record
    for path, wanted in source_examples.items():
        remaining = set(wanted)
        with gzip.open(path, 'rb') as stream:
            for number, line in enumerate(stream, 1):
                if number > max(wanted):
                    break
                if number not in wanted:
                    continue
                record = wanted[number]
                assert hashlib.sha256(line).hexdigest() == record['raw_line_sha256']
                raw = json.loads(line)
                assert raw['text'] == record['text'] and raw['tokens'] == record['tokens'] and raw['trace'] == record['trace']
                record.update({'question': raw['sample']['question'], 'main_prompt': raw['prompt'], 'reference_prompt': raw['reference_prompt'], 'seed': raw['seed'], 'config': raw['config']})
                remaining.remove(number)
        assert not remaining
    with (args.results / 'cda_branch_weight_behavior_examples.jsonl').open('w', encoding='utf-8') as stream:
        for record in examples:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    receipt = {
        'steps': len(events), 'recorded_eos_steps': int(events.is_recorded_eos.sum()),
        'before_eos_or_unterminated_steps': int((~events.is_recorded_eos).sum()),
        'extreme_examples': len(extremes), 'extreme_examples_recorded_eos': int(extremes.is_recorded_eos.sum()),
        'largest_absolute_weight': float(events.max_abs_weight.max()),
        'smallest_recorded_null_entropy': float(events[['h_null_prior', 'h_null_context']].min().min()),
        'non_eos_zero_sum_behavior_examples': len(examples),
        'position_definition': 'last saved token of a source response with terminated=true is recorded EOS',
        'new_generations': 0, 'GPU_initialized': False,
    }
    (args.results / 'weight_scale_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    report = [
        '# CDA 权重的幅度与生成位置',
        '',
        f"全部 {len(events):,} 个已保存步骤中，{receipt['recorded_eos_steps']:,} 个是终止状态记录的最后词元，{receipt['before_eos_or_unterminated_steps']:,} 个位于终止词元之前或属于未终止回复。最大绝对权重为 {receipt['largest_absolute_weight']:.6g}；最小已保存空输入熵为 {receipt['smallest_recorded_null_entropy']:.6g}。12 个最大绝对权重位置全部位于终止词元之前。公式 6 的正熵分母保留原值，公式 7 的分支权重与负 wa 沿用原记录。",
        '',
        '| 模型 | 首位 wa 中位数 | 首位最大绝对权重 | 终止前 wa 中位数 | 终止前 wc 均值 | 终止前最大绝对权重 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    table = pd.DataFrame(rows)
    for model in sorted(events.model.unique()):
        first = table.loc[table.aggregation.eq('model_first_position') & table.model.eq(model)].iloc[0]
        content = table.loc[table.aggregation.eq('model_position') & table.model.eq(model) & table.position.eq('before_eos_or_unterminated')].iloc[0]
        report.append(f"| {model} | {first.wa_median:.6f} | {first.max_abs_weight:.6g} | {content.wa_median:.6f} | {content.wc_mean:.6g} | {content.max_abs_weight:.6g} |")
    report.extend([
        '',
        '因此全路径的权重均值与首位权重分别报告。Gemma 的后续步骤存在极大权重，全步骤均值受这些值影响；中位数与分位数完整保存在 cda_weight_position_summary.csv。该文件也保存 C/E/A 与参考阳性/阴性的分组。',
        '',
        '弃权分支是带有弃权指令的模型分布，它可以继续生成菜名。以下两个来源已核验的 MiniCPM 完整回复在非 EOS 步骤具有 wp=wc=0、wa=1，最终状态分别为正确作答和错误作答：',
        '',
    ])
    for record in examples:
        report.append(f"- {record['sample_id']}，措辞 {record['marker']}，冻结状态 {record['state']}；完整回复：{record['text']!r}；零和正文位置 {record['zero_sum_non_eos_step_indices']}（从 0 起计）；来源 {record['source_path']} 第 {record['source_line']} 行。")
    report.extend([
        '',
        'cda_branch_weight_behavior_examples.jsonl 保留这两条回复的完整问题、提示、词元、原始 trace、seed 与来源 SHA。cda_weight_extreme_steps12.csv 保留极值步骤的原始四项熵、权重、生成位置和来源定位。全部统计读取现有数据，新增推理与 GPU 使用均为 0。',
        '',
    ])
    (args.results / 'CDA_WEIGHT_POSITION_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).loc[lambda frame: frame.aggregation.isin(['model_position', 'model_first_position']), ['model', 'position', 'steps', 'wp_median', 'wc_median', 'wa_median', 'wc_mean', 'wa_mean', 'max_abs_weight', 'h_null_context_min']].to_csv(index=False))
    print(json.dumps([{'model': row['model'], 'state': row['state'], 'sample_id': row['sample_id'], 'marker': row['marker'], 'text': row['text'], 'source_path': row['source_path'], 'source_line': row['source_line'], 'tokens': row['tokens']} for row in examples], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
