"""Build compact, source-bound tables from the existing CDA CPU audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.results / 'validation.json').read_text(encoding='utf-8'))
    responses = pd.read_parquet(args.results / 'cda_trace_audit.parquet')
    summary = pd.read_csv(args.results / 'cda_trace_summary.csv', float_precision='round_trip')
    full = summary.loc[summary.panel.eq('full_eval')].copy()
    assert len(responses) == 48480 and responses.steps.sum() == receipt['trace_steps']
    count_errors, weight_errors, weight_relative_errors = [], [], []
    for row in full.itertuples():
        selected = responses
        if row.model != 'ALL':
            selected = selected.loc[selected.model.eq(row.model)]
        if row.marker != 'ALL':
            selected = selected.loc[selected.marker.eq(row.marker)]
        if row.state != 'ALL':
            selected = selected.loc[selected.state.eq(row.state)]
        if row.uniform_reference != 'ALL':
            selected = selected.loc[selected.uniform_reference.eq(row.uniform_reference == 'positive')]
        count_error = max(abs(len(selected) - row.responses), abs(selected.steps.sum() - row.steps))
        count_errors.append(int(count_error))
        assert count_error == 0
        for name in ('wp', 'wc', 'wa'):
            recomputed = np.average(selected[name + '_response_mean'], weights=selected.steps)
            error = abs(recomputed - getattr(row, name + '_step_mean'))
            weight_errors.append(float(error))
            scale = max(1.0, abs(recomputed), abs(getattr(row, name + '_step_mean')))
            weight_relative_errors.append(float(error / scale))
            assert np.isclose(recomputed, getattr(row, name + '_step_mean'), rtol=1e-12, atol=1e-10), (row.aggregation, row.model, row.marker, row.state, row.uniform_reference, name, recomputed, getattr(row, name + '_step_mean'), error)
        assert selected.zero_sum_steps.sum() == row.zero_sum_steps
        assert selected.negative_wa_steps.sum() == row.negative_wa_steps

    condition_counts = []
    for (model, cid, marker), rows in responses.groupby(['model', 'condition_id', 'marker'], sort=True):
        n = len(rows)
        correct = int(rows.state.eq('C').sum())
        wrong = int(rows.state.eq('E').sum())
        abstain = int(rows.state.eq('A').sum())
        tp = int((rows.abstain & rows.uniform_reference).sum())
        fp = int((rows.abstain & ~rows.uniform_reference).sum())
        reference_positive = int(rows.uniform_reference.sum())
        condition_counts.append({
            'dataset': 'food101', 'split': 'eval', 'model': model, 'condition_id': int(cid),
            'method': 'cda_visual', 'marker': marker, 'reference_marker': marker,
            'configuration_status': 'registered_eval_observation', 'n': n, 'C': correct,
            'E': wrong, 'A': abstain, 'TP': tp, 'FP': fp, 'reference_positive': reference_positive,
            'accuracy': fraction(correct, n), 'abstention_precision': fraction(tp, abstain),
            'abstention_recall': fraction(tp, reference_positive), 'J': fraction(correct + tp, n),
            'precision_null_reason': 'no_abstentions' if abstain == 0 else '',
            'trace_steps': int(rows.steps.sum()), 'zero_sum_steps': int(rows.zero_sum_steps.sum()),
            'negative_wa_steps': int(rows.negative_wa_steps.sum()),
        })
    pd.DataFrame(condition_counts).to_csv(args.results / 'cda_condition_behavior_metrics.csv', index=False)

    model_state_reference = full.loc[full.aggregation.eq('model_state_reference')].copy()
    compact_columns = [
        'dataset', 'split', 'model', 'marker', 'state', 'uniform_reference', 'responses', 'steps',
        'wp_step_mean', 'wc_step_mean', 'wa_step_mean', 'wp_response_mean', 'wc_response_mean',
        'wa_response_mean', 'wp_first_position_mean', 'wc_first_position_mean', 'wa_first_position_mean',
        'zero_sum_steps', 'negative_wa_steps', 'zero_sum_steps_fraction', 'negative_wa_steps_fraction',
        'abstention_largest_steps', 'abstention_largest_steps_fraction', 'responses_any_zero_sum',
        'responses_all_zero_sum', 'responses_any_negative_wa', 'responses_abstention_largest_all_steps',
    ]
    model_state_reference[compact_columns].to_csv(args.results / 'cda_model_state_reference.csv', index=False)
    model_state_rows = []
    for (model, state), group in model_state_reference.groupby(['model', 'state'], sort=True):
        steps = int(group.steps.sum())
        count = int(group.responses.sum())
        row = {'model': model, 'state': state, 'responses': count, 'steps': steps}
        for name in ('wp', 'wc', 'wa'):
            row[name + '_step_mean'] = group[name + '_sum'].sum() / steps
            row[name + '_response_mean'] = group[name + '_response_mean_sum'].sum() / count
            row[name + '_first_position_mean'] = group[name + '_first_sum'].sum() / count
        for name in ('zero_sum_steps', 'negative_wa_steps', 'abstention_largest_steps'):
            row[name] = int(group[name].sum())
            row[name + '_fraction'] = row[name] / steps
        model_state_rows.append(row)
    pd.DataFrame(model_state_rows).to_csv(args.results / 'cda_model_state_weights.csv', index=False)
    model_all = full.loc[full.aggregation.eq('model_all')].copy()
    model_all[compact_columns].to_csv(args.results / 'cda_model_all_weights.csv', index=False)

    explanation = [
        '# CDA 既有轨迹 CPU 审计',
        '',
        f"覆盖五模型的 20 个注册 Food eval 条件，每条件 2,424 个输入，共 {receipt['response_rows']:,} 条回复、{receipt['trace_steps']:,} 个解码步骤。四种措辞属于不同生成条件；模型层汇总包含每个模型 9,696 条条件—输入记录与 2,424 个唯一输入。",
        '',
        f"公式 6、7 从已有四项熵逐步复算，{receipt['recomputed_steps']:,} 步均有完整字段；参数与权重向量最大误差为 0，零和/负权重标志差异为 0。权重和距离 1 的最大误差为 {receipt['weights_sum_max_error']:.3g}。注册 representative101 面板的 2,020 条回复、18,545 步与旧来源行 SHA、tokens、trace 和冻结状态全部一致。",
        '',
        f"零和连续扩展发生于 {receipt['zero_sum_steps']:,}/{receipt['trace_steps']:,} 步（{receipt['zero_sum_steps']/receipt['trace_steps']:.2%}），其 wp=wc=0、wa=1；负 wa 发生于 {receipt['negative_wa_steps']:,}/{receipt['trace_steps']:,} 步（{receipt['negative_wa_steps']/receipt['trace_steps']:.2%}），原值保留。权重允许负数，权重最高的分支属于分数组合的描述量，完整回复的弃权沿用冻结语义判定。",
        '',
        '| 模型 | 正确 C | 错误 E | 弃权 A | 平均 wa（全部步骤） | 零和步骤比例 | 负 wa 步骤比例 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in model_all.itertuples():
        own = responses.loc[responses.model.eq(row.model)]
        counts = own.state.value_counts()
        explanation.append(f"| {row.model} | {counts.get('C', 0):,} | {counts.get('E', 0):,} | {counts.get('A', 0):,} | {row.wa_step_mean:.6f} | {row.zero_sum_steps_fraction:.2%} | {row.negative_wa_steps_fraction:.2%} |")
    explanation.extend([
        '',
        '正确/错误/弃权与参考阳性/阴性的共同分组位于 cda_model_state_reference.csv。该表同时提供按步骤、按回复及首位的三个权重均值，分母和统计单位分别保存；条件级 J、准确率、弃权精确率/召回率及原始计数位于 cda_condition_behavior_metrics.csv。各措辞保持注册 eval 观测身份，开发集选定配置由主执行流程另行登记。',
        '',
        '## 配置与来源',
        '',
        '视觉迁移使用：prior 为原问题去图像条件，context 为原问题与正常图像，abstention 为原问题、正常图像与注册弃权措辞；空文本为 [N/A]，空视觉为同尺寸 RGB=(127,127,127) 图像。五会话均采用同一已生成词元前缀。身份、原始 DecodeConfig、原始 source_identity 与无动量实现标识保存于 cda_config_identity.csv。熵差方向保留 max(H_branch−H_null_branch,0)/H_null_branch。',
        '',
        'sources.csv 给出五个正式 gzip 的真实路径与已请求行范围；cda_trace_audit.parquet/CSV 给出每回复 source_file_id、source_line、score_source_line、seed、状态与参考值。cda_trace_steps.csv.gz 保存全部步骤，cda_trace_representative_steps.csv.gz 独立保存代表面板步骤。旧评分和参考源于 outputs/paper_20260929/scores.parquet 与 references.parquet；每个 model/sample_id 的参考连接均已核验。',
        '',
        '## 已保存字段的限制',
        '',
        '原 trace 保存 h_prior、h_context、h_null_prior、h_null_context、rp、rc、wp、wc、wa、三项 weights、零和/负权重标志、所选 token 与所选分数。弃权分支熵 h_abstention 和完整 prior/context/abstention 分支 logits 没有保存，公式 4 的全词表向量与对应 argmax 无法从这些数据复算；权重和检查单独记账。cda_execution_differences.csv 为空表，表示实际已查公式及标志没有差异。',
        '',
        'CPU 审计新增生成数量为 0，API 调用数量为 0，GPU 使用为 0。当前 src/kdm/cda.py 的 SHA 与旧代表性审计登记相同。统计描述实际权重和最终行为的联系，不据此指定单一分支造成回复变化。',
        '',
    ])
    (args.results / 'CDA_AUDIT_REPORT.md').write_text('\n'.join(explanation), encoding='utf-8')
    verification = {
        'summary_rows_checked': len(full), 'max_response_or_step_count_error': max(count_errors),
        'max_step_weighted_mean_reconstruction_error': max(weight_errors),
        'max_scale_normalized_mean_reconstruction_error': max(weight_relative_errors),
        'summary_reconstruction_tolerance': {'absolute': 1e-10, 'relative': 1e-12},
        'condition_behavior_rows': len(condition_counts), 'model_state_reference_rows': len(model_state_reference),
        'new_generations': 0, 'GPU_initialized': False,
    }
    (args.results / 'summary_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))
    print(pd.DataFrame(model_state_rows)[['model', 'state', 'responses', 'steps', 'wp_step_mean', 'wc_step_mean', 'wa_step_mean', 'zero_sum_steps_fraction', 'negative_wa_steps_fraction']].to_csv(index=False))


if __name__ == '__main__':
    main()
