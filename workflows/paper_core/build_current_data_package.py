"""Assemble one current CPU-only research handoff from accepted saved evidence.

Original archives and remote results are never modified.  Large verbatim records
remain source-addressable; their compact scientific projections are included.
"""
import argparse
import csv
import io
import json
import shutil
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, required=True,
                    help='Revision data_package directory containing accepted addenda and source projections')
HERE = parser.parse_args().data_dir.resolve()
BASE = HERE.parents[1]
PAPER = HERE.parent / 'KDM_Paper_v3'
CURRENT = HERE / 'current'
SOURCE = BASE / 'deliveries/KDM_Nine_Current_Review.zip'
REMOTE_BASE = '/home/g203-4028/projects/knowledge-deficit-mitigation/outputs/paper_core_20261002_dev_viz'
REMOTE_PACKAGE = REMOTE_BASE + '/final_review_package_complete_accepted_sources_v2_20261003/package'
MODELS = ['qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b',
          'internvl35_8b', 'onevision', 'phi35', 'qwen3vl']


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def disposition(name):
    path = PurePosixPath(name)
    if '..' in path.parts or path.is_absolute():
        raise ValueError(name)
    remote_only = {
        'all_food_condition_metrics483.csv',
        'dev_and_J/Food_eval_J/main69_actual_operating_points.csv',
        'vizwiz_final/new25_complete_records.parquet',
        'main/frozen_QA_raw_source_bindings.parquet',
        'support/mechanism/natural_reference/dictionaries/qa_dictionary.parquet',
        'support/mechanism/natural_reference/dictionaries/label_dictionary.parquet',
    }
    if name in remote_only:
        return 'remote_only', 'full source object retained; compact scores and source indices included'
    if name.startswith(('main/', 'mechanism/', 'support/references/', 'support/registered_lists/',
                        'support/registered_protocol/', 'support/dev_selection/',
                        'support/vizwiz/core5_512/', 'support/mechanism/', 'support/cases/',
                        'support/cda/', 'dev_and_J/dev_selection/', 'dev_and_J/Food_eval_J/',
                        'dev_and_J/frozen_prior_core5/', 'bounded_details_remaining4/')):
        if any(token in name for token in ('previous124_', 'prior_snapshot', '__pycache__')):
            return 'omitted_superseded', 'a complete current scientific projection is included'
        if path.suffix.lower() in ('.png', '.pdf', '.svg') and 'cases/' not in name:
            return 'omitted_presentation', 'paper figures are rebuilt in the separate paper package'
        return 'included', 'accepted scientific result or its configuration/source evidence'
    if name.startswith('vizwiz_final/') and '/source_metadata/' not in name:
        return 'included', 'closed fixed512 results and source receipts'
    if name.startswith('closeout/baseline/'):
        return 'included', 'same101 official-operator comparison; separate denominator from full eval'
    if name.startswith('closeout/mechanism/'):
        if path.name.endswith('_STATE.json'):
            return 'omitted_snapshot', 'current finite experiment tables and complete receipts included'
        return 'included', 'finite CDA or replay measurement'
    if name in ('MODEL_CHECKPOINTS.csv', 'all_food_condition_metrics483.csv',
                'nine_model_main_joint_J.csv', 'necessary_and_unnecessary_abstention_changes.csv'):
        return 'included', 'frozen full input comparison'
    return 'omitted_noncurrent', 'superseded status, duplicate presentation, or execution artifact; original archive retained'


def main():
    CURRENT.mkdir(parents=True, exist_ok=True)
    manifest = []
    status = []
    created = datetime.now(timezone.utc).isoformat()
    gemma_dir = HERE / 'addendum/gemma3_sid'
    gemma_receipt = json.loads((gemma_dir / 'receipt.json').read_text(encoding='utf-8'))
    gemma_metrics = csv_rows((gemma_dir / 'metrics_all.csv').read_bytes())
    gemma_samples = csv_rows((gemma_dir / 'per_sample.csv').read_bytes())
    assert gemma_receipt['passed'] and gemma_receipt['primary_complete'] and gemma_receipt['pending_unique_QA'] == 0
    assert len(gemma_metrics) == 1 and len(gemma_samples) == len({row['sample_id'] for row in gemma_samples}) == 2424
    assert set(Counter(row['target_class'] for row in gemma_samples).values()) == {24}
    assert len({row['target_class'] for row in gemma_samples}) == 101
    gemma = dict(gemma_metrics[0])
    gemma.update(W=gemma['W_decided'], checkpoint='google/gemma-3-4b-it',
                 main_marker=gemma['marker'], score_source='addendum/gemma3_sid/score_rows.parquet',
                 source_ids='["gemma_native_mask_sid_20261004"]',
                 selection='registered_single_native_configuration',
                 implementation_revision='gemma_native_mask_sid_20261004',
                 canonical_pending=0, abstain_pending=0, reference_pending=0)
    assert sum(int(row['correct_canonical']) for row in gemma_samples) == int(gemma['C']) == 1259
    assert sum(row['uniform_reference'] == 'True' for row in gemma_samples) == int(gemma['reference_positive']) == 422
    assert all(row['abstain'] == 'False' for row in gemma_samples)
    with ZipFile(SOURCE) as archive:
        prior = {row['member']: row for row in json.loads(archive.read('CURRENT_MANIFEST.json'))}
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            action, reason = disposition(name)
            original = prior.get(name, {})
            if action == 'included':
                target = CURRENT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as reader, target.open('wb') as writer:
                    shutil.copyfileobj(reader, writer)
            elif action == 'remote_only':
                target = CURRENT / name
                if target.exists():
                    assert target.resolve().is_relative_to(CURRENT.resolve())
                    assert target.stat().st_size == entry.file_size
                    assert zlib.crc32(target.read_bytes()) == entry.CRC
                    target.unlink()  # Only our verified extraction; source archive is retained.
            manifest.append(dict(package_path=name if action == 'included' else '',
                                 disposition=action, source_archive=str(SOURCE), source_member=name,
                                 source_path=original.get('source', REMOTE_PACKAGE + '/' + name),
                                 source_bytes=entry.file_size, archive_crc32=f'{entry.CRC:08x}',
                                 recorded_sha256=original.get('sha256', ''), reason=reason))
        main_metrics = csv_rows(archive.read('main/metrics_all_complete.csv'))
        mech_metrics = csv_rows(archive.read('mechanism/nine_model_all360_condition_metrics.csv'))
        viz_metrics = csv_rows(archive.read('vizwiz_final/all56_metrics.csv'))
        dev_metrics = csv_rows(archive.read('dev_and_J/dev_selection/dev_metrics52.csv'))
        assert len(main_metrics) == 123 and len(mech_metrics) == 360 and len(viz_metrics) == 56
        assert not any(row['model']=='gemma3_4b' and row['method']=='sid' for row in main_metrics)
        all_food = csv_rows(archive.read('all_food_condition_metrics483.csv'))
        operating = csv_rows(archive.read('dev_and_J/Food_eval_J/main69_actual_operating_points.csv'))
        assert len(all_food) == 483 and len(operating) == 69
        identity_fields = ('model','dataset','split','method','kind','marker','reference_marker','guided','reference_guided','replicate')
        def identity(row):
            values = dict(row)
            values['marker'] = row.get('main_marker') or row.get('marker')
            return tuple(str(values.get(key) or ('food101' if key=='dataset' else '')).lower() for key in identity_fields)
        all_food.append(gemma)
        operating.append(gemma)
        main_metrics.append(gemma)
        assert len({identity(row) for row in all_food}) == 484
        assert len({identity(row) for row in operating}) == 70
        assert sum(int(float(row.get('n') or row.get('N'))) for row in all_food) == 1173216
        assert len(main_metrics) == 124
        for path, rows in (('main/metrics_all_complete.csv',main_metrics),
                           ('Food_eval_all484_condition_metrics.csv',all_food),
                           ('Food_main70_actual_operating_points.csv',operating)):
            fields = list(dict.fromkeys(key for row in rows for key in row))
            write_csv(CURRENT / path,rows,fields)
            prior_entry = next((row for row in manifest if row['package_path']==path),None)
            if prior_entry:
                prior_entry['reason'] = 'frozen 123 conditions plus accepted Gemma original SID 2424; generation identity remains explicit'
            else:
                manifest.append(dict(package_path=path,disposition='included',source_archive=str(SOURCE),source_member='',
                                     source_path='frozen archive tables + addendum/gemma3_sid/metrics_all.csv',source_bytes='',
                                     archive_crc32='',recorded_sha256='',reason='exact count-preserving union; no duplicate condition identities'))
        for path, sid_column in (('nine_model_main_joint_J.csv','SID'),
                                 ('dev_and_J/Food_eval_J/J_main_comparison9.csv','sid')):
            rows = csv_rows(archive.read(path))
            target = [row for row in rows if row['model']=='gemma3_4b']
            assert len(target)==1 and target[0][sid_column] in ('','nan')
            target[0][sid_column] = gemma['J']
            write_csv(CURRENT / path,rows)
            next(row for row in manifest if row['package_path']==path)['reason'] = 'unchanged frozen values plus accepted Gemma original SID J'
        for model in MODELS:
            for stage, metrics, n_field, expected_n, source in (
                ('Food_eval_registered_main', main_metrics, 'n', 2424, 'main/metrics_all_complete.csv'),
                ('Food_eval_VCD_M3ID_matrix_and_ref_off', mech_metrics, 'N', 2424, 'mechanism/nine_model_all360_condition_metrics.csv'),
                ('VizWiz_official_eval512_all', viz_metrics, 'n', 512, 'vizwiz_final/all56_metrics.csv'),
                ('Food_dev404_new_registered_IP_conditions', dev_metrics, 'n', 404, 'dev_and_J/dev_selection/dev_metrics52.csv'),
            ):
                subset = [row for row in metrics if row['model'] == model]
                total = sum(int(float(row[n_field])) for row in subset)
                assert all(int(float(row[n_field])) == expected_n for row in subset)
                status.append(dict(stage=stage, model=model, condition_count=len(subset),
                                   completed_rows=total, expected_rows=len(subset)*expected_n,
                                   semantics_pending_rows=0, reference_pending_rows=0,
                                   status='closed', scope='actual registered conditions in source; prior core5 IP-VCD dev selection is additionally included',
                                   source=source))
        for stage, rows, conditions, scope, source in (
            ('Food_eval_total_main_and_mechanism',1173216,484,'124 main plus 360 mechanism conditions; accepted Gemma original SID included', 'Food_eval_all484_condition_metrics.csv'),
            ('VizWiz_official_eval512_main',23040,45,'nine models five primary methods; 166 unanswerable and 346 answerable per condition','vizwiz_final/main45_metrics.csv'),
            ('VizWiz_official_eval512_mechanism',5632,11,'guided comparisons used only for mechanism','vizwiz_final/all56_metrics.csv'),
            ('Food_dev404_IP_selection',29088,72,'21008 new rows plus 8080 prior core5 IP-VCD rows; 18 selected model-methods','dev_and_J/dev_selection/receipt.json;support/dev_selection/registered_dev_decisions.parquet'),
            ('Food_dev404_core5_four_method_selection',32320,80,'five models, four methods and four expressions on 404 identical dev inputs; IP-VCD overlaps previous row','support/dev_selection/registered_dev_decisions.parquet'),
            ('author_DoLa_SID_same101',1515,15,'nine DoLa and six SID conditions; 101 inputs per condition','supplementary/operator_summary101.csv'),
            ('CDA_finite_replay',803,108,'nine models, 12 paths per model, 803 token positions','supplementary/cda_model_behavior_weights.csv'),
            ('replay_hardware_and_schedule',36,6,'six paths and nine selected positions measured on two devices and two schedules','supplementary/replay_summary.csv'),
            ('replay_fixed_noise',4,1,'one MiniCPM path and two token positions on two devices','supplementary/replay_fixed_noise_positions.csv'),
        ):
            status.append(dict(stage=stage, model='all', condition_count=conditions,
                               completed_rows=rows, expected_rows=rows, semantics_pending_rows=0,
                               reference_pending_rows=0,status='closed',scope=scope,source=source))
        core_coverage = csv_rows(archive.read('support/mechanism/core5/coverage.csv'))
        extra_coverage = json.loads(archive.read('support/mechanism/extra4_representative101/verification.json'))
        for model in MODELS:
            status.append(dict(stage='Food_representative_first_token_four_views',model=model,condition_count=1,
                               completed_rows=101,expected_rows=101,semantics_pending_rows='',reference_pending_rows=0,
                               status='closed',scope='101 images, one per class; token proxies are distinct from response semantic metrics',
                               source='support/mechanism/core5/coverage.csv;support/mechanism/extra4_representative101/verification.json'))
            status.append(dict(stage='Food_unguided_perturbed_reference',model=model,condition_count=1,
                               completed_rows=101,expected_rows=101,semantics_pending_rows=0,reference_pending_rows=0,
                               status='closed',scope='101 natural-reference generations with saved semantic labels',
                               source='support/mechanism/natural_reference/'))
            if model in extra_coverage['representative_counts']:
                assert extra_coverage['representative_counts'][model] == extra_coverage['natural_counts'][model] == 101
        for row in core_coverage:
            status.append(dict(stage='core5_diagnostic_and_full_paths',model=row['model'],condition_count='',
                               completed_rows=row['diagnostic_rows'],expected_rows=row['diagnostic_rows'],
                               semantics_pending_rows='',reference_pending_rows=0,status='closed',
                               scope=f"{row['diagnostic_rows']} diagnostic inputs; {row['diagnostic_no_divergence']} without divergence; {row['cases']} full cases",
                               source='support/mechanism/core5/coverage.csv'))
        for row in json.loads(archive.read('bounded_details_remaining4/completed_only_manifest.json'))['sources']:
            status.append(dict(stage='extension4_finite_'+row['phase'],model=row['model'],condition_count='',
                               completed_rows=row['actual_events'],expected_rows=row['actual_events'],
                               semantics_pending_rows='',reference_pending_rows=0,status='closed',
                               scope=f"{row['completed_inputs']} inputs; {row['actual_complete_paths']} full paths; {row['actual_events']} measured positions",
                               source='bounded_details_remaining4/completed_only_manifest.json'))
    def copy_file(source, destination, role, original_source=None):
        target = CURRENT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(dict(package_path=destination, disposition='included', source_archive='', source_member='',
                             source_path=original_source or str(source), source_bytes=source.stat().st_size,
                             archive_crc32='', recorded_sha256='', reason=role))
    supplementary = PAPER / 'statistics/supplementary'
    wanted = ['dev_selection20.csv','dev_selected_eval20.csv','operator_summary101.csv',
              'operator_paired_101.csv','same101_J.csv','same101_IP_paired_changes.csv',
              'cda_model_behavior_weights.csv','cda_model_behavior_reference_weights.csv',
              'replay_summary.csv','replay_hardware_positions.csv','replay_schedule_positions.csv',
              'replay_fixed_noise_positions.csv','source_rows.csv','dev_selected20_sources.json',
              'dev_selected_IP_vs_native_J.csv','assembly_checks.json','README.md']
    for name in wanted:
        copy_file(supplementary / name, 'supplementary/' + name, 'newly assembled, unchanged scientific values with exact source rows')
    for name in ('appendix_supplement_v3.md', 'appendix_supplement_v3.tex'):
        copy_file(PAPER / name, 'supplementary/' + name, 'scientific appendix text and tables; paper package integrates the TeX')
    for name in ('VIZ_INDEPENDENT_STATUS.csv','VIZ_INDEPENDENT_SUMMARY.json'):
        copy_file(HERE / name,name,'current measured coverage; no draft labels counted')
    for name in ('core_scored_scope.json','viz25_compact_receipt.json','viz25_compact_scores.csv.gz','scientific_source_counts.json'):
        destination = ('vizwiz_final/' if name.startswith('viz25_') else 'current_evidence/') + name
        copy_file(HERE / 'source_inputs' / name, destination, 'CPU projection with no new inference or rescoring',
                  REMOTE_BASE + '/closeout_20261004/package_cpu_projection/' + name)
    for name in ('A100_generation_verification_20261004.json', 'K100_generation_verification_20261004.json'):
        copy_file(BASE / 'closeout_20261004/viz' / name, 'current_evidence/' + name,'actual complete raw receipts, keys and source checks')
    mini = HERE.parent / 'minicpm_sid'
    for name in ('MINICPM_SID_SCOPE.md','native_visual_mapping_inventory.json'):
        copy_file(mini / name,'SID_applicability/minicpm26/' + name,'strict original rank100 architecture evidence; no variant produced')
    q35_original = HERE / 'source_inputs/sid_architecture_scope_20261004.md'
    q35_text = q35_original.read_text(encoding='utf-8').split('## Qwen3.5-4B', 1)[1]
    q35_target = CURRENT / 'SID_applicability/qwen35_4b.md'
    q35_target.parent.mkdir(parents=True, exist_ok=True)
    q35_target.write_text('# Qwen3.5-4B: original SID applicability\n' + q35_text, encoding='utf-8')
    manifest.append(dict(package_path='SID_applicability/qwen35_4b.md',disposition='included',source_archive='',source_member='',
                         source_path='/home/g203-4028/projects/knowledge-deficit-mitigation/docs/supplemental/sid_architecture_scope_20261004.md#Qwen3.5-4B',
                         source_bytes=q35_target.stat().st_size,archive_crc32='',recorded_sha256='',reason='exact Qwen3.5 subsection; unrelated interim Gemma progress excluded'))
    for model, scope in (('minicpm26','857 of 2424 images supply only 64 visual tokens; fixed rank100 cannot apply'),
                         ('qwen35_4b','second decoder block is recurrent linear attention; original softmax attention and visual-key mask unavailable')):
        status.append(dict(stage='native_SID_architecture',model=model,condition_count=0,completed_rows=0,expected_rows='',
                           semantics_pending_rows='',reference_pending_rows='',status='not_applicable',scope=scope,
                           source='SID_applicability/'))
    status.append(dict(stage='native_SID_Gemma_completion',model='gemma3_4b',condition_count=1,
                       completed_rows=2424,expected_rows=2424,semantics_pending_rows=0,reference_pending_rows=0,
                       status='closed',scope='original-author operator; C1259 W1165 A0; J=1259/2424; distinct from prior six-model SID full runs',
                       source='addendum/gemma3_sid/receipt.json'))
    source_counts = json.loads((HERE / 'source_inputs/scientific_source_counts.json').read_text(encoding='utf-8'))
    for item in source_counts[:2]:
        for group in item['groups']:
            assert group['rows']==2424
            status.append(dict(stage='Food_'+group['split']+'_rank_and_ten_attempt_reference',model=group['model'],condition_count=11,
                               completed_rows=group['rows'],expected_rows=group['rows'],semantics_pending_rows=0,reference_pending_rows=0,
                               status='closed',scope='one reference record per image; rank and ten-attempt correct count retained',source=item['member']))
    viz_status = csv_rows((HERE / 'VIZ_INDEPENDENT_STATUS.csv').read_bytes())
    for row in viz_status:
        status.append(dict(stage='VizWiz_eval3501_ten_independent',model=row['model'],condition_count=10,
                           completed_rows=row['generated_rows'],expected_rows=row['expected_rows'],
                           semantics_pending_rows=row['semantics_pending_rows'],reference_pending_rows='',
                           status='generation_complete_semantics_open',scope=row['scope'],source='VIZ_INDEPENDENT_STATUS.csv'))
    addendum = HERE / 'addendum'
    if addendum.exists():
        for source in sorted(addendum.rglob('*')):
            if source.is_file():
                copy_file(source,'addendum/' + source.relative_to(addendum).as_posix(),'new full SID accepted results from current owner')
    write_csv(CURRENT / 'DATA_STATUS.csv', status)
    write_csv(CURRENT / 'MANIFEST.csv', manifest)
    summary = json.loads((HERE / 'VIZ_INDEPENDENT_SUMMARY.json').read_text(encoding='utf-8'))
    readme = f'''# KDM 九模型当前研究数据包

本包包含已有全量主比较、方法机制、开发选择、官方 VizWiz 512 题、有限归因、更新后的 DoLa/SID 同 101 题比较，以及新增 CDA 与重放测量。正文与重绘图另见同级论文包。

## 阅读顺序

1. `DATA_STATUS.csv`：每个阶段的实际分母与完成范围。
2. `main/metrics_all_complete.csv`、`Food_main70_actual_operating_points.csv`：Food 主比较及实际开发集选择；四表达的最佳观测值与开发选择保持不同标记。
3. `vizwiz_final/main45_metrics.csv`、`vizwiz_final/all56_metrics.csv`：官方 512 题回答分数和弃权指标。
4. `supplementary/appendix_supplement_v3.md`：新补入附录的结果与条件。
5. `MANIFEST.csv`：每个文件的原包成员、服务器来源、既有哈希或 CRC；大文件可由相同来源取回。

## 已闭合的科学数据

- Food 有 124 个主比较条件与 360 个 VCD/M3ID 矩阵和 ref-off 条件，共 484 条件、1,173,216 个已评分模型回答，每条件 2,424 题。主表共有 70 个实际工作点（原 69 点加 Gemma SID），见 `Food_main70_actual_operating_points.csv`。SID 的探索性 4×4 矩阵未纳入。
- Food 的 IP 开发选择使用同一 404 题，保留全部候选及实际选择配置。九模型 IP-VCD/IP-M3ID 共 18 个选定配置；另保留核心五模型先前四方法开发选择的 20 个工作点。
- VizWiz 512 题共有 56 个已决条件、28,672 行：45 个主比较条件及 11 个机制条件。每条件 166 个官方不可回答、346 个可回答输入。质量保留连续官方分数。
- DoLa 与已适用 SID 的作者实现更新有 15 个同 101 题条件、1,515 行。此表分母为 101，不代替旧全量 2,424 题条件。CDA 有 108 条路径、803 个词元位置；硬件/缓存次序重放 36 个位置记录；固定噪声复测 4 个记录。
- MiniCPM 与 Qwen3.5 的原始 SID 不适用证据见 `SID_applicability/`。Gemma 原 SID 完整 2,424 题已纳入主表：C={int(float(gemma['C'])):,}、W={int(float(gemma['W_decided'])):,}、A={int(float(gemma['A']))}、J={float(gemma['J'])*100:.5f}%，评分与参考未决为 0。其第二块原生 attention、固定 rank100 与作者算子身份在 `addendum/gemma3_sid/` 明确记录。旧六模型 SID 登记全量与更新算子的 101 题比较分别保留，不混成同一执行身份。

## 仍未闭合的范围

完整 VizWiz 独立作答已生成 315,090 行（九模型各 35,010），与官方 512 题主实验分别记账。四扩展模型的 74,230 个全问答键中，29,972 个已有接受的语义结论，44,258 个仍未决，对应 85,986 条可复用回答与 54,054 条待语义回答。新增实际接受的 2,181 个问答已计入，草稿不计。

核心五模型 175,050 条均已生成。已查中央评分只覆盖 Mistral 两个分片的 17,460 条，其中 3,349 条已决、14,111 条未决；剩余 157,590 条尚无全量 CPU 语义扫描回执。空白计数表示尚未测量，不表示零缺口。按模型唯一 QA 不能跨模型直接求和。详细范围见 `VIZ_INDEPENDENT_STATUS.csv`。

## 逐样本与来源

`main/main_scores.parquet` 与词典保留原 123 条件逐样本主评分，新增 Gemma SID 分区位于 `addendum/gemma3_sid/score_rows.parquet`；`mechanism/` 保留全部注册矩阵与 ref-off；`support/references/` 保留 Food 排名、试答计数和来源；`support/vizwiz/core5_512/new_scores.parquet` 与 `vizwiz_final/viz25_compact_scores.csv.gz` 合计保留全部 56 条件逐样本评分。后者仅投影已有记录，不重评分，保留实际回答、词元、配置、seed、原始连续分数及原完整记录的一基行号。Gemma 追加前的 483 条件与 69 工作点源表保留在原始归档，当前包只列新 484 条件与 70 工作点总表。

完整原记录中较大的重复嵌套裁定历史、自然参考的完整 QA 字典及 Food 原始 QA 绑定表在 `MANIFEST.csv` 标为 `remote_only`，服务器原件和原始输入 zip 均保持。包中来源行号可定位全部字段。当前文档只保留这一份说明和状态，历史进度快照不混入。

打包时间（UTC）：{created}。本轮数据包组装没有 GPU 推理、训练、新语义判断或评分变更。
'''
    (CURRENT / 'README.md').write_text(readme, encoding='utf-8')
    build_receipt = dict(created_utc=created, source_archive=str(SOURCE), source_archive_members=389,
                         source_original_unchanged=True, gpu_calls=0, scientific_values_recomputed=False,
                         dispositions=dict(Counter(row['disposition'] for row in manifest)),
                         included_files=sum(row['disposition']=='included' for row in manifest),
                         manifest_includes_full_original_sources=True, viz_coverage=summary)
    (CURRENT / 'BUILD_RECEIPT.json').write_text(json.dumps(build_receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    expected = {row['package_path'] for row in manifest if row['disposition']=='included'}
    expected.update({'DATA_STATUS.csv','MANIFEST.csv','README.md','BUILD_RECEIPT.json'})
    actual = {path.relative_to(CURRENT).as_posix() for path in CURRENT.rglob('*') if path.is_file()}
    assert actual == expected, ('unexpected or missing current files',actual ^ expected)
    target = HERE / 'KDM_Nine_Current_Data_20261004.zip'
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(CURRENT.rglob('*')):
            if path.is_file():
                package.write(path,path.relative_to(CURRENT).as_posix())
    with ZipFile(target) as package:
        assert len(package.namelist()) == len(actual)
        assert len(set(package.namelist())) == len(actual)
    print(json.dumps({'files':len(actual),'zip_bytes':target.stat().st_size,'zip':str(target),
                      'source_files_preserved':True,'status_rows':len(status)},ensure_ascii=False))


if __name__ == '__main__':
    main()
