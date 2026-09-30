"""Refresh the compact checkpoint from completed assets and actual run receipts."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json


def update(run):
    folder = ROOT / 'outputs/supplemental/remaining11' / run
    with (folder / 'assets/asset_gaps.csv').open() as source:
        gaps = list(csv.DictReader(source))
    totals = {}
    for stage in ('formal', 'independent', 'candidate'):
        rows = [row for row in gaps if row['stage'] == stage]
        totals[stage] = {field: sum(int(row[field]) for row in rows)
                         for field in ('expected_rows', 'reusable_rows', 'remaining_rows')}
    jobs = []
    for path in sorted((folder / 'records').rglob('progress.json')):
        record = json.loads(path.read_text())
        jobs.append({'record': str(path.relative_to(ROOT)), **record})
    score_files = sorted((folder / 'scores').rglob('summary.json'))
    scoring = [{'path': str(path.relative_to(ROOT)), **json.loads(path.read_text())}
               for path in score_files]
    state = {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'run': run, 'task': 'KDM remaining11 supplemental',
        'internal_submission_date': '2026-10-11',
        'paper_target': 'NAACL 2027 October ARR (user-provided target)',
        'frozen_five_models': {'rows': 853248, 'conditions': 352, 'status': 'retained'},
        'asset_gap_table': str((folder / 'assets/asset_gaps.csv').relative_to(ROOT)),
        'initial_actual_asset_totals': totals,
        'jobs': jobs, 'scoring_checkpoints': scoring,
        'label_review_pending': 'Read current immutable annotation batches and scoring summary',
        'next_steps': ['finish missing Food-101 generation', 'score and review completed parts',
                       'merge uniform references and full registered method conditions',
                       'derive controls, paired statistics, supplemental figures and appendix'],
        'vizwiz_registration': str((folder / 'assets/registered_inclusion.json').relative_to(ROOT)),
        'k100_admission': 'Existing tf553 path has broken Python 3.12 site-packages binding; native formal admission unavailable',
        'completion_claim': False,
    }
    atomic_json(folder / 'CURRENT_STATE.json', state)
    atomic_json(ROOT / 'CURRENT_STATE.json', state)
    checklist = ['# KDM remaining11 运行与验收', '',
                 '- [x] 阅读最终协议和历史定位索引。',
                 '- [x] 一次资源、GPU UUID、真实进程与锁检查。',
                 '- [x] 逐模型资产覆盖、身份和剩余键核对。',
                 '- [ ] 缺失生成完成；每条件和类别配额完整。',
                 '- [ ] 主评分、弃权与统一参考GT全部已决。',
                 '- [ ] 必要Luna标注与root复核合并。',
                 '- [ ] 完整方法比较、控制、配对统计与真实图表。',
                 '- [ ] 补充文档、论文附录与细粒度提交交付。', '']
    (folder / 'RUN_CHECKLIST.md').write_text('\n'.join(checklist))
    print(json.dumps({'totals': totals, 'jobs': len(jobs),
                      'state': str(folder / 'CURRENT_STATE.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run_20260930_140337')
    update(parser.parse_args().run)
