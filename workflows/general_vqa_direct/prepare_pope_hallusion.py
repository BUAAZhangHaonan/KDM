#!/usr/bin/env python3
"""Freeze official POPE1000 / HallusionBench image951 and fetch their exact images.

No model generation, no target-dependent image replacement, and no paid API calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.request

POPE_COMMIT = '08d957b917e5a378a2f99d35b6293c536a66298b'
HALL_COMMIT = '744007c232c292942c7f80eb61edb2465482da31'
HALL_HF_COMMIT = '2067dfdce5abc7efdbb0927f4721ae282f0b6c51'
SEED = 20261004
COUNTS = {'random': 332, 'popular': 334, 'adversarial': 334}

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze(path, value, jsonl=False):
    text = (''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in value)
            if jsonl else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    data = text.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Frozen existing file differs: {path}')
    else:
        with path.open('xb') as f:
            f.write(data)
    return sha(path)

def fetch(url, path):
    """TLS verification remains enabled; cache contains exact original bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size == 0:
            raise ValueError(f'Empty cache: {path}')
        return {'url': url, 'sha256': sha(path), 'bytes': path.stat().st_size}
    part = path.with_name(path.name + '.part')
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'KDM-official-dataset-preparation/1'})
            # Public COCO's published endpoint is HTTP; no certificate bypass is used.
            opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                      if url.startswith('http://images.cocodataset.org/')
                      else urllib.request.build_opener())
            with opener.open(request, timeout=45) as response, part.open('wb') as f:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            if not part.stat().st_size:
                raise ValueError(f'Empty response: {url}')
            part.replace(path)
            return {'url': url, 'sha256': sha(path), 'bytes': path.stat().st_size}
        except Exception as exc:
            error = exc
            if part.exists():
                part.unlink()
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                break
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Download failed without replacing sample: {url}: {error}')

def image_info(path):
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size, mode = list(image.size), image.mode
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'dimensions': size, 'mode': mode}

def sources(root, dataset):
    folder = root / 'data/general_vqa_direct_20261004/source_pope_hallusion' / dataset
    repo, commit = ('RUCAIBox/POPE', POPE_COMMIT) if dataset == 'pope' else ('tianyi-lab/HallusionBench', HALL_COMMIT)
    names = (['README.md', 'evaluate.py'] + [f'output/coco/coco_pope_{x}.json' for x in COUNTS]
             if dataset == 'pope' else ['README.md', 'HallusionBench.json', 'evaluation.py', 'utils.py', 'gpt4v_benchmark.py'])
    records = {}
    for name in names:
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
        path = folder / name
        records[name] = dict(fetch(url, path), local_path=str(path.relative_to(root)))
    freeze(folder / 'sources.json', {'repository': repo, 'commit': commit, 'members': records})
    return folder, records

def pope_rows(root):
    folder, records = sources(root, 'pope')
    selected, all_counts = [], {}
    for typ, n in COUNTS.items():
        name = f'output/coco/coco_pope_{typ}.json'
        raw = [json.loads(x) for x in (folder / name).read_text(encoding='utf-8').splitlines() if x.strip()]
        assert len(raw) == 3000 and len({str(x['question_id']) for x in raw}) == 3000
        assert Counter(x['label'] for x in raw) == {'yes': 1500, 'no': 1500}
        all_counts[typ] = len(raw)
        ranked = []
        for source_line, row in enumerate(raw, 1):
            key = json.dumps([SEED, 'pope', typ, str(row['question_id'])], separators=(',', ':'))
            ranked.append((hashlib.sha256(key.encode()).hexdigest(), source_line, row))
        for label in ['yes', 'no']:
            subset = sorted(x for x in ranked if x[2]['label'] == label)[:n // 2]
            for rank, (selection_hash, source_line, row) in enumerate(subset, 1):
                source_id = f"{typ}/{row['question_id']}"
                image_rel = Path('data/general_vqa_direct_20261004/images/pope') / row['image']
                selected.append({'id': 'pope:' + source_id, 'dataset': 'pope', 'split': 'eval',
                    'source_split': f'original_COCO_val2014_{typ}', 'source_id': source_id,
                    'question_id': row['question_id'], 'type': typ, 'question': row['text'],
                    'gold': row['label'], 'source_record': row, 'source_row': source_line,
                    'question_source_uri': records[name]['url'], 'question_source_sha256': records[name]['sha256'],
                    'image_paths': [image_rel.as_posix()],
                    'image_source_uri': ['http://images.cocodataset.org/val2014/' + row['image']],
                    'selection_seed': SEED, 'selection_hash': selection_hash, 'selection_rank_within_label': rank})
    selected.sort(key=lambda x: (x['dataset'], x['type'], int(x['question_id'])))
    assert len(selected) == 1000 and len({x['id'] for x in selected}) == 1000
    assert Counter((x['type'], x['gold']) for x in selected) == {(t, g): n // 2 for t, n in COUNTS.items() for g in ['yes', 'no']}
    return selected, {'source_questions_by_type': all_counts, 'selected_questions_by_type': COUNTS,
        'selection_rule': 'Within each type/yes-no stratum, take smallest SHA256(JSON([20261004,"pope",type,str(question_id)],separators=(",",":"))); final order dataset/type/numeric question_id.',
        'original_not_POPEv2': True, 'source_commit': POPE_COMMIT}

def hall_rows(root):
    folder, records = sources(root, 'hallusionbench')
    raw = json.loads((folder / 'HallusionBench.json').read_text(encoding='utf-8'))
    assert len(raw) == 1129 and Counter(str(x['visual_input']) for x in raw) == {'0': 178, '1': 447, '2': 504}
    selected = []
    for source_row, row in enumerate(raw, 1):
        if str(row['visual_input']) == '0':
            assert row['filename'] is None
            continue
        assert str(row['gt_answer']) in {'0', '1'}
        suffix = f"{row['category']}/{row['subcategory']}/{row['set_id']}_{row['figure_id']}.png"
        assert row['filename'].replace('\\', '/').endswith('/' + suffix)
        source_id = '/'.join(str(row[k]) for k in ['category', 'subcategory', 'visual_input', 'set_id', 'figure_id', 'question_id'])
        selected.append({'id': 'hallusionbench:' + source_id, 'dataset': 'hallusionbench', 'split': 'eval',
            'source_split': 'official_main_all_visual_inputs_1_or_2', 'source_id': source_id,
            'question': row['question'], 'gold': 'yes' if str(row['gt_answer']) == '1' else 'no',
            **{k: row[k] for k in ['category', 'subcategory', 'visual_input', 'set_id', 'figure_id', 'question_id', 'sample_note', 'gt_answer', 'gt_answer_details', 'filename']},
            'source_record': row, 'source_row': source_row,
            'question_source_uri': records['HallusionBench.json']['url'],
            'question_source_sha256': records['HallusionBench.json']['sha256'],
            'image_paths': ['data/general_vqa_direct_20261004/images/hallusionbench/' + suffix],
            'image_source_uri': [f'https://huggingface.co/datasets/rayguan/HallusionBench/resolve/{HALL_HF_COMMIT}/data/' + suffix],
            'image_source_revision': HALL_HF_COMMIT})
    assert len(selected) == 951 and len({x['id'] for x in selected}) == 951
    return selected, {'official_total_questions': 1129, 'excluded_visual_input0': 178,
        'selected_visual_input_counts': {'1': 447, '2': 504},
        'category_counts': dict(Counter(x['category'] for x in selected)),
        'subcategory_counts': dict(Counter(x['category'] + '/' + x['subcategory'] for x in selected)),
        'selection_rule': 'All and only official visual_input 1 or 2; original source order; no sampling or replacement.',
        'source_commit': HALL_COMMIT, 'image_source_revision': HALL_HF_COMMIT}

def resolve_hallusion_file_case(root, rows):
    """Match the author's pinned file list; do not select a different image."""
    info_path = root / 'data/general_vqa_direct_20261004/source_pope_hallusion/hallusionbench/hf_repository_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8-sig'))
    assert info['sha'] == HALL_HF_COMMIT
    names = {x['rfilename'] for x in info['siblings']}
    prefix = f'https://huggingface.co/datasets/rayguan/HallusionBench/resolve/{HALL_HF_COMMIT}/'
    mapping = {}
    for row in rows:
        url = row['image_source_uri'][0]
        name = url.removeprefix(prefix)
        assert name != url
        if name not in names:
            matches = [x for x in names if x.casefold() == name.casefold()]
            assert len(matches) == 1, (name, matches)
            actual = matches[0]
            assert Path(actual).with_suffix('') == Path(name).with_suffix('')
            assert Path(actual).suffix == '.PNG' and Path(name).suffix == '.png'
            row['image_source_uri_requested'] = row['image_source_uri']
            row['image_source_uri'] = [prefix + actual]
            row['image_source_path_case_mapping'] = {'official_question_filename': row['filename'], 'author_hf_filename': actual}
            mapping[name] = actual
    assert len(mapping) == 9
    return {'author_file_listing_sha256': sha(info_path), 'filename_extension_case_mapping': mapping,
            'filename_mapping_changes_question_or_image_identity': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', choices=['pope', 'hallusionbench'], required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--selection-only', action='store_true')
    p.add_argument('--existing-images-only', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    base = root / 'data/general_vqa_direct_20261004'
    rows, details = (pope_rows if args.dataset == 'pope' else hall_rows)(root)
    selection = base / f'selection_{args.dataset}.jsonl'
    selection_sha = freeze(selection, rows, jsonl=True)
    unique = {}
    for row in rows:
        path, url = row['image_paths'][0], row['image_source_uri'][0]
        if path in unique:
            assert unique[path] == url
        unique[path] = url
    print(json.dumps({'dataset': args.dataset, 'selection_frozen': True, 'rows': len(rows),
        'unique_images': len(unique), 'selection_sha256': selection_sha}), flush=True)
    if args.selection_only:
        return
    if args.dataset == 'hallusionbench':
        details.update(resolve_hallusion_file_case(root, rows))
        unique = {row['image_paths'][0]: row['image_source_uri'][0] for row in rows}
    started = time.monotonic()
    image_metadata, errors = {}, []
    def load_one(item):
        path, url = item
        target = root / path
        if args.existing_images_only:
            if not target.is_file():
                raise FileNotFoundError(target)
        else:
            fetch(url, target)
        return path, image_info(target)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_one, item): item for item in unique.items()}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                path, info = future.result()
                image_metadata[path] = info
            except Exception as exc:
                errors.append({'image_path': futures[future][0], 'url': futures[future][1], 'error': str(exc)})
            if index % 50 == 0 or index == len(unique):
                print(json.dumps({'dataset': args.dataset, 'images_processed': index, 'total': len(unique),
                    'valid': len(image_metadata), 'failed': len(errors), 'elapsed_s': round(time.monotonic()-started, 2)}), flush=True)
    if errors:
        failure = base / f'{args.dataset}_download_failure_{time.time_ns()}.json'
        freeze(failure, {'dataset': args.dataset, 'errors': errors, 'selection_sha256': selection_sha, 'no_sample_replacement': True})
        raise RuntimeError(f'{len(errors)} fixed images failed; no complete manifest published: {failure}')
    for row in rows:
        item = image_metadata[row['image_paths'][0]]
        row['image_sha256'] = [item['sha256']]
        row['image_dimensions'] = [item['dimensions']]
        row['image_bytes'] = [item['bytes']]
    manifest = base / f'manifest_{args.dataset}.jsonl'
    digest = freeze(manifest, rows, jsonl=True)
    receipt = dict(dataset=args.dataset, ready=True, rows=len(rows), unique_keys=len({x['id'] for x in rows}),
        unique_images=len(unique), zero_missing_images=True, all_images_decode=True, zero_duplicate_keys=True,
        image_sha256_computed=True, selection_sha256=selection_sha, manifest_sha256=digest,
        source_details=details, prompt_status='root_to_freeze_official_task_prompt', generation_started=False,
        actual_command=' '.join(__import__('sys').argv), completed_utc=now(), elapsed_s=round(time.monotonic()-started, 2))
    receipt_path = base / f'prepare_{args.dataset}_receipt.json'
    if not receipt_path.exists():
        freeze(receipt_path, receipt)
    else:
        previous = json.loads(receipt_path.read_text())
        assert previous['manifest_sha256'] == digest and previous['ready']
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
