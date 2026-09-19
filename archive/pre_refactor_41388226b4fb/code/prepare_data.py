"""Prepare evaluation data from food101 (via HF datasets): per-class pools,
seeded half/half split into grouping / evaluation parts, image export.

Caps: 24 images per class per part (48 total per class). Existence task later
reuses evaluation-part images. Deterministic seed.
"""
import os, sys, json, random, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DATA = ROOT / 'data'
IMG_DIR = DATA / 'images'
SEED = 42
PER_PART = 24


def main():
    from datasets import load_dataset
    ds = load_dataset('ethz/food101')
    names = ds['train'].features['label'].names
    print(f'source: ethz/food101 | splits: {[(k, len(v)) for k, v in ds.items()]} | classes: {len(names)}')

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    by_class = defaultdict(list)
    for split in ds:
        for ex in ds[split]:
            by_class[names[ex['label']]].append((split, ex['image_id'] if 'image_id' in ex else None, ex['image']))

    counts = {c: len(v) for c, v in by_class.items()}
    print(f'images per class: min={min(counts.values())} max={max(counts.values())} '
          f'median={sorted(counts.values())[len(counts)//2]}')

    rng = random.Random(SEED)
    manifest = []
    for c in sorted(by_class):
        pool = by_class[c]
        rng.shuffle(pool)
        take = pool[:2 * PER_PART]
        for i, (split, iid, img) in enumerate(take):
            part = 'group' if i < PER_PART else 'eval'
            fname = f"{c}_{part}_{i:03d}.jpg"
            dest = IMG_DIR / fname
            if not dest.exists():
                img.convert('RGB').save(dest, quality=95)
            manifest.append({'class': c, 'part': part, 'idx': i,
                             'image_path': str(dest), 'file': fname})

    with open(DATA / 'samples_manifest.jsonl', 'w') as f:
        for m in manifest:
            f.write(json.dumps(m) + '\n')

    fp = hashlib.sha256()
    for m in manifest:
        fp.update(m['file'].encode())
    report = {
        'source': 'ethz/food101', 'license': 'HF: ethz/food101 (Food-101, MIT/README at source)',
        'n_classes': len(names),
        'per_class_counts': {'min': min(counts.values()), 'max': max(counts.values())},
        'per_part_per_class': PER_PART,
        'n_group': sum(1 for m in manifest if m['part'] == 'group'),
        'n_eval': sum(1 for m in manifest if m['part'] == 'eval'),
        'manifest_sha256': fp.hexdigest(),
        'seed': SEED,
    }
    json.dump(report, open(DATA / 'data_report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))

    # contact sheet of 20 random eval images
    from PIL import Image, ImageDraw, ImageFont
    picks = rng.sample([m for m in manifest if m['part'] == 'eval'], 20)
    fig_dir = ROOT / 'outputs' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols, cell = 5, 300
    sheet = Image.new('RGB', (cols * cell, 4 * (cell + 22)), 'white')
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, m in enumerate(picks):
        im = Image.open(m['image_path']).convert('RGB')
        im.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 22)
        sheet.paste(im, (x + (cell - im.width) // 2, y))
        d.text((x + 4, y + cell + 2), m['class'][:30], fill='black', font=font)
    sheet.save(fig_dir / 'contact_sheet_20.png')
    print('contact sheet saved')


if __name__ == '__main__':
    main()
