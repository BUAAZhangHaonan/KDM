"""Stage 3: prepare the second image domain (Stanford Dogs).

Source: Donghyun99/Stanford-Dogs (HF parquet, 120 classes, 20,580 images).
Train+test combined into one per-class pool (the official split has no role
here; our split is by seed). Per class: shuffle seed 42, take 48 -> 24
grouping + 24 evaluation, exported as JPG data/dogs_images/{class}_{part}_{idx:03d}.jpg
+ data/samples_manifest_dogs.jsonl. Requirements checked: >=50 classes,
>=100 images/class (else abort with a report).

Usage:
  ./venv/bin/python code/stage3_prepare_dogs.py
"""
import os, sys, json, random, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
sys.path.insert(0, str(ROOT / 'code'))

REPO = 'Donghyun99/Stanford-Dogs'
N_PER_PART = 24


def main():
    from datasets import load_dataset

    ds = load_dataset(REPO)
    pool = defaultdict(list)
    names = ds['train'].features['label'].names
    for split in ds:
        for ex in ds[split]:
            pool[names[ex['label']]].append(ex['image'])

    def clean(cls):
        # strip WordNet-id prefix like 'n02085620-Chihuahua'
        c = cls.split('-', 1)[1] if (cls.startswith('n0') and '-' in cls) else cls
        return c.strip().lower().replace(' ', '_').replace('-', '_')

    counts = {c: len(v) for c, v in pool.items()}
    bad = {c: n for c, n in counts.items() if n < 100}
    if len(pool) < 50 or bad:
        rep = {'n_classes': len(pool), 'violations': bad}
        json.dump(rep, open(ROOT / 'data' / 'dogs_data_report_FAIL.json', 'w'), indent=1)
        print('REQUIREMENTS FAILED', rep)
        sys.exit(1)

    img_dir = ROOT / 'data' / 'dogs_images'
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in sorted(pool):
        imgs = pool[cls]
        random.Random(42).shuffle(imgs)
        take = imgs[:2 * N_PER_PART]
        cname = clean(cls)
        for part_i, part in enumerate(['group', 'eval']):
            for i in range(N_PER_PART):
                im = take[part_i * N_PER_PART + i].convert('RGB')
                fn = f"{cname}_{part}_{i:03d}.jpg"
                fp = img_dir / fn
                im.save(fp, 'JPEG', quality=92)
                rows.append({'file': fn, 'class': cname, 'part': part,
                             'image_path': str(fp)})
    with open(ROOT / 'data' / 'samples_manifest_dogs.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    rep = {'source': REPO, 'n_classes': len(pool), 'per_class_min': min(counts.values()),
           'per_class_max': max(counts.values()), 'n_group': 24 * len(pool),
           'n_eval': 24 * len(pool), 'seed': 42}
    json.dump(rep, open(ROOT / 'data' / 'dogs_data_report.json', 'w'), indent=1)
    print('OK', rep)


if __name__ == '__main__':
    main()
