"""Build the evaluation dataset: select images per LVIS frequency tier, download them,
construct naming + existence samples with gold answers and negatives.

Usage: venv/bin/python code/stage1_build_dataset.py [--target 600] [--oversample 1.2]
All outputs inside project dir. Deterministic seed.
"""
import os, sys, json, random, argparse, hashlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
IMG_DIR = DATA / 'images'
SEED = 42
AREA_FRAC = 0.15          # target instance area / image area threshold
TARGET_DEFAULT = 600
TIERS = {'high': 'f', 'low': 'r'}


def load_annotations():
    val = json.load(open(DATA / 'lvis_v1_val.json'))
    files = {'val': val}
    p = DATA / 'lvis_v1_train.json'
    if p.exists():
        files['train'] = json.load(open(p))
    else:
        print('WARNING: train annotations not present; rare pool limited to val')
    return files


def candidates_from(split, d, tier_code):
    """Images whose dominant (largest-area) annotation belongs to `tier_code`,
    with exactly one category of that tier present."""
    cats = {c['id']: c for c in d['categories']}
    tier_cats = {cid for cid, c in cats.items() if c['frequency'] == tier_code}
    images = {im['id']: im for im in d['images']}
    anns_by_img = {}
    for a in d['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    out = []
    for img_id, anns in anns_by_img.items():
        if not anns:
            continue
        biggest = max(anns, key=lambda a: a['area'])
        cat_id = biggest['category_id']
        if cat_id not in tier_cats:
            continue
        tier_present = {a['category_id'] for a in anns if a['category_id'] in tier_cats}
        if len(tier_present) != 1 or cat_id not in tier_present:
            continue
        im = images[img_id]
        area_frac = biggest['area'] / (im['height'] * im['width'])
        if area_frac < AREA_FRAC:
            continue
        out.append({
            'image_id': img_id, 'coco_url': im['coco_url'],
            'neg_category_ids': im['neg_category_ids'] or [],
            'category_id': cat_id, 'name': cats[cat_id]['name'],
            'synonyms': cats[cat_id]['synonyms'], 'synset': cats[cat_id]['synset'],
            'frequency': cats[cat_id]['frequency'], 'area_frac': round(area_frac, 4),
            'source': split, 'height': im['height'], 'width': im['width'],
        })
    return out


def pick_negatives(pool_by_tier, sample, rng, k=1):
    """Pick a same-tier category provably absent from the image (LVIS neg_category_ids),
    else a random same-tier category absent from annotations."""
    tier_cats = pool_by_tier['cat_by_id']
    negs = [c for c in sample['neg_category_ids'] if c in tier_cats and c != sample['category_id']]
    chosen = []
    if negs:
        c = rng.choice(negs)
        chosen.append({'category_id': c, 'name': tier_cats[c]['name'],
                       'synonyms': tier_cats[c]['synonyms'], 'synset': tier_cats[c]['synset']})
    else:  # fallback: random same-tier category
        c = rng.choice([c for c in tier_cats if c != sample['category_id']])
        chosen.append({'category_id': c, 'name': tier_cats[c]['name'],
                       'synonyms': tier_cats[c]['synonyms'], 'synset': tier_cats[c]['synset']})
    return chosen


def download_images(samples, nthreads=16):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in samples if not (IMG_DIR / f"{s['image_id']}.jpg").exists()]
    stats = {'attempted': len(todo), 'ok': 0, 'fail': 0, 'cached': len(samples) - len(todo)}

    def get(s):
        dest = IMG_DIR / f"{s['image_id']}.jpg"
        try:
            req = urllib.request.Request(s['coco_url'].replace('http://', 'https://'),
                                         headers={'User-Agent': 'Mozilla/5.0 (research)'})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, 'wb') as f:
                f.write(r.read())
            return True
        except Exception:
            try:  # retry over http once
                req = urllib.request.Request(s['coco_url'], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as r, open(dest, 'wb') as f:
                    f.write(r.read())
                return True
            except Exception:
                return False

    with ThreadPoolExecutor(nthreads) as ex:
        for r in as_completed([ex.submit(get, s) for s in todo]):
            stats['ok' if r.result() else 'fail'] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=TARGET_DEFAULT)
    ap.add_argument('--over', type=float, default=1.25, help='oversample factor before validity trim')
    args = ap.parse_args()

    rng = random.Random(SEED)
    files = load_annotations()
    report = {}

    pool = {}
    for tier_name, code in TIERS.items():
        cands = []
        seen = set()
        for split in ['val', 'train']:
            if split not in files:
                continue
            cs = candidates_from(split, files[split], code)
            new = [c for c in cs if c['image_id'] not in seen]
            for c in new:
                seen.add(c['image_id'])
            cands.extend(new)
            print(f'{tier_name}: {len(cs)} candidates from {split} (new {len(new)})')
        # round-robin over categories to maximize coverage, val-first ordering preserved
        by_cat = {}
        for c in cands:
            by_cat.setdefault(c['category_id'], []).append(c)
        for v in by_cat.values():
            v.sort(key=lambda x: (x['source'] != 'val', rng.random()))
        cat_ids = list(by_cat)
        rng.shuffle(cat_ids)
        ordered = []
        i = 0
        while True:
            added = 0
            for cid in cat_ids:
                if i < len(by_cat[cid]):
                    ordered.append(by_cat[cid][i]); added += 1
            if added == 0:
                break
            i += 1
        pool[tier_name] = {'ordered': ordered,
                           'cat_by_id': {c['id']: c for split in files.values() for c in split['categories'] if c['frequency'] == code}}
        report[tier_name] = {'n_candidates': len(cands),
                             'n_categories': len(by_cat),
                             'val_first_candidates': sum(1 for c in cands if c['source'] == 'val')}

    # take oversample*target per tier, download, verify, trim
    final = {}
    for tier_name in TIERS:
        want = int(args.target * args.over)
        final[tier_name] = pool[tier_name]['ordered'][:want]

    all_sel = final['high'] + final['low']
    t0 = time.time()
    stats = download_images(all_sel)
    print(f'download stats: {stats} in {time.time()-t0:.0f}s')

    # verify decode
    from PIL import Image
    valid = {'high': [], 'low': []}
    for tier_name in TIERS:
        for s in final[tier_name]:
            p = IMG_DIR / f"{s['image_id']}.jpg"
            try:
                with Image.open(p) as im:
                    im.verify()
                valid[tier_name].append(s)
            except Exception:
                pass
            if len(valid[tier_name]) >= args.target:
                break

    n = min(args.target, len(valid['high']), len(valid['low']))
    dataset = []
    for tier_name in TIERS:
        for s in valid[tier_name][:n]:
            s = dict(s)
            s['tier'] = tier_name
            s['image_path'] = str(IMG_DIR / f"{s['image_id']}.jpg")
            s['negatives'] = pick_negatives(pool[tier_name], s, rng)
            dataset.append(s)

    out = DATA / 'dataset.jsonl'
    with open(out, 'w') as f:
        for s in dataset:
            f.write(json.dumps(s) + '\n')
    report['final_n_per_tier'] = n
    report['tier_cat_coverage'] = {t: len({s['category_id'] for s in dataset if s['tier'] == t}) for t in TIERS}
    report['source_mix'] = {t: {src: sum(1 for s in dataset if s['tier'] == t and s['source'] == src) for src in ['val', 'train']} for t in TIERS}
    report['download'] = stats
    json.dump(report, open(DATA / 'dataset_report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))
    print(f'wrote {len(dataset)} samples -> {out}')


if __name__ == '__main__':
    main()
