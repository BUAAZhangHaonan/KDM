"""Freeze a balanced, paired visual-evidence manifest from existing Food-101 files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from stage9_common import ROOT,initialize,inside,read_jsonl,stable_key,write_json

STRATA={'q9b':'data/strata_q9b.json','llava16':'data/strata_llava16_food101.json'}


def prepare(root:Path,per_class=4,geometry_per_class=2,model_names=('q9b','llava16')):
    if not 1<=geometry_per_class<=per_class<=24:raise ValueError('Invalid balanced sample counts.')
    manifest=list(read_jsonl(root/'data/samples_manifest.jsonl'))
    output=root/'data/stage9';output.mkdir(parents=True,exist_ok=True)
    selected=[]
    for model in model_names:
        strata=json.loads((root/STRATA[model]).read_text())
        for source,label in [('deficient','low_acc'),('known','high_acc')]:
            classes=strata[source]
            for cls in sorted(classes):
                pool=[r for r in manifest if r['class']==cls and r['part']=='eval']
                pool.sort(key=lambda r:stable_key(910,cls,r['file']))
                if len(pool)<per_class:raise ValueError(f'Only {len(pool)} evaluation images for {cls}')
                for i,r in enumerate(pool[:per_class]):
                    path=Path(r['image_path'])
                    if not path.is_absolute():path=root/path
                    path=path.resolve()
                    if not path.is_relative_to(root.resolve()):raise ValueError('Images must be mounted inside project.')
                    if not path.exists():raise FileNotFoundError(path)
                    selected.append({**r,'model':model,'stratum':label,'image_path':str(path),
                                     'geometry':i<geometry_per_class,'stage9_seed':910})
    dest=inside(root,output/'manifest.jsonl')
    with dest.open('w') as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(output/'selection.json',dict(seed=910,per_class=per_class,
        geometry_per_class=geometry_per_class,n=len(selected),
        manifest_sha256=stable_key(dest.read_text()),
        selection_uses_intervention_results=False,conditions=['original','short64','short32']))
    return selected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--per-class',type=int,default=4);ap.add_argument('--geometry-per-class',type=int,default=2)
    a=ap.parse_args();r=prepare(initialize(a.root),a.per_class,a.geometry_per_class)
    print(f'Frozen {len(r)} model-image records.')

if __name__=='__main__':main()
