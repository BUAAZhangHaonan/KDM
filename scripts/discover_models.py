#!/usr/bin/env python3
"""Resolve declared checkpoint basenames; ambiguous matches need explicit review."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from kdm.io import atomic_json,within,file_hash

def checkpoint_files(path):
    path=Path(path)
    indexes=list(path.glob('*safetensors.index.json'))+list(path.glob('pytorch_model*.index.json'))
    if len(indexes)>1: raise ValueError(f'Multiple weight indexes: {path}')
    if indexes:
        names=sorted(set(json.loads(indexes[0].read_text())['weight_map'].values()))
        weights=[path/n for n in names]
    else: weights=sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
    if not weights: raise ValueError(f'No checkpoint weights: {path}')
    missing=[str(w) for w in weights if not w.is_file() or w.stat().st_size==0]
    if missing: raise ValueError(f'Missing or empty checkpoint shards: {missing}')
    return weights



def discover(candidates,roots):
    rows=[]
    for item in candidates:
        matches=[]
        for root in roots:
            for name in item['directory_names']:
                p=Path(root)/name
                if (p/'config.json').is_file():matches.append(str(p.resolve()))
        matches=sorted(set(matches))
        error=None
        if len(matches)==1:
            try: checkpoint_files(matches[0])
            except ValueError as exc: error=str(exc)
        rows.append({**item,'matches':matches,'resolved_path':matches[0] if len(matches)==1 else None,
                     'config_sha256':file_hash(Path(matches[0])/'config.json') if len(matches)==1 else None,
                     'checkpoint_error':error,
                     'resolution':'incomplete' if error else 'resolved' if len(matches)==1 else ('missing' if not matches else 'ambiguous')})
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--candidates',required=True)
    p.add_argument('--model-roots',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=discover(json.load(open(a.candidates)),a.model_roots);atomic_json(within(a.project,a.out),rows)
    if any(x['resolution']!='resolved' for x in rows):raise SystemExit('Model inventory needs explicit resolution')
