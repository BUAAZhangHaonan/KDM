"""Paths, strict JSONL handling and paired reliability metrics for stage 9."""
from __future__ import annotations
import json, os, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS=('q4b','q9b','llava16','internvl4b')
METHODS=('vcd','m3id','dola','deco')
ALLOWED_GPUS={0,1,4,5}


def initialize(root: Path=ROOT) -> Path:
    root=Path(root).resolve()
    for env,sub in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),
                    ('MPLCONFIGDIR','mpl'),('NLTK_DATA','nltk'),('TMPDIR','tmp')]:
        p=root/'cache'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[env]=str(p)
    os.environ['HF_HUB_DISABLE_XET']='1'
    return root


def inside(root:Path,path:Path) -> Path:
    root=root.resolve();p=Path(path).resolve()
    if not p.is_relative_to(root):raise ValueError(f'Output outside project: {p}')
    return p


def read_jsonl(path:Path):
    with Path(path).open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError as e:raise ValueError(f'{path}:{line_no}: invalid JSON') from e
            yield r


def write_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def stable_key(*parts) -> str:
    return hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()


def load_records(path:Path,require_complete:bool=True) -> dict:
    """Latest valid record wins only when duplicate contents are identical.
    Old error attempts followed by a valid record are allowed but counted by
    the caller. Unresolved errors and inconsistent duplicates raise.
    """
    rows={};errors={}
    for r in read_jsonl(path):
        if r.get('method')=='roundA':continue
        key=r.get('key') or f"{r.get('model')}:{r.get('condition','original')}:{r['file']}:{r['method']}"
        if 'error' in r:errors[key]=r['error'];continue
        if key in rows and rows[key]!=r:raise ValueError(f'Conflicting duplicate: {key}')
        rows[key]=r
        errors.pop(key,None)
    if require_complete and errors:raise RuntimeError(f'{len(errors)} unresolved error records in {path}')
    return rows


def confidence(r:dict,kind:str='sequence') -> float | None:
    if r.get('abstained') or r.get('outcome')=='abstain':return None
    if kind=='first':
        x=r.get('answer_maxp');return None if x is None else float(x)
    probs=r.get('step_probs')
    if not probs:return None
    a=np.asarray(probs,dtype=float)
    if np.any((a<0)|(a>1)) or not np.isfinite(a).all():raise ValueError('Invalid recorded probabilities.')
    # Sequence log probability is a ranking score, NOT semantic confidence.
    return float(np.log(a).sum()) if np.all(a>0) else -float('inf')


def ece(conf,correct,bins:int=10) -> float:
    c=np.asarray(conf,float);y=np.asarray(correct,float)
    if len(c)!=len(y) or not len(c) or not np.isfinite(c).all() or np.any((c<0)|(c>1)):
        raise ValueError('ECE requires matched finite probability vectors.')
    idx=np.minimum((c*bins).astype(int),bins-1)
    return float(sum(np.mean(idx==b)*abs(c[idx==b].mean()-y[idx==b].mean())
                     for b in range(bins) if np.any(idx==b)))


def transition_counts(direct,method) -> dict:
    d=np.asarray(direct,bool);m=np.asarray(method,bool)
    if d.shape!=m.shape or not len(d):raise ValueError('Paired nonempty outcomes required.')
    c=int(np.sum(~d&m));b=int(np.sum(d&~m));n=len(d)
    return dict(n=n,corrected_n=c,induced_n=b,acc_direct=float(d.mean()),acc_method=float(m.mean()),
                delta_accuracy=float((c-b)/n),
                correction_rate=(float(c/np.sum(~d)) if np.any(~d) else None),
                induced_error_rate=(float(b/np.sum(d)) if np.any(d) else None))


def risk_at_coverage(scores,correct,coverage:float) -> float:
    """Fixed count budget; use expected fractional selection at a tied boundary.
    No label-dependent ordering or arbitrary file-name ordering at ties.
    """
    s=np.asarray(scores,float);y=np.asarray(correct,bool)
    if s.shape!=y.shape or not len(s) or np.isnan(s).any() or not 0<coverage<=1:
        raise ValueError('Invalid risk inputs.')
    k=max(1,int(np.ceil(coverage*len(s))))
    cutoff=np.sort(s)[::-1][k-1]
    above=s>cutoff;tie=s==cutoff
    need=k-int(above.sum())
    errors=float(np.sum(~y[above]))+need*float(np.mean(~y[tie]))
    return errors/k


def cluster_bootstrap_pair(values,clusters,statistic,replicates=2000,seed=910):
    """Resample category clusters jointly for both configurations."""
    v=np.asarray(values);g=np.asarray(clusters)
    levels=np.unique(g)
    if len(levels)<2:raise ValueError('At least two category clusters are required.')
    indices=[np.flatnonzero(g==c) for c in levels]
    rng=np.random.default_rng(seed);out=[]
    for _ in range(replicates):
        pick=rng.integers(0,len(indices),len(indices))
        ids=np.concatenate([indices[i] for i in pick]);out.append(float(statistic(v[ids])))
    return float(np.quantile(out,.025)),float(np.quantile(out,.975))
