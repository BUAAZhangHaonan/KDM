from pathlib import Path
import io,json,math,sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'tests'))
from stage9_geometry import *
from stage9_common import *
from build_fixtures import cache_fixture,image_fixture,FakeEngine,fake_score


def test_real_data_endpoints():
    t=pd.read_csv(ROOT/'data_sample/stage6_core_selected.csv')
    assert len(t)==32 and (t.n==600).all()
    assert np.max(np.abs((t.corrected_n-t.induced_n)/t.n-(t.acc_method-t.acc_direct)))<1e-12
    assert int(((t.stratum=='high_acc')&(t.acc_method<t.acc_direct)).sum())==14
    assert (t.dconf_lo>0).all()
    assert (t.dconf_hi>=t.dconf_unchanged).all() and (t.dconf_unchanged>=t.dconf_lo).all()

@pytest.mark.parametrize('seed',range(12))
def test_interval_matches_bruteforce(seed):
    rng=np.random.default_rng(seed)
    for _ in range(30):
        z=rng.normal(size=7);r=rng.normal(size=7);j=int(rng.integers(7))
        iv=target_interval(z,r,j)
        for a in rng.uniform(0,8,40):
            star=np.where(z>=z.max()+np.log(.1),z+a*(z-r),-np.inf)
            assert iv.contains(float(a))==(int(np.argmax(star))==j)


def test_boundary_ties_and_support():
    # Target index 1 loses at equality; alpha=1 is therefore open.
    iv=target_interval([1,0],[1,-1],1)
    assert not iv.contains(1.) and iv.contains(1.1)
    assert target_interval([4,0],[0,-100],1).reason=='target_outside_support'
    assert target_interval([1,1],[0,0],1).empty
    assert not target_interval([1,1],[0,0],0).empty


def test_interval_intersection_and_json():
    assert Interval(1,2,False,False).intersect(Interval(2,3)).empty
    assert Interval(1,2).witness(4)==1.5
    assert Interval().to_dict()['upper'] is None
    assert Interval(empty=True).witness() is None
    assert Interval(7,9).witness(4) is None

@pytest.mark.parametrize('alpha',[0.,.05,.5,1.,2.,4.])
def test_exact_confidence_and_derivatives(alpha):
    z=np.array([1.,2.,.2,-4]);r=np.array([.5,.1,.4,-1.]);j=1
    d=confidence_terms(z,r,j,alpha)
    assert d['identity_error']<1e-12
    assert d['second_derivative']<=0
    if alpha>0:
        h=1e-5
        l=confidence_terms(z,r,j,alpha-h)['log_gain'];u=confidence_terms(z,r,j,alpha+h)['log_gain']
        assert abs((u-l)/(2*h)-d['derivative'])<1e-8
    assert -1e-10<=d['local_scale_energy_fraction']<=1+1e-10


def test_confidence_mask_and_uniform_reference():
    assert confidence_terms([5,0],[0,-10],1,1)['retained'] is False
    z=np.array([2.,1.,0.]);r=np.zeros(3)
    vals=[confidence_terms(z,r,0,a)['confidence_after'] for a in [0,.5,1,2]]
    assert np.all(np.diff(vals)>0)
    a=confidence_terms(z,r,0,1);b=confidence_terms(z+40,r-17,0,1)
    assert abs(a['log_gain']-b['log_gain'])<1e-12


def test_invalid_geometry_inputs():
    with pytest.raises(ValueError):target_interval([1,float('nan')],[0,0],0)
    with pytest.raises(ValueError):target_interval([1,2],[0,0],0,beta=0)
    with pytest.raises(ValueError):softmax([-np.inf,-np.inf])


def test_metrics_ties_and_transition():
    assert risk_at_coverage([.9,.9,.1],[True,False,True],.2)==.5
    t=transition_counts([True,True,False,False],[True,False,True,False])
    assert t['corrected_n']==1 and t['induced_n']==1 and t['delta_accuracy']==0
    assert ece([.5,.5],[True,False])==0
    assert confidence({'outcome':'abstain','step_probs':[.9]}) is None
    assert confidence({'outcome':'wrong','step_probs':[.5,.5]},'sequence')==pytest.approx(np.log(.25))
    with pytest.raises(ValueError):risk_at_coverage([],[],.8)


def test_json_and_paths(tmp_path):
    p=tmp_path/'x.jsonl';p.write_text('{"key":"a","file":"f","method":"vcd","error":"old"}\n{"key":"a","file":"f","method":"vcd","outcome":"correct"}\n')
    assert len(load_records(p))==1
    p.write_text('{broken')
    with pytest.raises(ValueError):list(read_jsonl(p))
    with pytest.raises(ValueError):inside(tmp_path,tmp_path.parent/'outside')
    assert len(stable_key('a',2))==64


def test_cache_pipeline(tmp_path):
    from stage9_analyze import run
    cache_fixture(tmp_path)
    rows,curve=run(tmp_path,tmp_path/'outputs/tables/stage9',('q9b','llava16'),32)
    assert len(rows)==32 and len(curve)==160
    assert (tmp_path/'outputs/tables/stage9/paired_reliability.csv').exists()


def test_prepare_run_finalize(tmp_path):
    from stage9_prepare import prepare
    from stage9_run import run,probe_sequence,target_paths,image_condition
    from stage9_finalize import finalize
    from PIL import Image
    image_fixture(tmp_path);selected=prepare(tmp_path,2,1)
    assert len(selected)==32
    for model in ('q9b','llava16'):
        result=run(tmp_path,model,FakeEngine(),'Name the object.',fake_score,use_legacy=False)
        assert result['new_generations']==16*3*3
    curve,effects,geometry=finalize(tmp_path,32)
    assert len(curve)==36 and len(geometry)==32
    assert all(r['category']=='at_least_one_name_reachable' for r in geometry)
    em=FakeEngine();p=probe_sequence(em,{'reference':False},{'reference':True},[1,2])
    assert p['validation_passed'] and p['feasible_default']
    assert probe_sequence(em,{}, {},list(range(13)))['status']=='exceeds_frozen_generation_budget'
    with pytest.raises(ValueError):image_condition(Image.new('RGB',(10,10)),'short32')
    with pytest.raises(ValueError):image_condition(Image.new('RGB',(100,100)),'unregistered')


def test_resume_and_reuse(tmp_path):
    from stage9_prepare import prepare
    from stage9_run import run
    image_fixture(tmp_path);prepare(tmp_path,2,1)
    em=FakeEngine();result=run(tmp_path,'q9b',em,'Name the object.',fake_score,use_legacy=False)
    again=run(tmp_path,'q9b',em,'Name the object.',fake_score,use_legacy=False)
    assert again['new_generations']==0
    source=tmp_path/'outputs/raw/stage9/q9b_evidence_0.jsonl'
    original=[r for r in read_jsonl(source) if r['condition']=='original']
    legacy=tmp_path/'outputs/raw/q9b_s6_eval_naming.jsonl'
    legacy.write_text(''.join(json.dumps(r)+'\n' for r in original))
    source.unlink();(tmp_path/'outputs/raw/stage9/q9b_geometry_0.jsonl').unlink()
    reused=run(tmp_path,'q9b',em,'Name the object.',fake_score,use_legacy=True)
    assert reused['reused_generations']==16*3


def test_pope_byte_download_and_pipeline(tmp_path,monkeypatch):
    import stage9_pope as p
    from PIL import Image
    image_bytes=io.BytesIO();Image.new('RGB',(96,96)).save(image_bytes,format='JPEG')
    annotations={};payload={}
    for subset in ('random','adversarial'):
        rows=[]
        for i in range(2):
            for q in range(6):rows.append(dict(image=f'COCO_val2014_{i:012}.jpg',question_id=i*6+q,text='Is there an object?',label='yes' if q<3 else 'no'))
        b=('\n'.join(json.dumps(r) for r in rows)+'\n').encode();url=f'fixture://{subset}'
        annotations[subset]=(url,p.blob_sha(b));payload[url]=b
    def fetch(url):return payload[url] if url in payload else image_bytes.getvalue()
    selected=p.prepare(tmp_path,2,fetch,annotations);assert len(selected)==24
    for model in ('q9b','llava16'):p.run(tmp_path,model,FakeEngine())
    assert len(p.summarize(tmp_path,32))==4
    assert p.parse_answer(' Yes, indeed.')=='yes' and p.parse_answer('unknown') is None
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'example'
    monkeypatch.setattr(p.urllib.request,'urlopen',lambda *a,**k:Response())
    assert p.fetch_bytes('fixture://data')==b'example'
