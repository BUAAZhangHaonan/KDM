"""Run every visualization and validate vector/raster output, not aesthetics."""
from pathlib import Path
import importlib.util,sys,json
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from stage9_geometry import target_interval


def test_numpy_boundary_types_serialize():
    interval=target_interval([2,1,-2],[2,-1,0],1)
    assert json.loads(json.dumps(interval.to_dict(),allow_nan=False))['lower']==pytest.approx(.5)


def test_every_visualization(tmp_path):
    from PIL import Image
    import fitz
    spec=importlib.util.spec_from_file_location('stage9_figures',ROOT/'code/viz/stage9_figures.py')
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.setup()
    d=ROOT/'data_sample'
    calls=[(v.error_exchange,[d/'stage6_core_selected.csv']),
           (v.confidence_tradeoff,[d/'stage6_core_selected.csv']),
           (v.offset_reversal,[d/'correction_selected.csv']),
           (v.reachable_intervals,[]),
           (v.risk_curves,[d/'synthetic_risk_coverage.csv']),
           (v.evidence_curve,[d/'synthetic_evidence_accuracy.csv']),
           (v.observed_reachability,[d/'synthetic_name_reachability.csv'])]
    for i,(fn,args) in enumerate(calls):
        out=tmp_path/f'figure_{i}'
        fn(*args,out,**({'fixture':True} if i>=4 else {}))
        with Image.open(out.with_suffix('.png')) as im:
            im.verify()
        pdf=fitz.open(out.with_suffix('.pdf'))
        assert len(pdf)==1 and pdf[0].rect.width>300
        text=pdf[0].get_text()
        assert ('SIMULATED FIXTURE' if i>=4 else 'MATHEMATICAL EXAMPLE' if i==3 else 'REAL DATA') in text
        assert not any(font[2]=='Type3' for font in pdf[0].get_fonts())
        pdf[0].get_pixmap(matrix=fitz.Matrix(.5,.5))
