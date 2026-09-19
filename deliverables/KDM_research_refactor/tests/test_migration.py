import ast,subprocess
from pathlib import Path
import pytest
from kdm.migration import migrate,extract_backbone

SOURCE='''
import time
import torch
from engine import HP
NORM_CHAINS=['norm']
class FamilyModel:
    def __init__(self,model_path,device,attn_implementation=None):
        self.model=Factory.from_pretrained(model_path,device_map=device)
    def _patch_conv3d(self): pass
    def build(self,image,text): return {}
    def old_experiment(self): pass
class InternVLModel:
    def __init__(self,model_path,device): pass
    def build(self,image,text): return {}
    def _prefill(self,inputs,ohs=False): pass
    def _step(self,tok,cache,ohs=False): pass
    def old_experiment(self): pass
def get_engine(model_path,device,attn_implementation=None):
    return FamilyModel(model_path,device,attn_implementation=attn_implementation)
'''

def repo(tmp_path):
    p=tmp_path/'repo';p.mkdir();(p/'code').mkdir();(p/'code/stage3_engine.py').write_text(SOURCE)
    (p/'README.md').write_text('old');(p/'data').mkdir();(p/'data/immutable').write_text('keep')
    subprocess.run(['git','init','-q'],cwd=p,check=True)
    subprocess.run(['git','add','.'],cwd=p,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.test','commit','-qm','fixture'],cwd=p,check=True)
    return p

def test_extraction():
    out=extract_backbone(SOURCE);assert 'old_experiment' not in out and 'from engine' not in out
    assert 'max_memory' in out;compile(out,'x','exec')

def test_migration_readonly_and_apply(tmp_path):
    p=repo(tmp_path);bundle=Path(__file__).parents[1]
    r=migrate(p,bundle,allow_source_change=True);assert (p/'code').is_dir()
    migrate(p,bundle,apply=True,allow_source_change=True)
    assert not (p/'code').exists() and (p/r['archive']/'code/stage3_engine.py').exists()
    assert (p/'data/immutable').read_text()=='keep'
    assert (p/'src/kdm/models/backbone.py').is_file()
    assert (p/'docs/current/PAPER_STORY.md').is_file()

def test_migration_collision_no_moves(tmp_path):
    p=repo(tmp_path);(p/'docs/current').mkdir(parents=True);(p/'docs/current/file').write_text('keep')
    subprocess.run(['git','add','.'],cwd=p,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.test','commit','-qm','collision'],cwd=p,check=True)
    with pytest.raises(FileExistsError):migrate(p,Path(__file__).parents[1],True,True)
    assert (p/'code/stage3_engine.py').exists()
