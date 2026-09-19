"""Verify extraction keeps official normalization state without dataset API calls."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('asset_fetcher',Path(__file__).parents[1]/'scripts/fetch_assets.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
SOURCE='''
import re
class VQAEval:
    def __init__(self,vqa,vqaRes,n=2):
        self.n=n
        self.params={'question_id':vqa.getQuesIds()}
        self.contractions={"dont":"don't"}
        self.manualMap={'two':'2'}
        self.articles=['a','an','the']
        self.periodStrip=re.compile(r"\\.")
        self.commaStrip=re.compile(r"(\\d),(\\d)")
        self.punct=['?']
    def evaluate(self):
        print "python two only"
    def processPunctuation(self,inText):
        return inText.replace('?', '')
    def processDigitArticle(self,inText):
        return ' '.join(self.manualMap.get(w,w) for w in inText.lower().split() if w not in self.articles)
'''

def test_construct_normalizer_without_vqa_objects():
    ns={};exec(mod.extract_vqa_normalizer(SOURCE),ns)
    obj=ns['VQAEval'](None,None)
    assert obj.processDigitArticle(obj.processPunctuation('The two?'))=='2'
    assert not hasattr(obj,'params')

def test_refuse_incomplete_reference_source():
    with pytest.raises(ValueError):mod.extract_vqa_normalizer(SOURCE.replace('self.punct=', 'self.unexpected='))


def test_fetch_records_failure_and_continues(tmp_path,monkeypatch):
    import json,sys
    manifest=tmp_path/'assets.json'
    manifest.write_text(json.dumps([{'group':'papers','url':'bad','local_path':'cache/bad'}, {'group':'papers','url':'good','local_path':'cache/good'}]))
    def fake_download(url,dest,expected=None):
        if url=='bad':raise RuntimeError('specific transport failure')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b'ok');return 'hash'
    monkeypatch.setattr(mod,'download',fake_download)
    monkeypatch.setattr(sys,'argv',['fetch_assets','--root',str(tmp_path),'--manifest',str(manifest),'--group','papers'])
    with pytest.raises(SystemExit):mod.main()
    reports=list((tmp_path/'outputs/records').glob('assets_papers_*.json'))
    rows=json.loads(reports[0].read_text())
    assert rows[0]['status']=='error' and rows[0]['error']=='specific transport failure'
    assert rows[1]['status']=='ok' and rows[1]['downloaded_sha256']=='hash'
