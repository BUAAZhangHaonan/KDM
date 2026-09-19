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
