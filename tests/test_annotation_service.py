import copy,importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from serve_annotation_judge import validate_request,canonical_sha

def payload():
    return {'model':'judge','temperature':0,'max_tokens':512,'response_format':{'type':'json_object'},
        'messages':[{'role':'system','content':'Classify answer behavior. Output only JSON.'},
                    {'role':'user','content':'{"question":"Is there a cup?","answer":"No"}'}]}

def test_blinded_fixed_request():
    assert len(validate_request(payload(),'judge'))==2

@pytest.mark.parametrize('mutation',[{'model':'generator'},{'temperature':1},{'max_tokens':32},{'tools':[]}])
def test_refuses_changed_identity_or_protocol(mutation):
    obj=payload();obj.update(mutation)
    with pytest.raises(ValueError):validate_request(obj,'judge')

def test_refuses_generator_identity_in_request():
    obj=payload();obj['messages'][1]['content']='{"question":"q","answer":"a","model":"subject"}'
    with pytest.raises(ValueError,match='blinded'):validate_request(obj,'judge')

def test_receipt_hash_ignores_key_order():
    assert canonical_sha({'model':'x','versions':{'torch':'a'}})==canonical_sha({'versions':{'torch':'a'},'model':'x'})
