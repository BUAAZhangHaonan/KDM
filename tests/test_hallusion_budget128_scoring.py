"""CPU semantic/budget regressions; fixture reviews are never production labels."""
import json
from dataclasses import asdict
import pytest
from kdm.decoding import DecodeConfig
from kdm.io import stable_hash
from workflows.hallusion_blind import score_delta as scoring
from workflows.hallusion_blind.score128 import eligible_conditions,review_indices
from test_hallusion_score_delta import HallFixture,inputs,review_fixture

def budget_inputs(tmp_path,answer='Yes',terminated=False):
    path,conditions,sample=inputs(tmp_path,answer)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    for c,r in zip(conditions,records):
        c['config']=asdict(DecodeConfig(method=c['method'],max_tokens=128))
        r.update(condition_identity=stable_hash(c),config=c['config'],terminated=terminated,
            truncated=not terminated,finish_reason='eos' if terminated else 'length',
            generation_source='new_budget128',tokens=[1,0] if terminated else [1]*128)
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return path,conditions,sample

def test_length_answer_can_be_correct_without_becoming_abstention(tmp_path):
    path,conditions,_=budget_inputs(tmp_path)
    scores,metrics,ready,_,_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()),token_budget=128)
    assert not ready and all(s['score']==1 and s['abstain'] is False for s in scores)
    assert all(s['terminated'] is False and s['truncated'] is True and s['finish_reason']=='length' for s in scores)
    assert all(m['truncated']==1 and m['natural_eos']==0 and m['observed_accuracy']==1 for m in metrics)
    assert all(m['accuracy'] is None and m['expected_n']==951 for m in metrics)
    with pytest.raises(ValueError,match='outside final Hall EOS scoring'):
        scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()))

def test_length_wrong_answer_and_missing_judgment_are_distinct(tmp_path):
    path,conditions,_=budget_inputs(tmp_path,'No')
    scores,*_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()),token_budget=128)
    assert all(s['score']==0 and s['abstain'] is False for s in scores)
    path,conditions,_=budget_inputs(tmp_path,'The visible shape resembles')
    scores,metrics,ready,_,_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()),token_budget=128)
    assert all(s['score'] is None and s['abstain'] is None for s in scores)
    assert len(ready)==1 and all(m['pending_quality']==1 for m in metrics)

def test_unclear_keeps_official_code_two_and_zero_accuracy(tmp_path):
    class HallUnclearFixture(HallFixture):
        def validate_quality(self,sample,answer,review):
            q=super().validate_quality(sample,answer,review)
            if q['quality_label']=='unclear':q['official_correctness']=2
            return q
    path,conditions,_=budget_inputs(tmp_path,'The visible shape resembles')
    hall=HallUnclearFixture()
    request=scoring.score_rows([path],conditions,hall,{}, {}, ({},{},set()),token_budget=128)[2][0]
    review=review_fixture(request);review['quality_review']['quality_label']='unclear'
    batch=tmp_path/'review'/'batch_cpu_fixture';batch.mkdir(parents=True)
    (batch/'pending_review.jsonl').write_text(json.dumps(request)+'\n')
    (batch/'decisions.jsonl').write_text(json.dumps(review)+'\n')
    (batch/'write_receipt.json').write_text('{}')
    scores,metrics,ready,_,_=scoring.score_rows([path],conditions,hall,{}, {},
        review_indices([batch.parent]),token_budget=128)
    assert not ready and all(s['quality']['official_correctness']==2 and s['score']==0 for s in scores)
    assert all(s['abstain'] is False for s in scores)
    assert all(m['W']==1 and m['A']==0 for m in metrics)

@pytest.mark.parametrize('change',[
    {'tokens':[1]*127}, {'truncated':False}, {'finish_reason':'eos'},
    {'generation_source':'new_eos_only'}, {'terminated':0}, {'tokens':[True]*128}])
def test_inconsistent_budget_rows_rejected(tmp_path,change):
    path,conditions,_=budget_inputs(tmp_path)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[0].update(change)
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError):
        scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()),token_budget=128)

def test_silent_nonbudget_parameter_change_rejected(tmp_path):
    path,conditions,_=budget_inputs(tmp_path)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[0]['config']['temperature']=.7
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='frozen Food operating point'):
        scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()),token_budget=128)

def test_original_sid_excludes_whole_panels_without_filtering_questions():
    conditions=[{'model':m,'method':method} for m in ('qwen25vl','qwen3vl','gemma3_4b') for method in ('direct','sid')]
    eligible,applicability=eligible_conditions(conditions)
    assert len(eligible)==4 and all(c['model']=='gemma3_4b' or c['method']=='direct' for c in eligible)
    assert all(a['registered_denominator']==951 for a in applicability)
    excluded=[a for a in applicability if a['status']=='not_applicable_full_panel']
    assert len(excluded)==2 and all(a['required_generation']==0 for a in excluded)
