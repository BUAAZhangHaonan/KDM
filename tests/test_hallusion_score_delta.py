"""CPU fixtures exercise incremental scoring; they do not create real judgments."""
import json
from pathlib import Path
import pytest
from kdm.io import stable_hash
from workflows.hallusion_blind import score_delta as scoring

class HallFixture:
    def source_record(self,sample):assert sample["dataset"]=="hallusionbench"
    def quality_key(self,sample,answer):return stable_hash([sample["prompt"],answer,sample["gt_answer_details"]])
    def quality_request(self,sample,answer):
        return {"quality_key":self.quality_key(sample,answer),"dataset":"hallusionbench",
                "question":sample["prompt"],"answer":answer,"gt_answer_details":sample["gt_answer_details"],
                "question_kind":"binary","reference_kind":"ordinary","memberships":[]}
    def literal_binary_quality(self,sample,answer):
        if answer.strip().lower() not in {"yes","no"}:return None
        score=int(answer.strip().lower()==sample["gold"])
        return {"quality_key":self.quality_key(sample,answer),"quality_label":"correct" if score else "incorrect",
                "score":score,"official_correctness":score,"source":"fixture_literal"}
    def check_behavior_prediction(self,sample,prediction):assert prediction in ("yes","no",None)
    def validate_quality(self,sample,answer,review):
        request=self.quality_request(sample,answer)
        for f in ("quality_key","question","answer","gt_answer_details"):assert review[f]==request[f]
        score=int(review["quality_label"]=="correct")
        return {"quality_key":request["quality_key"],"quality_label":review["quality_label"],
                "score":score,"official_correctness":score,"source":"fixture_review"}

def inputs(tmp_path,answer="The visible shape is a canine."):
    sample={"id":"hallusion_fixture","dataset":"hallusionbench","prompt":"Is a dog shown?",
            "gt_answer_details":"Yes, a dog is shown.","gold":"yes","source_id":"fixture"}
    conditions=[];rows=[]
    for method in ("direct","vcd"):
        c={"model":"fixture","method":method,"kind":"native_unguided","marker":"NONE",
           "reference_marker":"NONE","guided":False,"reference_guided":False,"replicate":0,
           "hall_expected_n":951,"checkpoint":"fixture"}
        conditions.append(c)
        rows.append({**{f:c[f] for f in scoring.FIELDS},"condition_identity":stable_hash(c),
            "key":method,"sample":sample,"text":answer,"terminated":True,"status":"ok",
            "tokens":[1,0],"wall_s":.1,"identity":"fixture","dataset_identity":"fixture","claim_identity":"fixture"})
    path=tmp_path/"raw.jsonl";path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path,conditions,sample

def test_pending_never_defaults_wrong_and_keeps_method_denominators(tmp_path):
    path,conditions,sample=inputs(tmp_path)
    scores,metrics,ready,held,sources=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()))
    assert len(scores)==2 and all(r["score"] is None and r["abstain"] is None for r in scores)
    assert len(ready)==1 and len(ready[0]["memberships"])==2
    assert ready[0]["need_behavior"] and ready[0]["need_quality"]
    assert ready[0]["gt_answer_details"]==sample["gt_answer_details"]
    assert len(metrics)==2 and all(m["generated"]==1 and m["missing_generation"]==950 for m in metrics)
    assert all(m["accuracy"] is None and m["abstention_rate"] is None for m in metrics)

def test_rules_resolve_whole_yes_but_partial_panel_not_final(tmp_path):
    path,conditions,_=inputs(tmp_path,"Yes")
    scores,metrics,ready,_,_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()))
    assert all(r["score"]==1 and r["abstain"] is False for r in scores)
    assert not ready
    assert all(m["observed_accuracy"]==1 and m["accuracy"] is None for m in metrics)

def test_claimed_keys_are_excluded_from_dispatch_queue(tmp_path):
    path,conditions,_=inputs(tmp_path)
    result=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()))
    key=result[2][0]["joint_key"]
    _,_,ready,held,_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},{key}))
    assert not ready and len(held)==1

def review_fixture(request):
    behavior={"label":"answer_assertive","abstain":False,"answer_text":request["answer"],
              "predicted_answer":"yes","evidence_span":request["answer"]}
    metadata={"author":"/root/cpu_fixture","model":"gpt-5.6-luna","effort":"medium","call_id":None}
    return {**{f:request[f] for f in ("joint_key","qa_key","quality_key","question","answer","gt_answer_details")},
        "behavior_review":{**{f:request[f] for f in ("qa_key","question","answer")},"decision":behavior,**metadata},
        "quality_review":{**{f:request[f] for f in ("quality_key","question","answer","gt_answer_details")},
            "quality_label":"correct","answer_evidence":request["answer"],"reference_evidence":request["gt_answer_details"],
            "reason":"CPU structural fixture only; not a real annotation",**metadata}}

def test_actual_completion_required_and_entire_joint_batch_bound(tmp_path):
    path,conditions,_=inputs(tmp_path)
    request=scoring.score_rows([path],conditions,HallFixture(),{}, {}, ({},{},set()))[2][0]
    batch=tmp_path/"reviews"/"batch_fixture";batch.mkdir(parents=True)
    (batch/"pending_review.jsonl").write_text(json.dumps(request)+"\n")
    reviewed=review_fixture(request)
    (batch/"decisions.jsonl").write_text(json.dumps(reviewed)+"\n")
    b,q,claimed=scoring.read_review_batches(batch.parent)
    assert not b and not q and request["joint_key"] in claimed
    (batch/"write_receipt.json").write_text("{}")
    reviews=scoring.read_review_batches(batch.parent)
    scores,_,ready,held,_=scoring.score_rows([path],conditions,HallFixture(),{}, {}, reviews)
    assert all(r["score"]==1 and r["abstain"] is False for r in scores)
    assert not ready and not held
    reviewed["behavior_review"]["effort"]="high"
    (batch/"decisions.jsonl").write_text(json.dumps(reviewed)+"\n")
    with pytest.raises(ValueError,match="gpt-5.6-luna medium"):
        scoring.read_review_batches(batch.parent)

def test_foreign_condition_and_duplicate_predictions_rejected(tmp_path):
    path,conditions,_=inputs(tmp_path,"Yes")
    with pytest.raises(ValueError,match="outside frozen"):
        scoring.score_rows([path],conditions[:1],HallFixture(),{}, {}, ({},{},set()))
    with pytest.raises(ValueError,match="Duplicate generated"):
        scoring.score_rows([path,path],conditions,HallFixture(),{}, {}, ({},{},set()))
