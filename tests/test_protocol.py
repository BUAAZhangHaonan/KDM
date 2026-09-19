import json
import pytest
from kdm.protocol import validate_census_collection, selected_samples
from kdm.pipeline import census_tasks, task_id


def test_selection_refuses_missing_unguided_and_missing_candidate(tmp_path):
    samples=[{"id":"s", "dataset":"food101", "split":"eval"}]
    path=tmp_path/"census.jsonl"
    rows=[dict(t, model="m", key=task_id("m",t), status="ok", tokens=[1], terminated=True)
          for t in census_tasks(samples)]
    def write(items):
        path.write_text("".join(json.dumps(r)+"\n" for r in items))
    write(rows)
    validate_census_collection([path],samples,["m"])
    with pytest.raises(ValueError, match="Missing fixed candidate"):
        validate_census_collection([path],samples,["m","other"])
    write(rows[:1])
    with pytest.raises(ValueError, match="Incomplete guided or unguided"):
        validate_census_collection([path],samples,["m"])
    write(rows+[rows[0]])
    with pytest.raises(ValueError, match="duplicate census task"):
        validate_census_collection([path],samples,["m"])


def test_selected_manifest_preserves_entire_dataset_and_split():
    samples=[{"id":str(i),"dataset":"food101" if i<2 else "vizwiz","split":"dev" if i==0 else "eval"} for i in range(3)]
    selection=[{"model":"m","dataset":"food101","n":2,"n_abstain":1,"selected":True},
               {"model":"m","dataset":"vizwiz","n":1,"n_abstain":0,"selected":False}]
    assert selected_samples(samples,selection,"m")==samples[:2]
    selection[0]["n"]=1
    with pytest.raises(ValueError, match="denominator"):
        selected_samples(samples,selection,"m")


def test_runtime_rejects_unresolved_and_unlocked_model(tmp_path):
    from kdm.protocol import validate_runtime
    with pytest.raises(ValueError, match="unresolved"):
        validate_runtime(tmp_path,{"key":"m","availability":"missing"},"m",["0"])
    spec={"key":"m","availability":"resolved","gpu_count":2}
    with pytest.raises(ValueError, match="GPU count"):
        validate_runtime(tmp_path,spec,"m",["0"])
    spec["gpu_count"]=1
    with pytest.raises(ValueError, match="worker"):
        validate_runtime(tmp_path,spec,"m",["0"])
