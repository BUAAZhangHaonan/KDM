import json,zipfile,subprocess,importlib.util
from pathlib import Path
import pytest
from PIL import Image
from kdm.io import Ledger,read_jsonl,within,atomic_json
from kdm.data import food_from_existing,vizwiz_manifest,extract_safe
from kdm.scoring import lexical_label,food_correct,vqa_score
from kdm.annotation import build_queue,validate_annotations


def test_ledger_resume_and_identity(tmp_path):
    p=tmp_path/'out.jsonl';l=Ledger(p,{'v':1});l.add('one',{'value':2})
    assert Ledger(p,{'v':1}).keys=={'one'}
    with pytest.raises(ValueError):Ledger(p,{'v':2})
    with pytest.raises(ValueError):l.add('one',{})


def test_duplicate_and_bad_json(tmp_path):
    p=tmp_path/'bad';p.write_text('{}\n\n')
    with pytest.raises(ValueError):list(read_jsonl(p))
    with pytest.raises(ValueError):atomic_json(tmp_path/'n',{'v':float('nan')})


def test_path_and_archive_safety(tmp_path):
    with pytest.raises(ValueError):within(tmp_path,'../outside')
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escaped','x')
    with pytest.raises(ValueError):extract_safe(archive,tmp_path/'dest')


def test_complete_manifests(tmp_path):
    img=tmp_path/'x.png';Image.new('RGB',(8,8)).save(img)
    raw=tmp_path/'source.jsonl';raw.write_text('\n'.join(json.dumps({'image_path':str(img),'file':str(i),'class':'soup','part':part}) for i,part in enumerate(['group','eval']))+'\n')
    out=tmp_path/'new.jsonl';food_from_existing(raw,out)
    rows=list(read_jsonl(out));assert len(rows)==2;assert {r['split'] for r in rows}=={'dev','eval'}
    annotation=tmp_path/'val.json';annotation.write_text(json.dumps([{'image':'x.png','question':'q','answers':[{'answer':'a'}]*10,'answerable':0}]))
    vizwiz_manifest(annotation,tmp_path,tmp_path/'viz.jsonl');v=list(read_jsonl(tmp_path/'viz.jsonl'))
    assert len(v)==1 and v[0]['annotated_answerable']==0

@pytest.mark.parametrize('text,expected',[('UNKNOWN','abstain'),('I do not know.','abstain'),('', 'invalid'),('No',None),('Probably milk',None),('It is unknown whether this is milk',None)])
def test_exact_and_semantic_labels(text,expected):assert lexical_label(text)==expected


def test_scoring_named_categories():
    aliases={'cup_cakes':['cupcake','cupcakes'],'crab_cakes':['crab cake','crab cakes']}
    assert food_correct('Cupcake','cup_cakes',aliases)
    assert not food_correct('person','crab_cakes',aliases)
    assert vqa_score('yes',['yes']*3+['no']*7)==pytest.approx(.9)
    assert vqa_score('yes',['yes']*4+['no']*6)==1


def test_full_annotation(tmp_path):
    p=tmp_path/'rows';p.write_text(json.dumps({'key':'a','text':'Perhaps milk','sample':{'question':'q'}})+'\n')
    q=tmp_path/'queue';assert build_queue([p],q)==1
    with pytest.raises(ValueError):validate_annotations(q)
    q.write_text(json.dumps({'key':'a','text':'Perhaps milk','label':'answer_uncertain','evidence':'Perhaps','answer_text':'milk'})+'\n')
    assert validate_annotations(q)['a']['label']=='answer_uncertain'


def test_judge_parser():
    path=Path(__file__).parents[1]/'scripts/annotate_responses.py'
    spec=importlib.util.spec_from_file_location('judge_script',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.parse_label('{"label":"answer_uncertain","evidence_span":"Maybe","answer_text":"milk"}','Maybe milk')['answer_text']=='milk'
    with pytest.raises(ValueError):m.parse_label('{"label":"abstain","evidence_span":"invented"}','UNKNOWN')
