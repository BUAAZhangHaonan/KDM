import json
from pathlib import Path
import pytest
from kdm.io import stable_hash
from kdm.human_review import build_human_review_queue,merge_human_revisions,validate_human_review


def fixture(tmp_path):
    sample={'id':'s','question':'What is shown?'}
    common={'model':'hidden-model','sample':sample,'guided':True,'marker':'UNKNOWN','kind':'main'}
    rows=[{**common,'key':'a','method':'direct','text':'UNKNOWN'},
          {**common,'key':'b','method':'vcd','text':'Maybe milk'},
          {**common,'key':'c','method':'deco','text':''},
          {**common,'key':'d','method':'direct','marker':'UNCLEAR','text':'milk'}]
    labels=['abstain','answer_uncertain','invalid','answer_assertive']
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    annotations=tmp_path/'auto.jsonl';annotations.write_text(''.join(json.dumps({'key':r['key'],'text':r['text'],
        'label':label,'evidence':r['text'],'answer_text':'milk' if 'answer_' in label else ''})+'\n' for r,label in zip(rows,labels)))
    queue=tmp_path/'queue.jsonl';build_human_review_queue([records],annotations,queue)
    return records,annotations,queue,rows


def approve(row):
    return {'key':row['key'],'text_sha256':stable_hash(row['text']),'action':'approve','revision':1,'previous_revision':0,
            'reviewer':'synthetic-external-reviewer','reviewed_at':'2026-09-19T01:00:00Z'}


def save(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_queue_is_full_prioritized_and_blinded(tmp_path):
    records,ann,queue,rows=fixture(tmp_path)
    q={row['key']:row for row in map(json.loads,queue.read_text().splitlines())}
    assert set(q)=={'a','b','c','d'}
    assert all('model' not in row and 'method' not in row for row in q.values())
    assert 'hidden-model' not in queue.read_text()
    assert 'abstention_change' in q['a']['review_reasons'] and 'abstention_change' in q['b']['review_reasons']
    assert 'invalid_output' in q['c']['review_reasons']
    assert q['d']['review_reasons']==['all_remaining_outputs']
    assert all(row['review_status']=='pending' for row in q.values())


def test_external_review_requires_every_output(tmp_path):
    records,ann,queue,rows=fixture(tmp_path);decisions=tmp_path/'external.jsonl'
    save(decisions,[approve(r) for r in rows[:-1]])
    output=tmp_path/'partial.jsonl';receipt=merge_human_revisions(queue,[decisions],output)
    assert not receipt['all_reviewed'] and receipt['missing_or_unresolved_keys']==['d']
    with pytest.raises(ValueError,match='incomplete'):validate_human_review(output,[records])
    with pytest.raises(ValueError,match='receipt'):validate_human_review(ann,[records])


def test_revision_history_merge_and_validation(tmp_path):
    records,ann,queue,rows=fixture(tmp_path);decisions=tmp_path/'external.jsonl'
    first=[approve(r) for r in rows]
    revision={**first[1],'revision':2,'previous_revision':1,'action':'revise','reviewer':'synthetic-adjudicator',
        'reviewed_at':'2026-09-19T02:00:00+00:00','reason':'explicit uncertainty was confirmed',
        'annotation':{'label':'answer_uncertain','evidence':'Maybe','answer_text':'milk'}}
    save(decisions,first+[revision]);output=tmp_path/'reviewed.jsonl'
    assert merge_human_revisions(queue,[decisions],output)['all_reviewed']
    checked=validate_human_review(output,[records]);assert checked['b']['evidence']=='Maybe'
    r={r['key']:r for r in map(json.loads,output.read_text().splitlines())}['b']
    assert r['initial_annotation']['evidence']=='Maybe milk' and len(r['review_history'])==2
    assert r['reviewer']=='synthetic-adjudicator'
    decisions.write_text(decisions.read_text()+'\n')
    with pytest.raises(ValueError,match='decision file changed'):validate_human_review(output,[records])


@pytest.mark.parametrize('change',[{'key':'unknown'},{'text_sha256':'changed'},{'reviewer':''},
    {'revision':2},{'reviewed_at':'2026-09-19T01:00:00'},
    {'action':'revise','reason':'x','annotation':{'label':'abstain','evidence':'UNKNOWN','answer_text':'UNKNOWN'}}])
def test_invalid_external_decisions_are_rejected(tmp_path,change):
    records,ann,queue,rows=fixture(tmp_path);decision={**approve(rows[0]),**change}
    path=tmp_path/'external.jsonl';save(path,[decision])
    with pytest.raises(ValueError):merge_human_revisions(queue,[path],tmp_path/'out.jsonl')


def test_changed_raw_response_does_not_reuse_review(tmp_path):
    records,ann,queue,rows=fixture(tmp_path);decisions=tmp_path/'external.jsonl'
    save(decisions,[approve(r) for r in rows]);out=tmp_path/'out.jsonl'
    merge_human_revisions(queue,[decisions],out)
    rows[0]['method']='modified';save(records,rows)
    with pytest.raises(ValueError,match='changed original'):validate_human_review(out,[records])


def test_unresolved_submission_remains_incomplete(tmp_path):
    records,ann,queue,rows=fixture(tmp_path);decisions=tmp_path/'external.jsonl'
    submitted=[approve(r) for r in rows];submitted[0]['action']='unresolved';save(decisions,submitted)
    out=tmp_path/'out.jsonl';receipt=merge_human_revisions(queue,[decisions],out)
    assert receipt['missing_or_unresolved_keys']==['a']
    with pytest.raises(ValueError,match='incomplete'):validate_human_review(out,[records])


def test_failed_judge_output_requires_explicit_human_revision(tmp_path):
    records,annotations,_,rows=fixture(tmp_path)
    auto=list(map(json.loads,annotations.read_text().splitlines()));save(annotations,auto[1:])
    annotations.with_suffix('.identity.json').write_text(json.dumps({'identity':'fixed','definition':{'queue_sha':'original-queue'}}))
    failure={'key':'a','text':'UNKNOWN','question':'What is shown?','annotation_identity':'fixed','queue_sha256':'original-queue',
        'error':'ValueError: nonempty evidence required','response_text':'ORIGINAL MALFORMED JUDGE JSON'}
    errors=tmp_path/'errors.jsonl';save(errors,[failure]);queue=tmp_path/'failure_queue.jsonl'
    build_human_review_queue([records],annotations,queue,[errors])
    queued={r['key']:r for r in map(json.loads,queue.read_text().splitlines())}
    assert queued['a']['initial_annotation'] is None and queued['a']['judge_failures']==[failure]
    assert 'judge_parse_failure' in queued['a']['review_reasons']
    decisions=tmp_path/'external.jsonl';save(decisions,[approve(r) for r in rows])
    with pytest.raises(ValueError,match='complete human revise'):merge_human_revisions(queue,[decisions],tmp_path/'invalid.jsonl')
    submitted=[approve(r) for r in rows];submitted[0]['action']='unresolved';save(decisions,submitted)
    unresolved=tmp_path/'unresolved.jsonl';merge_human_revisions(queue,[decisions],unresolved)
    with pytest.raises(ValueError,match='incomplete'):validate_human_review(unresolved,[records])
    submitted[0].update(action='revise',reason='Synthetic test only: quote the explicit abstention marker',
        annotation={'label':'abstain','evidence':'UNKNOWN','answer_text':''})
    save(decisions,submitted);output=tmp_path/'revised.jsonl';merge_human_revisions(queue,[decisions],output)
    assert validate_human_review(output,[records])['a']['label']=='abstain'
    failure['annotation_identity']='wrong';save(errors,[failure])
    with pytest.raises(ValueError,match='identity differs'):build_human_review_queue([records],annotations,tmp_path/'badqueue.jsonl',[errors])


def test_all_failed_annotation_ledger_can_enter_review(tmp_path):
    records,annotations,_,rows=fixture(tmp_path);annotations.write_text('')
    annotations.with_suffix('.identity.json').write_text(json.dumps({'identity':'fixed','definition':{'queue_sha':'q'}}))
    errors=tmp_path/'all_errors.jsonl'
    failures=[{'key':r['key'],'text':r['text'],'question':r['sample']['question'],
        'annotation_identity':'fixed','queue_sha256':'q','error':'synthetic transport failure','response_text':None} for r in rows]
    save(errors,failures);queue=tmp_path/'all_failed_queue.jsonl'
    assert build_human_review_queue([records],annotations,queue,[errors])['required_count']==4
    assert all(r['initial_annotation'] is None for r in map(json.loads,queue.read_text().splitlines()))
    failures[0]['key']='unknown';save(errors,failures)
    with pytest.raises(ValueError,match='Unknown annotation failure key'):
        build_human_review_queue([records],annotations,tmp_path/'unknown_queue.jsonl',[errors])
