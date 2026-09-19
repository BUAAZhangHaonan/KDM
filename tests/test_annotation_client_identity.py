import importlib.util,json
from pathlib import Path
import pytest
from kdm.io import file_hash,stable_hash


def module():
    spec=importlib.util.spec_from_file_location('annotation_client',Path(__file__).parents[1]/'scripts/annotate_responses.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def fixed_identity(tmp_path):
    (tmp_path/'configs/kdm').mkdir(parents=True);(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/kdm/models.json').write_text(json.dumps([{'hf_model_id':'subject/model'}]))
    (tmp_path/'scripts/serve_annotation_judge.py').write_text('# fixture service')
    spec={'alias':'judge','endpoint':'http://127.0.0.1:18765/v1','hf_model_id':'separate/judge',
        'versions':{'transformers':'fixture'},'device_map':{'':'cuda:0'},'gpu_count':1,
        'independence':{'independent_checkpoint':True,'subject_hf_model_ids':['subject/model']}}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    receipt={'spec_sha256':file_hash(path),'spec':spec,'model':'judge','versions':spec['versions'],
        'device_map':spec['device_map'],'physical_gpus':['5'],'parameter_devices':['cuda:0'],
        'service_id':'synthetic-service','started_utc':'2026-09-19T01:00:00Z',
        'script_sha256':file_hash(tmp_path/'scripts/serve_annotation_judge.py')}
    return path,{'receipt':receipt,'receipt_sha256':stable_hash(receipt)}


class Response:
    status_code=200
    def __init__(self,payload):self.payload=payload;self.text=json.dumps(payload)
    def raise_for_status(self):pass
    def json(self):return self.payload


def test_fixed_receipt_required_before_annotation(tmp_path):
    mod=module();path,identity=fixed_identity(tmp_path)
    class Session:
        def get(self,url,**kwargs):
            assert url=='http://127.0.0.1:18765/v1/identity' and kwargs['allow_redirects'] is False
            return Response(identity)
    assert mod.validate_judge_identity(Session(),'http://127.0.0.1:18765/v1',path,'judge',tmp_path)==identity
    identity['receipt']['model']='changed'
    with pytest.raises(ValueError,match='digest'):mod.validate_judge_identity(Session(),'http://127.0.0.1:18765/v1',path,'judge',tmp_path)


@pytest.mark.parametrize('mutation',['changed_receipt','length'])
def test_changed_service_or_truncated_json_never_becomes_label(tmp_path,monkeypatch,mutation):
    mod=module();path,identity=fixed_identity(tmp_path)
    queue=tmp_path/'queue.jsonl';queue.write_text(json.dumps({'key':'x','text':'Maybe milk','question':'q','label':None})+'\n')
    content=json.dumps({'label':'answer_uncertain','evidence_span':'Maybe','answer_text':'milk'})
    payload={'model':'judge','id':'synthetic-reply','kdm_judge_receipt_sha256':identity['receipt_sha256'],
        'choices':[{'finish_reason':'stop','message':{'content':content}}]}
    if mutation=='length':payload['choices'][0]['finish_reason']='length'
    else:payload['kdm_judge_receipt_sha256']='different-service'
    class Session:
        def get(self,*args,**kwargs):return Response(identity)
        def post(self,*args,**kwargs):return Response(payload)
    monkeypatch.setattr(mod.requests,'Session',Session);out=tmp_path/'labels.jsonl'
    monkeypatch.setattr('sys.argv',['annotate','--root',str(tmp_path),'--queue',str(queue),'--endpoint','http://127.0.0.1:18765/v1',
        '--judge-model','judge','--judge-spec',str(path),'--out',str(out)])
    with pytest.raises(SystemExit):mod.main()
    assert out.read_text()==''
    failure=json.loads(Path(str(out)+'.errors.jsonl').read_text())
    assert failure['key']=='x' and json.loads(failure['response_text'])==payload



def test_only_single_complete_json_fence_is_a_valid_container():
    mod=module();body=json.dumps({'label':'abstain','evidence_span':'Food','answer_text':''})
    # Preserve the model's actual semantic error; this test only verifies the envelope.
    parsed=mod.parse_label('```json\n'+body+'\n```','Food')
    assert parsed['label']=='abstain' and parsed['parse_format']=='single_complete_json_fence'
    assert mod.parse_label(body,'Food')['parse_format']=='raw_json'
    for content in ('comment\n```json\n'+body+'\n```','```json\n'+body+'\n```\ntrailing',
                    '```json\n'+body+'\n```\n```json\n'+body+'\n```',body+' '+body):
        with pytest.raises(ValueError):mod.parse_label(content,'Food')


def test_parse_failure_continues_and_resume_needs_explicit_retry(tmp_path,monkeypatch):
    mod=module();path,identity=fixed_identity(tmp_path);queue=tmp_path/'queue.jsonl'
    queue.write_text(''.join(json.dumps({'key':key,'text':'Food','question':'q','label':None})+'\n' for key in ('a','b')))
    bad={'label':'abstain','evidence_span':'','answer_text':''}
    good={'label':'answer_assertive','evidence_span':'Food','answer_text':'Food'}
    calls=[]
    class Session:
        def get(self,*args,**kwargs):return Response(identity)
        def post(self,*args,**kwargs):
            calls.append(kwargs['json'])
            content=json.dumps(bad if len(calls)==1 else good)
            return Response({'model':'judge','id':'fake','kdm_judge_receipt_sha256':identity['receipt_sha256'],
                'choices':[{'finish_reason':'stop','message':{'content':content}}]})
    monkeypatch.setattr(mod.requests,'Session',Session);out=tmp_path/'out.jsonl'
    argv=['annotate','--root',str(tmp_path),'--queue',str(queue),'--endpoint','http://127.0.0.1:18765/v1',
          '--judge-model','judge','--judge-spec',str(path),'--out',str(out)]
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(SystemExit) as exc:mod.main()
    assert exc.value.code==1 and len(calls)==2
    assert [json.loads(line)['key'] for line in out.read_text().splitlines()]==['b']
    failure=json.loads(Path(str(out)+'.errors.jsonl').read_text())
    assert json.loads(json.loads(failure['response_text'])['choices'][0]['message']['content'])==bad
    with pytest.raises(SystemExit):mod.main()
    assert len(calls)==2
    monkeypatch.setattr('sys.argv',argv+['--retry-failed']);mod.main()
    assert len(calls)==3 and {json.loads(line)['key'] for line in out.read_text().splitlines()}=={'a','b'}
