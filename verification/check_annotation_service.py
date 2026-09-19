"""Synthetic semantic-judge interface check; never formal annotations."""
import json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import requests
from kdm.scoring import semantic_judge_request
from annotate_responses import parse_label
from serve_annotation_judge import canonical_sha
from kdm.io import file_hash
spec_path=ROOT/'configs/runtime/semantic_judge.json';spec=json.loads(spec_path.read_text())
client=requests.Session();client.trust_env=False
response=client.get(spec['endpoint']+'/identity',timeout=10,allow_redirects=False);response.raise_for_status();identity=response.json()
assert canonical_sha(identity['receipt'])==identity['receipt_sha256']
assert identity['receipt']['spec_sha256']==file_hash(spec_path)
examples=[('Is there a cup in the image?','No','answer_assertive'),
          ('What drink is shown?','Maybe it is milk.','answer_uncertain'),
          ('What specific food is shown?','Food','answer_assertive'),
          ('What specific food is shown?','I cannot determine the food in this image.','abstain'),
          ('What specific food is shown?','','invalid')]
report={'purpose':'synthetic_software_interface_only','judge_identity':identity,'cases':[]}
out=ROOT/'outputs/verification'/('judge_semantic_examples_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
for question,answer,expected in examples:
    request={'model':spec['alias'],'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'},
        'messages':[{'role':'system','content':'Classify answer behavior. Output only JSON.'},
                    {'role':'user','content':json.dumps(semantic_judge_request(question,answer),ensure_ascii=False)}]}
    response=client.post(spec['endpoint']+'/chat/completions',json=request,timeout=180,allow_redirects=False)
    case={'question':question,'answer':answer,'expected':expected,'http_status':response.status_code,'response':response.json()}
    try:
        response.raise_for_status();body=response.json()
        assert body['kdm_judge_receipt_sha256']==identity['receipt_sha256']
        assert body['model']==spec['alias']
        assert body['choices'][0]['finish_reason']=='stop'
        label=parse_label(body['choices'][0]['message']['content'],answer)
        case.update(parsed=label,passed=label['label']==expected)
    except Exception as exc:case.update(passed=False,error=f'{type(exc).__name__}: {exc}')
    report['cases'].append(case);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':len(report['cases']),'passed':case['passed'],'record':str(out)}),flush=True)
report['passed']=all(x['passed'] for x in report['cases']);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if not report['passed']:raise SystemExit(1)
