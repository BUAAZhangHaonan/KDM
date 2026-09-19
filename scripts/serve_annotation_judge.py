#!/usr/bin/env python3
"""Fixed local semantic judge. No formal labels are manufactured by this service."""
import argparse,datetime,hashlib,importlib.metadata,json,os,sys,uuid
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kdm.io import file_hash,within,atomic_json
from discover_models import checkpoint_files


def validate_request(obj,alias):
    if obj.get('model')!=alias or obj.get('temperature')!=0 or obj.get('max_tokens')!=512:
        raise ValueError('Judge identity and decoding settings must match its fixed specification')
    if obj.get('response_format')!={'type':'json_object'}:
        raise ValueError('Only semantic JSON requests are accepted')
    if set(obj)-{'model','messages','temperature','max_tokens','response_format'}:
        raise ValueError('Unknown request options')
    messages=obj.get('messages')
    if not isinstance(messages,list) or len(messages)!=2 or [m.get('role') for m in messages]!=['system','user']:
        raise ValueError('Expected one system message and one blinded question/answer request')
    if messages[0].get('content')!='Classify answer behavior. Output only JSON.':
        raise ValueError('Unexpected judge system message')
    if any(not isinstance(m.get('content'),str) for m in messages):raise ValueError('Text-only judge')
    request=json.loads(messages[1]['content'])
    if not isinstance(request,dict) or 'model' in request or 'method' in request or 'key' in request:
        raise ValueError('Judge request is not blinded')
    return messages


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def runtime_spec(spec):
    if Path(sys.executable).absolute()!=Path(spec['environment_python']).absolute():raise ValueError('Wrong judge Python')
    for name,want in spec['versions'].items():
        if importlib.metadata.version(name)!=want:raise ValueError(f'Judge version mismatch: {name}')
    cards=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(cards)!=1 or cards[0] not in {'0','1','4','5'}:raise ValueError('Judge needs one authorized physical GPU')
    if Path('/proc/self/fd/20').resolve()!=ROOT/'outputs/locks'/f'gpu_{cards[0]}.lock':raise ValueError('Use scripts/worker.sh')
    checkpoint=Path(spec['model_path'])
    actual={p.name:p.stat().st_size for p in checkpoint_files(checkpoint)}
    if actual!={x['filename']:x['size_bytes'] for x in spec['weights']}:raise ValueError('Judge weights changed or incomplete')
    for name,sha in spec['processor_files'].items():
        if file_hash(checkpoint/name)!=sha:raise ValueError(f'Judge processor changed: {name}')
    if file_hash(checkpoint/'config.json')!=spec['config_sha256']:raise ValueError('Judge model config changed')
    candidates=json.loads((ROOT/'configs/kdm/models.json').read_text())
    if spec['hf_model_id'] in {x['hf_model_id'] for x in candidates}:raise ValueError('Judge must be an independent checkpoint')
    return cards


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();spec_path=within(ROOT,args.spec);spec=json.loads(spec_path.read_text())
    cards=runtime_spec(spec);run=within(ROOT,args.run_dir)
    if run.exists():raise ValueError('Judge run directory must be fresh')
    run.mkdir(parents=True)
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText,AutoConfig
    config=AutoConfig.from_pretrained(spec['model_path'],local_files_only=True)
    if spec['quantization']!='native_fp8_explicit_dequantize_bfloat16':raise ValueError('Unexpected judge quantization policy')
    config.quantization_config['dequantize']=True
    processor=AutoProcessor.from_pretrained(spec['model_path'],local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(spec['model_path'],config=config,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    devices=sorted({str(p.device) for p in model.parameters()})
    if devices!=['cuda:0']:raise ValueError(f'Judge CPU/disk/off-device weights are prohibited: {devices}')
    receipt={'service_id':str(uuid.uuid4()),'model':spec['alias'],'spec_sha256':file_hash(spec_path),
        'spec':spec,'physical_gpus':cards,'parameter_devices':devices,'device_map':{'':'cuda:0'},
        'versions':{name:importlib.metadata.version(name) for name in spec['versions']},
        'script_sha256':file_hash(__file__),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'json_mode':'prompt_only_strict_client_validation_no_constrained_decoder','formal_annotations_completed':0}
    receipt_sha=canonical_sha(receipt);identity={'receipt':receipt,'receipt_sha256':receipt_sha}
    atomic_json(run/'identity.json',identity)
    request_log=(run/'requests.jsonl').open('a',encoding='utf-8')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*values):pass
        def send(self,status,payload):
            body=json.dumps(payload,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/v1/identity':self.send(200,identity)
            else:self.send(404,{'error':'unknown route'})
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.send(404,{'error':'unknown route'});return
            request_id='kdm-judge-'+str(uuid.uuid4());record={'id':request_id,'receipt_sha256':receipt_sha}
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>1048576:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size));messages=validate_request(body,spec['alias'])
                record['request']=body
                inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt').to('cuda:0')
                if inputs['input_ids'].shape[1]>spec['max_input_tokens']:raise ValueError('Judge input exceeds registered token limit; no truncation')
                with torch.inference_mode():
                    result=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True)
                tokens=result[0,inputs['input_ids'].shape[1]:].tolist()
                text=processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
                terminated=bool(tokens and tokens[-1] in eos)
                record.update(output_tokens=tokens,input_tokens=inputs['input_ids'][0].tolist(),text=text,terminated=terminated,status='ok')
                request_log.write(json.dumps(record,ensure_ascii=False)+'\n');request_log.flush()
                self.send(200,{'id':request_id,'model':spec['alias'],'kdm_judge_receipt_sha256':receipt_sha,
                    'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop' if terminated else 'length'}],
                    'usage':{'prompt_tokens':inputs['input_ids'].shape[1],'completion_tokens':len(tokens)}})
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}')
                request_log.write(json.dumps(record,ensure_ascii=False)+'\n');request_log.flush()
                self.send(500,{'id':request_id,'error':record['error'],'kdm_judge_receipt_sha256':receipt_sha})
    print(json.dumps({'ready':True,'endpoint':spec['endpoint'],'receipt_sha256':receipt_sha}),flush=True)
    HTTPServer(('127.0.0.1',spec['port']),Handler).serve_forever()

if __name__=='__main__':main()
