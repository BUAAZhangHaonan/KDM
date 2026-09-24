import os,sys,json,time,copy,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'workflows/food_closed_v3')]
import admission
from kdm.io import atomic_json,file_hash,read_jsonl
from kdm.execution import validate_host,resolve_image_path
from kdm.protocol import validate_freeze
from kdm.pipeline import make_backend,closed_rank
from workflows.acceleration_v4.vllm_closed_v6 import summarize
from importlib.metadata import version
from PIL import Image
BASE=ROOT/'outputs/records/remaining11_onevision_v1'
def main():
 model='onevision';specpath=ROOT/f'configs/runtime/{model}.json';spec=json.loads(specpath.read_text());observed={k:version(k) for k in spec['versions']};assert observed==spec['versions']
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0';host,info=validate_host(ROOT,['0']);assert host=='6403'
 assert Path(os.readlink('/proc/self/fd/20'))==ROOT/'outputs/locks/gpu_0.lock'
 path=Path('/home/team/lvshuyang/Models/llava-onevision-qwen2-7b-ov-hf');checks={x['filename']:(path/x['filename']).stat().st_size==x['size_bytes'] for x in spec['weights']};checks['config']=file_hash(path/'config.json')==spec['model_config_sha256'];checks.update({n:file_hash(path/n)==h for n,h in spec['processor']['files'].items()});assert all(checks.values())
 freeze=validate_freeze(ROOT);runtime=copy.deepcopy(spec);runtime['kwargs']['model_path']=str(path);runtime['environment_python']=sys.executable
 samples=[s for s in read_jsonl(ROOT/'data/current/all.jsonl') if s['dataset']=='food101'];indices=[i*(len(samples)-1)//15 for i in range(16)];selected=[samples[i] for i in indices];names=sorted(json.loads((ROOT/'configs/kdm/food_aliases.json').read_text()))
 result={'model':model,'backend':spec,'backend_spec_sha256':file_hash(specpath),'source_blobs':freeze['source_blobs'],'manifest_sha256':file_hash(ROOT/'data/current/all.jsonl'),'workflow_sha256':file_hash(__file__),'scope':'real native16 candidate scores, evenly spaced original Food101 samples; relocated exact checkpoint/runtime/precision','execution':{'host':host,'physical_gpu':0,'gpu_uuid':info['gpu_uuids']['0'],'python':sys.executable,'versions':observed,'checkpoint_checks':checks,'dtype':'float16'},'indices':indices,'rows':[]}
 atomic_json(BASE/'native_progress.json',{'status':'loading','pid':os.getpid(),'completed':0})
 backend=make_backend(runtime,'cuda:0');tokens=[backend.encode(n.replace('_',' ')) for n in names]
 for sample in selected:
  t=time.perf_counter();image=Image.open(resolve_image_path(sample['image_path'],ROOT)).convert('RGB')
  score=closed_rank(backend,image,sample['question'],names,sample['class']);prompt=score['prompt']
  msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
  text=backend.em.proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  inputs=backend.em.build(image,prompt)
  row={'reference':{'status':'ok','model':model,'sample':sample,'wall_s':time.perf_counter()-t,**score},'unexpanded_prompt_ids':backend.encode(text),'expanded_prompt_ids':inputs['input_ids'][0].tolist(),'processor_summary':summarize(dict(inputs)),'candidates_tokens':tokens}
  result['rows'].append(row);atomic_json(BASE/'native_partial.json',result);atomic_json(BASE/'native_progress.json',{'status':'running','pid':os.getpid(),'completed':len(result['rows']),'last_sample_id':sample['id']});print('NATIVE_CLOSED',len(result['rows']),flush=True)
 assert len(result['rows'])==16
 atomic_json(BASE/'native_closed16_reference.json',result);atomic_json(BASE/'native_progress.json',{'status':'complete','pid':os.getpid(),'completed':16,'reference_sha256':file_hash(BASE/'native_closed16_reference.json')})
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic_json(BASE/'native_error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
