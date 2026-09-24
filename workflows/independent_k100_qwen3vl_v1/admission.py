"""K100 independent-generation host admission. Never modifies frozen registry."""
from pathlib import Path
import copy,fcntl,hashlib,importlib.metadata as metadata,json,os,socket,subprocess,sys
ROOT=Path('/home/k100/projects/knowledge-deficit-mitigation')
HOSTNAME='k100-X785-H30'
GPU_UUID='GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea'
PYTHON=ROOT/'.environments/vllm024/bin/python'
VERSIONS={'vllm':'0.24.0','torch':'2.11.0','transformers':'5.12.1','tokenizers':'0.22.2','numpy':'2.3.5','Pillow':'11.3.0','safetensors':'0.8.0','torchvision':'0.26.0','accelerate':'1.14.0'}
MODEL_PATHS={'qwen3vl':'/home/k100/Models/Qwen3-VL-8B-Instruct'}
sys.path.insert(0,str(ROOT/'src'))
import kdm.execution as execution
_original_registry=execution.read_registry
_original_resolver=execution.resolve_image_path
_verified_images={}
def expanded_registry(root):
 data=copy.deepcopy(_original_registry(root))
 data['hosts']['RTX_Pro_6000']={'hostname':HOSTNAME,'root':str(ROOT),'allowed_gpus':[0],'gpu_uuids':{'0':GPU_UUID}}
 return data
def image_resolver(logical,root=None):
 root=Path(root or ROOT).resolve();assert root==ROOT,'Unexpected K100 image root'
 target=_original_resolver(logical,root)
 catalog_path=root/'outputs/records/image_content_catalog.json';st=catalog_path.stat()
 catalog=execution._catalog(str(catalog_path),(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns))
 assert catalog['manifest_sha256']==execution.unchanged_file_hash(root/'data/current/all.jsonl'),'Manifest identity mismatch'
 expected=catalog['images'][str(logical)];stat=target.stat();sig=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
 key=(str(target),expected['sha256'],expected['size_bytes'])
 if _verified_images.get(key)!=sig:
  assert stat.st_size==expected['size_bytes'] and sha(target)==expected['sha256'],'Image bytes changed'
  after=target.stat();assert sig==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Image modified during validation'
  _verified_images[key]=sig
 return target
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkpoint(model):
 spec_path=ROOT/'configs/runtime'/f'{model}.json';spec=json.loads(spec_path.read_text());path=Path(MODEL_PATHS[model])
 assert path.is_dir() and path.parent==Path('/home/k100/Models'),'Unapproved model root'
 assert sha(path/'config.json')==spec['model_config_sha256'],'Changed model configuration'
 for name,digest in spec['processor']['files'].items():assert sha(path/name)==digest,'Changed processor '+name
 for weight in spec['weights']:
  f=path/weight['filename'];assert f.is_file() and f.stat().st_size==weight['size_bytes'],'Incomplete weight '+str(f)
  proof_path=ROOT/'outputs/records/independent_k100_qwen3vl_v1/checkpoint_full_sha256.json'
  proof=json.loads(proof_path.read_text());assert proof['spec_sha256']==sha(spec_path) and proof['checkpoint_path']==str(path)
  hit=next(x for x in proof['checks'] if x['filename']==weight['filename'])
  assert hit['sha256']==weight['hub_recorded_sha256'],'Full weight SHA256 mismatch'
  st=f.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(hit['size_bytes'],hit['mtime_ns'],hit['inode'],hit['device']),'Weight modified since full hashing'
 return {'path':str(path),'spec_sha256':sha(spec_path),'config_sha256':spec['model_config_sha256'],'processor_files':spec['processor']['files'],'weights':spec['weights'],'verification':'config/processor exact SHA256 and full weight SHA256 checked against registered Hub hashes; file stat unchanged since proof'}
def register(model=None):
 assert Path(__file__).resolve().parents[2]==ROOT and socket.gethostname()==HOSTNAME,'K100 root/host mismatch'
 assert Path(sys.executable).absolute()==PYTHON,'Wrong isolated runtime'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Only physical GPU0 is authorized'
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).strip()
 assert [v.strip() for v in gpu.split(',')]==['0',GPU_UUID],'GPU UUID changed'
 fd=os.fstat(20);expected=(ROOT/'outputs/locks/gpu_0.lock').stat()
 assert (fd.st_dev,fd.st_ino)==(expected.st_dev,expected.st_ino),'Missing expected worker lock on fd20'
 fcntl.flock(20,fcntl.LOCK_EX|fcntl.LOCK_NB)
 observed={p:metadata.version(p) for p in VERSIONS};assert observed==VERSIONS,('Runtime version mismatch',observed)
 selected=[model] if model is not None else list(MODEL_PATHS)
 assert set(selected)<=set(MODEL_PATHS),'Model not admitted on K100'
 checkpoints={key:checkpoint(key) for key in selected}
 execution.read_registry=expanded_registry
 execution.resolve_image_path=image_resolver
 return {'schema':'kdm_independent_k100_admission_v1','host':'RTX_Pro_6000','hostname':HOSTNAME,'root':str(ROOT),'physical_gpus':['0'],'gpu_uuids':{'0':GPU_UUID},'python':str(PYTHON),'versions':observed,'model_paths':dict(MODEL_PATHS),'checkpoints':checkpoints,'admission_sha256':sha(__file__),'frozen_registry_modified':False,'numerical_equivalence_claimed':False}
