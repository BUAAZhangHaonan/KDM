"""Exact native Gemma image preprocessing over a private, persistent CPU pipe.

No model weights are loaded. Only Gemma3ImageProcessor.preprocess is delegated.
The native package profile, checkpoint processor files, both source modules and
native backend code are bound by the handshake identity. Failed workers are not
restarted; callers must preserve the error and stop.
"""
import atexit,enum,hashlib,json,os,pickle,struct,subprocess,sys,threading,traceback
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
NATIVE=ROOT/'.environments/mprisk-tf553/bin/python'
CHECKPOINT=Path('/home/team/lvshuyang/Models/gemma-3-4b-it')
MAX_FRAME=1024*1024*1024

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pack(x):
 import numpy as np
 import torch
 from PIL import Image
 if isinstance(x,Image.Image):return {'__kdm__':'pil','mode':x.mode,'size':x.size,'data':x.tobytes()}
 if torch.is_tensor(x):
  if x.device.type!='cpu':raise ValueError('Image preprocessing bridge accepts only CPU input tensors')
  a=x.detach().contiguous()
  return {'__kdm__':'torch','dtype':str(a.dtype).split('.')[-1],'shape':tuple(a.shape),'data':a.view(torch.uint8).numpy().tobytes()}
 if isinstance(x,np.ndarray):
  if x.dtype.hasobject:raise TypeError('Object arrays not supported')
  a=np.ascontiguousarray(x);return {'__kdm__':'numpy','dtype':a.dtype.str,'shape':a.shape,'data':a.tobytes()}
 if isinstance(x,enum.Enum):return pack(x.value)
 if isinstance(x,Mapping):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,tuple):return {'__kdm__':'tuple','items':[pack(v) for v in x]}
 if isinstance(x,list):return [pack(v) for v in x]
 if isinstance(x,np.generic):return pack(x.item())
 if x is None or isinstance(x,(str,int,float,bool,bytes)):return x
 raise TypeError('Unsupported CPU bridge input: '+str(type(x)))

def unpack(x):
 import numpy as np
 import torch
 from PIL import Image
 if isinstance(x,list):return [unpack(v) for v in x]
 if isinstance(x,dict):
  tag=x.get('__kdm__')
  if tag=='pil':return Image.frombytes(x['mode'],x['size'],x['data'])
  if tag=='torch':return torch.frombuffer(bytearray(x['data']),dtype=getattr(torch,x['dtype'])).reshape(x['shape'])
  if tag=='numpy':return np.frombuffer(x['data'],dtype=np.dtype(x['dtype'])).copy().reshape(x['shape'])
  if tag=='tuple':return tuple(unpack(v) for v in x['items'])
  return {k:unpack(v) for k,v in x.items()}
 return x

def send(stream,obj):
 data=pickle.dumps(obj,protocol=4)
 if len(data)>MAX_FRAME:raise ValueError('Bridge frame exceeds1GiB')
 stream.write(struct.pack('!Q',len(data)));stream.write(data);stream.flush()

def receive(stream):
 def exact(n):
  out=bytearray()
  while len(out)<n:
   chunk=stream.read(n-len(out))
   if not chunk:raise EOFError('Native preprocessing pipe closed')
   out.extend(chunk)
  return bytes(out)
 size=struct.unpack('!Q',exact(8))[0]
 if size>MAX_FRAME:raise ValueError('Bridge frame exceeds1GiB')
 return pickle.loads(exact(size))

def native_main():
 sink=sys.stdout.buffer;sys.stdout=sys.stderr
 try:
  from importlib.metadata import version
  from transformers import AutoProcessor
  import inspect
  spec=json.loads((ROOT/'configs/runtime/gemma3_4b.json').read_text())
  actual={p:version(p) for p in spec['versions']}
  if actual!=spec['versions']:raise ValueError('Native processor environment differs from frozen Gemma profile')
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Native image helper must have no visible GPU')
  files=spec['processor']['files']
  if any(sha(CHECKPOINT/name)!=digest for name,digest in files.items()):raise ValueError('Checkpoint processor file changed')
  processor=AutoProcessor.from_pretrained(str(CHECKPOINT),trust_remote_code=True,local_files_only=True).image_processor
  sources={inspect.getfile(type(processor)),inspect.getfile(type(processor).__mro__[1])}
  identity={'schema':'kdm_gemma_native_pixels_bridge_v1','bridge_sha256':sha(__file__),'native_python':str(NATIVE),
   'versions':actual,'checkpoint':str(CHECKPOINT),'processor_files':files,'native_processor_class':str(type(processor)),
   'native_code_sha256':{p:sha(p) for p in sorted(sources)},'processor_config':processor.to_dict(),'device':'cpu','model_weights_loaded':False}
  send(sink,{'ok':True,'identity':identity})
  while True:
   try:message=receive(sys.stdin.buffer)
   except EOFError:return
   if message.get('close'):return
   request=unpack(message)
   if request['processor_config']!=processor.to_dict():raise ValueError('Caller processor configuration differs from exact native processor')
   result=processor.preprocess(*request['args'],**request['kwargs'])
   send(sink,{'ok':True,'output':pack(dict(result))})
 except BaseException as exc:
  send(sink,{'ok':False,'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()})
  raise

class NativePixelsBridge:
 def __init__(self):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1')
  self.process=subprocess.Popen([str(NATIVE),str(Path(__file__).resolve()),'--native-worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,env=env,cwd=ROOT)
  self.lock=threading.Lock();self.closed=False;self.calls=0
  response=receive(self.process.stdout)
  if not response.get('ok'):raise RuntimeError(response)
  self.identity=response['identity']
  if self.identity['bridge_sha256']!=sha(__file__):raise RuntimeError('Bridge source identity changed')
  atexit.register(self.close)
 def preprocess(self,processor,args,kwargs):
  from transformers.feature_extraction_utils import BatchFeature
  with self.lock:
   if self.closed or self.process.poll() is not None:raise RuntimeError('Native image helper exited; no automatic retry')
   send(self.process.stdin,pack({'processor_config':processor.to_dict(),'args':args,'kwargs':kwargs}))
   response=receive(self.process.stdout)
   if not response.get('ok'):raise RuntimeError(response)
   self.calls+=1
   return BatchFeature(data=unpack(response['output']))
 def close(self):
  if self.closed:return
  self.closed=True
  try:
   if self.process.poll() is None:send(self.process.stdin,{'close':True})
  except (BrokenPipeError,OSError):pass
  self.process.stdin.close()
  try:self.process.wait(timeout=10)
  except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
  self.process.stdout.close()

_bridge=None

def install():
 """Call before constructing external or vLLM processors; patches only local Gemma class."""
 global _bridge
 if _bridge is not None:return _bridge
 from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor
 bridge=NativePixelsBridge()
 def native_preprocess(self,*args,**kwargs):return bridge.preprocess(self,args,kwargs)
 Gemma3ImageProcessor.preprocess=native_preprocess
 _bridge=bridge
 return bridge

if __name__=='__main__':
 if sys.argv[1:]!=['--native-worker']:raise SystemExit('Only private native-worker entry is supported')
 native_main()
