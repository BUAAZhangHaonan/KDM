"""CPU-only reproduction of post-processor threadpool inheritance; no GPU calls."""
import json,os,pathlib,sys,multiprocessing as mp,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
def child(send):
 import torch
 send.send({'stage':'before_cpu_zeros','pid':os.getpid()})
 x=torch.zeros((128,32768),dtype=torch.int32,device='cpu')
 send.send({'stage':'after_cpu_zeros','sum':int(x.sum().item())})
def main():
 sys.path.insert(0,str(ROOT/'workflows/independent_v5'));sys.path.insert(0,str(ROOT/'workflows/independent_k100_v1'))
 import independent,admission
 admission.execution.read_registry=admission.expanded_registry
 independent.closed.PATHS.update(admission.MODEL_PATHS);independent.resolve_image_path=admission.image_resolver
 sc=independent.Generator('llava16_mistral')
 samples=[json.loads(x) for x in (ROOT/'data/current/interface16.jsonl').read_text().splitlines()]
 for s in samples:sc.prepare(s)
 report={'gpu_calls':False,'setup':'same actual independent Generator.prepare on frozen16, then CPU zeros(128,32768) in child','parent_threads':len(list(pathlib.Path('/proc/self/task').iterdir())),'cases':{}}
 for method in ['fork','spawn']:
  ctx=mp.get_context(method);receive,send=ctx.Pipe(duplex=False);p=ctx.Process(target=child,args=(send,));start=time.monotonic();p.start();send.close();p.join(15);events=[]
  while receive.poll():
   try:events.append(receive.recv())
   except EOFError:break
  alive=p.is_alive();case={'pid':p.pid,'elapsed_s':time.monotonic()-start,'events':events,'alive_after_15s':alive}
  if alive:
   case['wchan']=pathlib.Path(f'/proc/{p.pid}/wchan').read_text();p.terminate();p.join(5);case['our_test_child_terminated']=True
  case['exit_code']=p.exitcode;report['cases'][method]=case
 out=ROOT/'outputs/records/independent_v5/cpu_fork_diagnosis_v1.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
