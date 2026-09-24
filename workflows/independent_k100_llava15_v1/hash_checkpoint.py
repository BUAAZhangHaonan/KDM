from pathlib import Path
import json,hashlib,time
R=Path(__file__).resolve().parents[2];B=R/'outputs/records/independent_k100_llava15_v1';specpath=R/'configs/runtime/llava15_7b.json';spec=json.loads(specpath.read_text());path=R/'cache/models/llava-1.5-7b-hf'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
checks=[]
for w in spec['weights']:
 f=path/w['filename'];st=f.stat();digest=sha(f);assert digest==(w.get('hub_recorded_sha256') or w['sha256']);assert st.st_size==w['size_bytes'];checks.append({'filename':w['filename'],'sha256':digest,'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev})
assert sha(path/'config.json')==spec['model_config_sha256']
for n,h in spec['processor']['files'].items():assert sha(path/n)==h
proof={'spec_sha256':sha(specpath),'checkpoint_path':str(path),'checks':checks,'passed':True,'finished_unix':time.time()}
(B/'checkpoint_full_sha256.json').write_text(json.dumps(proof,indent=2));print('WEIGHTS_VERIFIED',sum(x['size_bytes'] for x in checks),flush=True)
