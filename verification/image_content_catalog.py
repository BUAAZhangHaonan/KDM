"""One-pass original image identity catalog and read-only relocated input verification."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path,PurePosixPath

def digest(path):
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   block=f.read(1024*1024)
   if not block:break
   h.update(block)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
  raise RuntimeError(f'File metadata changed while hashing: {path}')
 return {'sha256':h.hexdigest(),'size_bytes':after.st_size}

def save_new(path,value):
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'));f.write('\n')

def load_manifest(path,expected):
 raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if h!=expected:raise ValueError(f'Manifest identity mismatch: {h} != {expected}')
 rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
 paths=[r['image_path'] for r in rows]
 if len(rows)!=9167 or len(set(paths))!=9167:raise ValueError('Expected all 9167 rows and distinct original image paths')
 return rows,paths,h

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','verify']);p.add_argument('--manifest',required=True);p.add_argument('--catalog',required=True);p.add_argument('--receipt',required=True);p.add_argument('--expected-manifest-sha256',required=True);p.add_argument('--source-root',required=True);p.add_argument('--target-root');a=p.parse_args()
 catalog=Path(a.catalog);receipt=Path(a.receipt)
 if receipt.exists() or (a.mode=='create' and catalog.exists()):raise FileExistsError('Refusing to overwrite catalog or receipt')
 started=time.monotonic();report={'schema':1,'mode':a.mode,'passed':False,'script_path':str(Path(__file__).resolve()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':sys.argv,'manifest_path':str(Path(a.manifest).resolve()),'source_root':a.source_root,'target_root':a.target_root,'hashed_images':0,'total_hashed_bytes':0,'failures':[],'scope':'One SHA256 pass over each original manifest image only; no model weights, filtering, GPU, or source image mutation'}
 try:
  rows,paths,mhash=load_manifest(Path(a.manifest),a.expected_manifest_sha256);report['manifest_sha256']=mhash;report['manifest_rows']=len(rows);report['unique_image_paths']=len(paths)
  source=PurePosixPath(a.source_root)
  for original in paths:
   rel=PurePosixPath(original).relative_to(source)
   if '..' in rel.parts or not PurePosixPath(original).is_absolute():raise ValueError('Unsafe source path in manifest')
  if a.mode=='create':
   images={}
   for i,original in enumerate(paths,1):
    images[original]=digest(Path(original));report['hashed_images']+=1;report['total_hashed_bytes']+=images[original]['size_bytes']
    if i%1000==0:print(json.dumps({'hashed_images':i,'total_bytes':report['total_hashed_bytes']}),flush=True)
   save_new(catalog,{'schema':1,'manifest_sha256':mhash,'images':images})
   report['catalog_path']=str(catalog.resolve());report['catalog_sha256']=hashlib.sha256(catalog.read_bytes()).hexdigest();report['catalog_bytes']=catalog.stat().st_size;report['passed']=True
  else:
   if not a.target_root:raise ValueError('verify requires --target-root')
   raw=catalog.read_bytes();data=json.loads(raw);report['catalog_path']=str(catalog.resolve());report['catalog_sha256']=hashlib.sha256(raw).hexdigest()
   if data.get('schema')!=1 or data.get('manifest_sha256')!=mhash or set(data.get('images',{}))!=set(paths):raise ValueError('Catalog does not cover this exact full manifest')
   target=Path(a.target_root).resolve(strict=True)
   for i,original in enumerate(paths,1):
    actual=target/str(PurePosixPath(original).relative_to(source))
    try:
     actual.resolve(strict=True).relative_to(target)
     got=digest(actual);report['hashed_images']+=1;report['total_hashed_bytes']+=got['size_bytes']
     if got!=data['images'][original]:report['failures'].append({'original':original,'target':str(actual),'expected':data['images'][original],'actual':got})
    except Exception as e:report['failures'].append({'original':original,'target':str(actual),'error':type(e).__name__+': '+str(e)})
    if i%1000==0:print(json.dumps({'checked_images':i,'hashed_images':report['hashed_images'],'failures':len(report['failures'])}),flush=True)
   report['passed']=not report['failures'] and report['hashed_images']==9167
 except Exception as e:
  report['error']=type(e).__name__+': '+str(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  report['elapsed_seconds']=time.monotonic()-started;save_new(receipt,report)
 print(json.dumps({k:report.get(k) for k in ['mode','passed','hashed_images','total_hashed_bytes','elapsed_seconds','error']}),flush=True)
 return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
