"""Versioned Gemma native-pixel verification/production entry; no implicit review."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import gemma_native_pixels_v1 as pixels
import independent as base
import reviewed_entry
import verify_engine
from kdm.io import file_hash,stable_hash,atomic_json,within
CPU_PROOF=ROOT/'outputs/records/independent_v5/gemma_native_pixels_cpu_v1/audit.json'

def binding_for(bridge):
 proof=json.loads(CPU_PROOF.read_text())
 assert proof['passed'] is True and proof['count']==16 and len(proof['rows'])==16
 assert proof['native_bridge_identity']==bridge.identity
 assert proof['script_sha256']==file_hash(Path(__file__).with_name('gemma_native_pixels_cpu_audit_v1.py'))
 for r in proof['rows']:
  assert all(r['external_native_tensor_parity'].values()) and r['actual_engine_native_pixel_parity'] and r['expanded_tokens_equal']
 return {'schema':'kdm_gemma_native_pixels_execution_binding_v1','entry_sha256':file_hash(__file__),
  'bridge_identity':bridge.identity,'cpu_proof':str(CPU_PROOF.relative_to(ROOT)),'cpu_proof_sha256':file_hash(CPU_PROOF),
  'cpu_audit_source_sha256':proof['script_sha256'],'engine_verifier_sha256':file_hash(verify_engine.__file__),
  'review_entry_sha256':file_hash(reviewed_entry.__file__),'independent_source_sha256':file_hash(base.__file__),
  'closed_preparer_sha256':file_hash(base.closed.__file__)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','generate']);args,rest=parser.parse_known_args()
 assert '--model' in rest and rest[rest.index('--model')+1]=='gemma3_4b','Gemma-only entry'
 bridge=pixels.install()
 try:
  binding=binding_for(bridge)
  original_admission=base.worker_admission
  def resource_admission():return {**original_admission(),'native_pixels_bridge':binding,'native_pixels_bridge_binding_sha256':stable_hash(binding)}
  base.worker_admission=resource_admission
  if args.mode=='verify':
   p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--native-reference',required=True);p.add_argument('--out',required=True);a=p.parse_args(rest)
   original_json=verify_engine.atomic_json
   def bound_json(path,data):
    if Path(path).name in {'check.json','failure.json'}:data={**data,'native_pixels_bridge':binding,'native_pixels_bridge_binding_sha256':stable_hash(binding)}
    return original_json(path,data)
   verify_engine.atomic_json=bound_json
   verify_engine.run(a)
  else:
   def validate(model,native_path,check_path,sc):
    check_path=within(ROOT,check_path);check=json.loads(check_path.read_text());review=json.loads(check_path.with_name('review.json').read_text())
    assert check['native_pixels_bridge']==binding and check['native_pixels_bridge_binding_sha256']==stable_hash(binding)
    assert review['native_pixels_bridge_binding_sha256']==stable_hash(binding)
    assert review['native_pixels_entry_sha256']==file_hash(__file__)
    native,proof=reviewed_entry.validate(model,native_path,check_path,sc)
    return native,{**proof,'native_pixels_bridge':binding,'native_pixels_bridge_binding_sha256':stable_hash(binding)}
   base.validate_independent_proof=validate
   sys.argv=[sys.argv[0]]+rest;base.main()
 finally:bridge.close()

if __name__=='__main__':main()
