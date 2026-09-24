"""Use original free-screening logic with Vicuna's dated full-coverage receipt."""
import sys,math,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.independent_v5 import postprocess as base
from kdm.io import read_jsonl,stable_hash,file_hash,within

def checked_closed(record,model,samples,matcher):
 base.require(model=='llava16_vicuna','This entry is for completed Vicuna cohort only')
 done=base.load(record/'complete.json');progress=base.load(record/'progress.json')
 proof_path=record/'full_coverage_validation_v1.json';proof=base.load(proof_path)
 raw=within(ROOT,progress['output']);side_path=raw.with_suffix('.identity.json');side=base.load(side_path);definition=side['definition'];digest=file_hash(raw)
 base.require(done['complete'] is True and done['closed']==4848 and progress['status']=='complete','Incomplete candidate cohort')
 base.require(definition['model']==model and definition['schema']=='kdm_vllm_tree_closed_v4' and definition['manifest_sha256']==file_hash(ROOT/'data/current/all.jsonl'),'Candidate identity mismatch')
 base.require(done['identity']==definition and side['identity']==stable_hash(definition),'Candidate identity hash mismatch')
 base.require(done['output_sha256']==digest==proof['raw_sha256'] and proof['unique_rows']==4848 and proof['classes']==101 and proof['finite_scores_and_token_means_and_ranks'] is True and proof['identity_valid'] is True and proof['sidecar_sha256']==file_hash(side_path),'Candidate source receipt mismatch')
 ranks={}
 for row in read_jsonl(raw):
  sid=row['sample']['id'];base.require(sid not in ranks and row['sample']==samples.get(sid) and row['model']==model and row['identity']==side['identity'] and row['status']=='ok','Candidate row mismatch')
  base.require(row['key']==stable_hash([model,sid,'closed']),'Candidate key mismatch')
  scored=matcher.classify(row)
  for s in row['candidate_scores']:
   n=s['n_tokens'];base.require(type(n) is int and n>0 and math.isclose(s['sum_logp']/n,s['mean_logp'],rel_tol=1e-8,abs_tol=1e-8),'Candidate mean mismatch')
  ranks[sid]=scored['gold_rank']
 base.require(set(ranks)==set(samples) and file_hash(raw)==digest,'Candidate unique coverage/hash mismatch')
 return ranks,dict(record=str(record.relative_to(ROOT)),raw=str(raw.relative_to(ROOT)),raw_sha256=digest,sidecar_sha256=file_hash(side_path),receipt_sha256=file_hash(proof_path),receipt_file=str(proof_path.relative_to(ROOT)),identity=side['identity'],versioned_validation_entry_sha256=file_hash(Path(__file__)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--closed-record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();base.checked_closed=checked_closed;base.run(a.record,a.closed_record,a.out)
