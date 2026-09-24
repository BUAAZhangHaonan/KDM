"""User-authorized separate vLLM cohort; preserves unsuccessful exact-parity evidence."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import independent as base
from kdm.io import file_hash,stable_hash,within,read_jsonl
from importlib.metadata import version

def validate(model,native_path,check_path,sc):
 native_path=within(ROOT,native_path);check_path=within(ROOT,check_path)
 native=json.loads(native_path.read_text());check=json.loads(check_path.read_text())
 review_path=check_path.with_name('review.json');review=json.loads(review_path.read_text())
 assert review['accepted_separate_cohort'] is True and review['engine_check_sha256']==file_hash(check_path)
 assert review['user_authorization']=='2026-09-23 use vLLM if differences are normal backend variation'
 assert review['input_sampling_checkpoint_configuration_reviewed'] is True
 assert review['exact_greedy_equivalence_claimed'] is False
 assert native['schema']=='kdm_native_independent_reference_v1' and native['model']==model
 assert check['schema']=='kdm_vllm_independent_reference_check_v1' and check['model']==model
 assert check['native_reference_sha256']==file_hash(native_path)
 assert check['implementation_sha256']==file_hash(Path(base.__file__))
 assert check['closed_preparer_sha256']==file_hash(Path(base.closed.__file__))
 assert review['review_entry_sha256']==file_hash(Path(__file__))
 versions={p:version(p) for p in ['vllm','torch','transformers']}
 assert versions==check['versions'] and native['manifest_sha256']==file_hash(ROOT/'data/current/all.jsonl')
 fixed={s['id'] for s in read_jsonl(ROOT/'data/current/interface16.jsonl')}
 assert len(native['rows'])==len(check['rows'])==16
 assert {r['sample']['id'] for r in native['rows']}=={r['sample_id'] for r in check['rows']}==fixed
 eos=native['eos_token_ids'];assert sorted(check['eos_token_ids'])==sorted(eos)
 base.parameters(model,'eos-check',0,eos)
 for r in native['rows']:
  _,prompt,ids,inp=sc.prepare(r['sample'])
  assert prompt==r['prompt'] and ids==r['unexpanded_prompt_ids'] and inp['input_ids'][0].tolist()==r['expanded_prompt_ids']
  assert all(k in inp and base.closed.summarize(inp[k])==base.closed.core_summary(v) for k,v in r['processor_summary'].items())
 by_id={r['sample']['id']:r for r in native['rows']};mismatches=0
 for r in check['rows']:
  n=by_id[r['sample_id']];tokens=r['engine_greedy_tokens'];p=r['engine_greedy_selected_log_probabilities']
  assert 1<=len(tokens)<=32 and len(p)==len(tokens) and all(math.isfinite(x) for x in p+n['greedy_selected_log_probabilities'])
  assert not any(t in eos for t in tokens[:-1]) and (tokens[-1] in eos or len(tokens)==32)
  same=tokens==n['greedy_tokens'];assert same==r['greedy_tokens_equal'];mismatches+=not same
  assert r['engine_expanded_prompt_ids_equal'] and r['engine_processor_tensor_parity'] and all(r['engine_processor_tensor_parity'].values())
  assert r['ten_distinct_stable_seeds'] and r['native_eos_handling'] and r['finite_selected_logprobs']
  assert math.isfinite(r['ten_request_wall_s']) and r['ten_request_wall_s']>0
  if model=='qwen35_4b' and review.get('zero_cache_reason')=='vllm024_qwen35_align_minimum528_exceeds_prompt':
   assert r['num_cached_tokens']>=0
  else:assert r['num_cached_tokens']>0
 assert review['greedy_mismatch_count']==mismatches
 spec=native['backend']
 for w in spec['weights']:assert (Path(sc.path)/w['filename']).stat().st_size==w['size_bytes']
 assert file_hash(Path(sc.path)/'config.json')==spec['model_config_sha256']
 for name,digest in spec['processor']['files'].items():assert file_hash(Path(sc.path)/name)==digest
 if model=='gemma3_4b':
  a=check['gemma_attention_audit'];assert a['metadata_ranges_checked'] and not a['kernel_execution_claimed'] and a['module_sha256']==file_hash(ROOT/'workflows/acceleration_v4/gemma_audit_worker.py')
 return native,{'native_reference_sha256':file_hash(native_path),'verification_sha256':file_hash(check_path),'review_sha256':file_hash(review_path),'review_entry_sha256':file_hash(Path(__file__)),'versions':versions,'eos_token_ids':eos,'source_spec_sha256':stable_hash(spec),'exact_greedy_equivalence':False,'greedy_mismatch_count':mismatches,'separate_engine_cohort':True}

def main():
 base.validate_independent_proof=validate
 base.main()

if __name__=='__main__':main()
