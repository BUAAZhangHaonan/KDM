#!/usr/bin/env python3
"""CPU-only source reuse audit; original raw, identities, and receipts stay unchanged."""
import json,pathlib,hashlib,collections,ast,gzip,math,datetime,sys,argparse,re
ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description="Verify bounded original completed Food matrix parts by scientific tuples, complete frozen decoding, prompts, and seeds.")
parser.add_argument("--source-host",required=True)
parser.add_argument("--claim",action="append",required=True,help="Registered model=original claim; one finite claim per flag.")
parser.add_argument("--out",required=True)
args=parser.parse_args()
root=ROOT;sys.path.insert(0,str(root/"src"))
claims=[x.split("=",1) for x in args.claim]
if any(len(x)!=2 or x[0] not in ("phi35","onevision","qwen3vl","internvl35_8b") or not re.fullmatch(r"[A-Za-z0-9_.-]+",x[1]) for x in claims):raise ValueError("Only explicit finite registered original claim names are allowed")
from kdm.io import stable_hash,stable_seed
from kdm.prompts import task_prompt,MARKERS
run=root/"outputs/supplemental/remaining11/run_20260930_140337"
out=(root/args.out).resolve()
out.relative_to(root/"outputs")
out.mkdir(parents=True,exist_ok=False)
defs=ast.parse((root/"src/kdm/decoding.py").read_text())
base={x.target.id:ast.literal_eval(x.value) for c in defs.body if isinstance(c,ast.ClassDef) and c.name=="DecodeConfig" for x in c.body if isinstance(x,ast.AnnAssign)}
samples={x["id"]:x for x in (json.loads(l) for l in (root/"data/current/all.jsonl").read_text().splitlines()) if x["dataset"]=="food101" and x["split"]=="eval"}
assert len(samples)==2424
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bk(b):
 return {k:b.get(k) for k in ["key","hf_model_id","dtype","thinking_mode","model_config_sha256","adapter_source_sha256"]}|{"weights":[{k:w.get(k) for k in ["filename","size_bytes","hub_revision","hub_recorded_sha256"]} for w in b.get("weights",[])],"processor_files":b.get("processor",{}).get("files",{})}
counts=collections.Counter();conditions=collections.Counter();holds=collections.Counter();seen={};sources=[];bindings=[];uid_changed=0
for model,claim in claims:
 spec=json.loads((root/f"configs/runtime/{model}.json").read_text())
 ps=sorted((run/"records"/model/"food101/formal"/claim).glob("*part_*.complete.json"))
 for rp in ps:
  rec=json.loads(rp.read_text());raw=root/rec["raw_path"];side=raw.with_suffix(".identity.json")
  sidej=json.loads(side.read_text());defn=sidej["definition"];be=defn["backend"]
  errors=[]
  if rec.get("generation_complete") is not True:errors.append("not_real_complete")
  if sha(raw)!=rec["raw_sha256"] or sha(side)!=rec["identity_sha256"]:errors.append("file_sha_differs_receipt")
  if sidej["identity"]!=stable_hash(defn):errors.append("sidecar_definition_hash")
  if bk(be)!=bk(spec):errors.append("scientific_checkpoint_processor_adapter_differs_registered")
  if defn["base_config"]!=base:errors.append("base_decode_differs_current_frozen_defaults")
  for rel,dig in defn.get("source_provenance",{}).get("registered_input_sha256",{}).items():
   if not (root/rel).is_file() or sha(root/rel)!=dig:errors.append("source_registered_input_changed:"+rel)
  for rel,dig in defn.get("source_provenance",{}).get("algorithm_git_blobs",{}).items():
   content=(root/rel).read_bytes();actual=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
   if actual!=dig:errors.append("source_frozen_algorithm_changed:"+rel)
  partmatched=[];partrows=0
  with raw.open() as f:
   for line,l in enumerate(f,1):
    row=json.loads(l);partrows+=1;counts[(model,"all_source_rows")]+=1
    sm=row["sample"];method=row["method"];kind=row["kind"]
    if method not in ("vcd","m3id") or kind not in ("main","reference_instruction_removed"):continue
    counts[(model,"matrix_candidate_rows")]+=1
    why=list(errors)
    if row["model"]!=model or row["identity"]!=sidej["identity"]:why.append("row_model_identity")
    expected=samples.get(sm.get("id"))
    if expected is None:why.append("not_registered_eval")
    elif {k:v for k,v in sm.items() if k!="image_path"}!={k:v for k,v in expected.items() if k!="image_path"} or pathlib.Path(sm["image_path"]).name!=pathlib.Path(expected["image_path"]).name:why.append("sample_scientific_fields_differs")
    validcondition=row["guided"] is True and row["marker"] in MARKERS and row["reference_marker"] in MARKERS and row["replicate"]==0
    if kind=="main":validcondition=validcondition and row["reference_guided"] is True
    else:validcondition=validcondition and row["reference_guided"] is False and row["marker"]==row["reference_marker"]
    if not validcondition:why.append("condition_outside_registered40")
    cfg={**base,"method":method}
    if method=="m3id":
     offset=row.get("offset_prompt_tokens")
     if not isinstance(offset,list) or not offset:why.append("missing_real_m3id_prompt_tokens")
     else:cfg["m3id_offset"]=len(offset)
    if row["config"]!=cfg:why.append("full_decode_config_differs")
    if row["seed"]!=stable_seed(sm["id"],model,row["replicate"]):why.append("sample_model_replicate_seed_differs")
    if row["prompt"]!=task_prompt(sm["question"],row["marker"],row["guided"]):why.append("main_prompt_differs")
    if row["reference_prompt"]!=task_prompt(sm["question"],row["reference_marker"],row["reference_guided"]):why.append("reference_prompt_differs")
    if row.get("status")!="ok":why.append("source_status_not_ok")
    if not row.get("terminated"):
     counts[(model,"fixed_budget_terminated_source_rows")]+=1
     if len(row.get("tokens",[]))!=base["max_tokens"]:why.append("unterminated_before_fixed_budget")
    tokens=row.get("tokens",[])
    if not 1<=len(tokens)<=32 or len(row.get("selected_log_probabilities",[]))!=len(tokens) or not all(math.isfinite(x) for x in row.get("selected_log_probabilities",[])):why.append("tokens_or_probability_invalid")
    task={k:row[k] for k in ["method","kind","marker","reference_marker","guided","reference_guided","replicate"]}
    registeredkey=stable_hash({"model":model,"task":task,"sample":sm["id"]})
    science={"model":model,"checkpoint":be["hf_model_id"],"dataset":"food101","split":"eval","sample_id":sm["id"],**task,"config":cfg,"seed":row["seed"],"prompt":row["prompt"],"reference_prompt":row["reference_prompt"]}
    tsha=stable_hash(science)
    if why:
     counts[(model,"unresolved_rows")]+=1
     for w in set(why):holds[(model,w)]+=1
     continue
    uid_changed+=int(row["key"]!=registeredkey)
    if tsha in seen:counts[(model,"duplicate_scientific_tuple_rows")]+=1
    else:
     counts[(model,"unique_reusable_scientific_tuples")]+=1;seen[tsha]=registeredkey
     partmatched.append(registeredkey)
     conditions[(model,method,kind,row["marker"],row["reference_marker"])]+=1
     bindings.append({"registered_key":registeredkey,"original_key":row["key"],"scientific_tuple_sha256":tsha,"condition":science,"source_host":args.source_host,"source_raw":str(raw),"source_line":line,"source_identity":str(side),"source_receipt":str(rp),"raw_sha256":rec["raw_sha256"],"identity_sha256":rec["identity_sha256"],"receipt_sha256":sha(rp)})
  assert partrows==rec["rows"]
  sources.append({"host":args.source_host,"root":str(root),"model":model,"dataset":"food101","stage":"formal","claim_id":claim,"part":rec["part"],"count":partrows,"raw_path":str(raw),"raw_sha256":rec["raw_sha256"],"identity_path":str(side),"identity_sha256":rec["identity_sha256"],"receipt_path":str(rp),"receipt_sha256":sha(rp),"completed_at_utc":rec["finished_utc"],"generation_complete":True,"matched_matrix_rows":len(partmatched),"scientific_identity_and_decode_errors":errors,"verified_actual_completed_part":True})
with gzip.open(out/"reusable_scientific_tuple_bindings.jsonl.gz","wt") as f:
 for x in bindings:f.write(json.dumps(x,ensure_ascii=False)+"\n")
manifest={"schema":"kdm_original_registered_formal_completed_parts_v1","bounded_claims":[x[1] for x in claims],"sources":sources,"originals_modified":False}
(out/"original_actual_part_source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
summary={"schema":"kdm_matrix_scientific_tuple_intersection_v1","host":args.source_host,"checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"science_scope":"registered Food eval VCD/M3ID main4x4 and ref-off4 each, 40x2424 per model","runtime_fingerprint_used_as_condition":False,"full_decode_config_verified":True,"seed_prompt_scientific_sample_verified":True,"checkpoint_processor_and_adapter_verified":True,"original_same_UID_not_required":True,"actual_uid_changed":uid_changed,"counts":[{"model":m,"measure":k,"count":v} for (m,k),v in sorted(counts.items())],"conditions":[{"model":k[0],"method":k[1],"kind":k[2],"marker":k[3],"reference_marker":k[4],"count":v} for k,v in sorted(conditions.items())],"unresolved_reasons":[{"model":m,"reason":k,"count":v} for (m,k),v in sorted(holds.items())],"actual_parts":len(sources),"unique_reusable_bindings":len(bindings),"bindings_path":str(out/"reusable_scientific_tuple_bindings.jsonl.gz"),"original_parts_manifest_path":str(out/"original_actual_part_source_manifest.json"),"input_decode_source_sha256":sha(root/"src/kdm/decoding.py"),"frozen_samples_sha256":sha(root/"data/current/all.jsonl"),"scope_covers_full_archive":False}
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))

