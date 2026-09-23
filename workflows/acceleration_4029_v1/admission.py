"""Versioned4029 relocation for qwen35_4b only; frozen sources/specs are untouched."""
from __future__ import annotations
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from kdm.io import file_hash,stable_hash,within
import kdm.execution as execution
import kdm.protocol as protocol

TARGET_ROOT=Path("/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation")
HOSTNAME="WS-4029GP-TRT"
MODEL="qwen35_4b"
PYTHON=TARGET_ROOT/".environments/native311/bin/python"
MODEL_PATH=TARGET_ROOT/"models/Qwen3.5-4B"
READ_ONLY_MODEL_ROOT=Path("/home/hdd3/zhanghaonan/projects/holocue/models/Qwen3.5-4B")
UUIDS={"4":"GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
       "5":"GPU-9b75e96c-6e9c-a160-def5-b2d5dd70bdc4",
       "6":"GPU-29f1d73d-b468-c7bc-3bcf-50be58b67d33",
       "7":"GPU-56668d27-36e1-c40a-c1c2-272ef76b13a1"}
VERSIONS={"torch":"2.9.0+cu128","transformers":"5.17.0","tokenizers":"0.23.2","numpy":"2.3.5",
          "Pillow":"12.0.0","safetensors":"0.8.0","accelerate":"1.12.0","scipy":"1.17.0","torchvision":"0.24.0+cu128"}
DEFINITION={"schema":"kdm_qwen35_4b_4029_relocation_v1","host":"4029","hostname":HOSTNAME,
            "root":str(TARGET_ROOT),"model":MODEL,"physical_gpu_uuids":UUIDS,
            "environment_python":str(PYTHON),"model_reference_path":str(MODEL_PATH),
            "allowed_read_only_model_target":str(READ_ONLY_MODEL_ROOT),"versions":VERSIONS,
            "method_parameters_changed":False,"source_spec_modified":False,"n_shards":4,
            "shard_to_gpu":{str(i):str(i+4) for i in range(4)}}
_original_registry=execution.read_registry
_original_resolver=execution.resolve_image_path
_installed=False
_verified_images={}
_active_shard=None


def expanded_registry(root):
 data=copy.deepcopy(_original_registry(root))
 data["hosts"]["4029"]={"hostname":HOSTNAME,"root":str(TARGET_ROOT),"allowed_gpus":[4,5,6,7],"gpu_uuids":dict(UUIDS)}
 data["model_hosts"][MODEL]="4029"
 return data


def check_host(root,cards,query=True):
 if Path(root).resolve()!=TARGET_ROOT or socket.gethostname()!=HOSTNAME:
  raise ValueError("4029 relocation requires exact registered hostname/project root")
 if len(cards)!=1 or cards[0] not in UUIDS:raise ValueError("Exactly one of4029 physical GPUs4/5/6/7 required")
 if query:
  rows=subprocess.check_output(["nvidia-smi","-i",cards[0],"--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True).splitlines()
  if len(rows)!=1 or [x.strip() for x in rows[0].split(",")]!=[cards[0],UUIDS[cards[0]]]:
   raise ValueError("4029 physical GPU UUID mismatch")


def runtime_spec(source_spec):
 if source_spec.get("key")!=MODEL or source_spec.get("gpu_count")!=1:
  raise ValueError("4029 overlay only permits original single-card qwen35_4b")
 # Original source spec remains the input to all native/method proof checks.
 relocated=copy.deepcopy(source_spec)
 relocated["environment_python"]=str(PYTHON)
 relocated["kwargs"]["model_path"]=str(MODEL_PATH)
 relocated["processor"]["path"]=str(MODEL_PATH)
 return relocated


def validate_versions(observed):
 if observed!=VERSIONS:raise ValueError("4029 environment versions differ from the source environment")


def validate_model_files(source_spec):
 actual=MODEL_PATH.resolve()
 if actual not in {MODEL_PATH,READ_ONLY_MODEL_ROOT}:raise ValueError("Model symlink escaped approved read-only roots")
 if not actual.is_dir():raise ValueError("Approved model root is missing")
 for weight in source_spec["weights"]:
  path=actual/weight["filename"]
  if not path.is_file() or path.stat().st_size!=weight["size_bytes"]:
   raise ValueError("Missing/incomplete same-checkpoint weight")
 if file_hash(actual/"config.json")!=source_spec["model_config_sha256"]:raise ValueError("Model config changed")
 for name,digest in source_spec["processor"]["files"].items():
  if file_hash(actual/name)!=digest:raise ValueError("Processor file changed: "+name)
 return str(actual)


def image_resolver(logical,root=None):
 root=Path(root or ROOT).resolve()
 if root!=TARGET_ROOT:raise ValueError("Relocated image resolver only applies on4029 target root")
 target=_original_resolver(logical,root)
 data=expanded_registry(root);catalog_path=within(root,data["image_catalog"])
 stat_catalog=catalog_path.stat()
 catalog=execution._catalog(str(catalog_path),(stat_catalog.st_ino,stat_catalog.st_size,stat_catalog.st_mtime_ns,stat_catalog.st_ctime_ns))
 if catalog.get("schema")!=1 or catalog["manifest_sha256"]!=execution.unchanged_file_hash(root/"data/current/all.jsonl"):
  raise ValueError("Original image content catalog mismatch")
 expected=catalog["images"].get(str(logical))
 if not isinstance(expected,dict):raise ValueError("Mapped image lacks frozen content identity")
 stat=target.stat();signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
 key=(str(target),expected["sha256"],expected["size_bytes"])
 if _verified_images.get(key)!=signature:
  if stat.st_size!=expected["size_bytes"] or file_hash(target)!=expected["sha256"]:
   raise ValueError("4029 image bytes differ from original manifest catalog")
  after=target.stat()
  if signature!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
   raise ValueError("Image changed during verification")
  _verified_images[key]=signature
 return target


def install(shard):
 global _installed,_active_shard
 if type(shard) is not int or not 0<=shard<4:raise ValueError("Only shards0..3/n4")
 if _installed and _active_shard!=shard:raise ValueError("Cannot change shard in the same process")
 _active_shard=shard
 if not _installed:
  execution.read_registry=expanded_registry
  execution.resolve_image_path=image_resolver
  import kdm.pipeline as pipeline
  pipeline.resolve_image_path=image_resolver
  _installed=True


def validate_runtime(root,source_spec,model,cards):
 if model!=MODEL:raise ValueError("Unregistered4029 model")
 check_host(root,cards)
 if _active_shard is None or cards!=[str(_active_shard+4)]:
  raise ValueError("4029 shard must use its assigned physical GPU")
 if Path(os.readlink("/proc/self/fd/20"))!=Path(root)/"outputs/locks"/f"gpu_{cards[0]}.lock":
  raise ValueError("Inherited approved GPU lock missing")
 if os.path.abspath(sys.executable)!=str(PYTHON) or sys.version_info[:2]!=(3,11):
  raise ValueError("Use the relocated Python3.11 environment")
 observed={name:version(name) for name in VERSIONS};validate_versions(observed)
 spec_path=Path(root)/f"configs/runtime/{MODEL}.json"
 if source_spec!=json.loads(spec_path.read_text()):raise ValueError("Do not pass a modified spec to provenance validation")
 freeze=protocol.validate_freeze(root)
 if file_hash(spec_path)!=freeze["files"][str(spec_path.relative_to(root))]:raise ValueError("Frozen source spec changed")
 protocol.validate_native_runtime_files(root,source_spec,model)
 methods=json.loads((Path(root)/"configs/kdm/method_plan.json").read_text())[model]["food101"]
 protocol.validate_method_runtime(root,source_spec,methods)
 actual=validate_model_files(source_spec)
 catalog_path=within(root,expanded_registry(root)["image_catalog"])
 return {"host":"4029","hostname":HOSTNAME,"project_root":str(TARGET_ROOT),"physical_gpus":cards,
         "gpu_uuids":{cards[0]:UUIDS[cards[0]]},"migration_definition":DEFINITION,
         "migration_definition_sha256":stable_hash(DEFINITION),"original_registry_sha256":file_hash(Path(root)/"configs/runtime/hosts.json"),
         "original_spec_sha256":file_hash(spec_path),"runtime_environment_python":str(PYTHON),"runtime_versions":observed,
         "runtime_model_path":str(MODEL_PATH),"runtime_model_resolved_read_only":actual,
         "image_catalog_sha256":file_hash(catalog_path),
         "original_native_and_method_evidence_validated":True,"new_host_native_equivalence_claimed":False,
         "relocation_files_sha256":{name:file_hash(ROOT/"workflows/acceleration_4029_v1"/name) for name in ("admission.py","runner.py","worker.sh")},
         "parent_formal_runner_sha256":file_hash(ROOT/"workflows/acceleration_v4/formal_runner.py")}


def make_backend(source_spec,device):
 validate_model_files(source_spec)
 from kdm.pipeline import make_backend as original_make_backend
 return original_make_backend(runtime_spec(source_spec),device)


if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--root",required=True);parser.add_argument("--cards",required=True)
 args=parser.parse_args();check_host(args.root,args.cards.split(","))
 if os.path.abspath(sys.executable)!=str(PYTHON):raise ValueError("Wrong relocated Python")
 validate_versions({name:version(name) for name in VERSIONS})
 print(json.dumps({"host":"4029","cards":args.cards,"gpu_uuid_checked":True,"versions_match":True}))

