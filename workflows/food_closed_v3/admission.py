"""Explicit user-authorized GPU0 addition, without editing frozen registry."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import kdm.execution as execution
original_read_registry=execution.read_registry
def expanded_registry(root):
    data=original_read_registry(root)
    h=data["hosts"]["6403"]
    h["allowed_gpus"]=[0,1]
    h["gpu_uuids"]["0"]="GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402"
    return data
execution.read_registry=expanded_registry
if __name__=="__main__":
    execution.validate_host(ROOT,sys.argv[1].split(","))
