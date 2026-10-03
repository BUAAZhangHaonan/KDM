#!/usr/bin/env python3
"""Bound actual load/generation time while using the original native Viz entry."""
import argparse,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.supplemental.remaining4 import native
from workflows.paper_core.ip_dev_full import BudgetInputs

def main():
 p=argparse.ArgumentParser();p.add_argument('--allocated-gpu-seconds',type=float,required=True);a,rest=p.parse_known_args()
 assert a.allocated_gpu_seconds>0 and '--dataset'in rest and rest[rest.index('--dataset')+1]=='vizwiz'
 assert '--method'in rest and rest[rest.index('--method')+1]=='vcd'
 started=time.perf_counter();factory=native.make_backend
 def budgeted(spec,*args,**kwargs):
  backend=factory(spec,*args,**kwargs)
  return BudgetInputs(backend,started,spec['gpu_count'],a.allocated_gpu_seconds,ROOT/'outputs/unused_fixed_native_viz_input_capture.jsonl')
 native.make_backend=budgeted;sys.argv=[str(ROOT/'workflows/supplemental/remaining4/native.py'),*rest];native.main()
if __name__=='__main__':main()
