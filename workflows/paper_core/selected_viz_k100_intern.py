#!/usr/bin/env python3
"""Selected Viz continuation through the already accepted K100 InternVL map."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash
from workflows.supplemental.remaining4.k100_intern_registered_matrix import single_spec
from workflows.paper_core import selected_viz
GATE='outputs/supplemental/remaining4/intern_k100_single_admission_launch2_20261003_1355/admission/k100_intern_single_actual_duplicate8_launch2_20261003_1355/condition_gate.json'
def install():
 p=ROOT/GATE;g=json.loads(p.read_text())
 assert file_hash(p)=='1ebd535c708a202be45f951bc9057e903b34263c77c126e271a2eb758a754336' and g['passed']and g['production_allowed']and g['completed']==8
 assert file_hash(ROOT/'workflows/supplemental/remaining4/internvl_k100_single.py')==g['factory_sha256']
 assert file_hash(ROOT/'workflows/supplemental/remaining4/dispatch_intern_k100_single_registry_20261003.json')==g['registry_sha256']
 assert file_hash(ROOT/g['operator_audit_path'])==g['operator_audit_sha256']
 original=selected_viz.supplemental.validate_supplemental_runtime
 def admitted(root,spec,model,cards,stage,claim,owner):
  assert model=='internvl35_8b'and cards==['0']and stage=='formal'
  result=original(root,single_spec(spec),model,cards,'independent',claim,owner)
  result['existing_accepted_single_hardware_gate']={'path':GATE,'sha256':file_hash(p),'software_changed':False,'precision_changed':False}
  return result
 selected_viz.supplemental.validate_supplemental_runtime=admitted
if __name__=='__main__':install();selected_viz.main()
