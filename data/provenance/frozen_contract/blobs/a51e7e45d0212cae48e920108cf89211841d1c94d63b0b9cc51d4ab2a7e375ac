#!/usr/bin/env python3
"""Retain every original sample in selected model--dataset conditions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdm.io import read_jsonl, within, file_hash, atomic_json
from kdm.data import write_manifest
from kdm.protocol import selected_samples, validate_method_runtime, validate_freeze
from kdm.selection_provenance import validate_selection
from kdm.reports import validated_method_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--method-plan", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    root=Path(a.root).resolve()
    manifest_path=within(root,a.manifest);selection_path=within(root,a.selection)
    samples = list(read_jsonl(manifest_path))
    freeze=validate_freeze(root)
    verified_selection=validate_selection(root,selection_path,manifest_path,freeze)
    selection=verified_selection['selection']
    plan_path=within(root,a.method_plan);plan_relative=str(plan_path.relative_to(root))
    if freeze.get('status')!='frozen' or freeze.get('files',{}).get(plan_relative)!=file_hash(plan_path):
        raise ValueError('Selected execution plan differs from frozen method-plan identity')
    plan=validated_method_plan(json.loads(plan_path.read_text()),[(r['model'],r['dataset']) for r in selection])
    planned=[]
    for model in sorted({r['model'] for r in selection}):
        rows=selected_samples(samples,selection,model)
        if not rows:continue
        spec_path=within(root,Path('configs/runtime')/(model+'.json'))
        runtime=json.loads(spec_path.read_text())
        if runtime.get('key')!=model:raise ValueError('Runtime spec belongs to a different model')
        for dataset in sorted({r['dataset'] for r in rows}):
            subset=[row for row in rows if row['dataset']==dataset]
            methods=plan[model][dataset]
            validate_method_runtime(root,runtime,methods)
            out=within(root,Path(a.out_dir)/model/(dataset+'.jsonl'))
            planned.append((model,dataset,subset,methods,out,spec_path))
    entries=[]
    for model,dataset,subset,methods,out,spec_path in planned:
        if out.exists():
            if list(read_jsonl(out))!=subset:raise ValueError('Cannot overwrite a different selected manifest')
        else:write_manifest(subset,out,image_root=root)
        entries.append({'model':model,'dataset':dataset,'manifest':str(out.relative_to(root)),
            'manifest_sha256':file_hash(out),'samples':len(subset),'eval_samples':sum(s['split']=='eval' for s in subset),
            'methods':list(methods),'model_spec':str(spec_path.relative_to(root)),
            'experiment_arguments':['--mode','experiment','--manifest',str(out.relative_to(root)),
                '--model',model,'--model-spec',str(spec_path.relative_to(root)),'--methods',','.join(methods),
                '--method-plan',plan_relative],
            'execution_note':'GPU allocation and output path must come from the approved worker schedule; runtime proof remains a separate gate'})
        print(model,dataset,len(subset),out)
    receipt={'schema':'kdm_selected_model_dataset_manifests_v1','source_manifest_sha256':file_hash(manifest_path),
        'selection_sha256':file_hash(selection_path),'selection_provenance':verified_selection['provenance'],'method_plan':plan_relative,
        'method_plan_sha256':file_hash(plan_path),'conditions':entries,
        'sample_policy':'all original samples and dev/eval assignments in each selected condition'}
    index=within(root,Path(a.out_dir)/'execution_plan.json')
    if index.exists() and json.loads(index.read_text())!=receipt:raise ValueError('Cannot overwrite a different selected execution plan')
    if not index.exists():atomic_json(index,receipt)


if __name__ == "__main__":
    main()
