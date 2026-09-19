"""Validate the fixed census and dataset selection before interventions."""
from collections import Counter
from .io import read_jsonl
from .pipeline import census_tasks, task_id


def validate_census_collection(paths, samples, candidates):
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Invalid fixed candidate inventory")
    expected_samples = {s["id"]: s for s in samples}
    if len(expected_samples) != len(samples):
        raise ValueError("Duplicate manifest samples")
    observed_models = set()
    grouped = {}
    for path in paths:
        rows = list(read_jsonl(path))
        models = {r["model"] for r in rows}
        if len(models) != 1:
            raise ValueError("Each census source file must contain exactly one model")
        model = next(iter(models))
        if model not in candidates:
            raise ValueError("Unexpected model census")
        observed_models.add(model)
        grouped.setdefault(model, []).extend(rows)
    for model, rows in grouped.items():
        expected = {task_id(model, t): t for t in census_tasks(samples)}
        seen = set()
        for r in rows:
            key = r["key"]
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate census task")
            seen.add(key)
            if any(r.get(k) != v for k, v in expected[key].items()):
                raise ValueError("Census task contents differ from the frozen manifest")
            if r.get("status") != "ok" or not r.get("tokens") or not isinstance(r.get("terminated"), bool):
                raise ValueError("Census lacks successful real token and termination evidence")
        if seen != set(expected):
            raise ValueError("Incomplete guided or unguided census")
    if observed_models != set(candidates):
        raise ValueError("Missing fixed candidate census: " + ", ".join(sorted(set(candidates)-observed_models)))


def selected_samples(samples, selection, model):
    rows = [r for r in selection if r["model"] == model]
    datasets = Counter(s["dataset"] for s in samples)
    if len(rows) != len(datasets) or {r["dataset"] for r in rows} != set(datasets):
        raise ValueError("Selection must record every dataset condition for this model")
    for row in rows:
        if row["n"] != datasets[row["dataset"]] or type(row["selected"]) is not bool:
            raise ValueError("Selection denominator or boolean status is invalid")
        if not 0 <= row["n_abstain"] <= row["n"] or row["selected"] != (row["n_abstain"] > 0):
            raise ValueError("Selection must depend solely on original semantic abstention")
    selected = {r["dataset"] for r in rows if r["selected"]}
    return [s for s in samples if s["dataset"] in selected]


def code_identity(root):
    """Record committed source blobs; output-only commits do not alter a run."""
    import subprocess
    paths = ["src/kdm", "configs/kdm", "docs/current/PREREGISTER.md"]
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=root)
    if result.returncode:
        raise ValueError("Commit active source/protocol changes before formal model execution")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=root, text=True)
    if untracked.strip():
        raise ValueError("Untracked active source/protocol must be committed before execution")
    lines = subprocess.check_output(["git", "ls-files", "--stage", "--", *paths], cwd=root, text=True).splitlines()
    if not lines:
        raise ValueError("No committed research source identity")
    return lines


def validate_execution_runtime(root, spec, model, cards):
    """Verify the actual local model/env/GPU locks; usable before native proof exists."""
    import json
    import os
    import sys
    from pathlib import Path
    from importlib.metadata import version
    from .io import within, file_hash
    if spec.get("key") != model or spec.get("availability") != "resolved":
        raise ValueError("Model spec is unresolved or belongs to another candidate")
    if len(cards) != spec.get("gpu_count") or len(cards) != len(set(cards)):
        raise ValueError("Allocated physical GPU count differs from model spec")
    for i, card in enumerate(sorted(cards, key=int)):
        lock = Path(root) / "outputs/locks" / f"gpu_{card}.lock"
        try:
            owned = Path(os.readlink(f"/proc/self/fd/{20+i}"))
        except OSError as exc:
            raise ValueError("Formal execution requires inherited worker.sh GPU locks") from exc
        if owned != lock:
            raise ValueError("Worker lock does not match allocated physical GPU")
    validate_environment(spec)
    from .execution import validate_host, execution_receipt
    validate_host(root, cards, model)
    validate_local_checkpoint(spec)
    return execution_receipt(root, cards, model)


def validate_environment(spec):
    import os,sys
    from importlib.metadata import version
    if os.path.abspath(sys.executable) != os.path.abspath(spec['environment_python']):
        raise ValueError("Use the model spec's environment Python")
    observed={package:version(package) for package in spec['versions']}
    if observed!=spec['versions']:
        raise ValueError('Runtime package version differs from registered spec')
    return {'environment_python':os.path.abspath(sys.executable),'versions':observed}


def validate_processor_runtime(root,spec):
    """CPU processor checks bind actual environment/host without initializing a GPU."""
    from .execution import local_host,read_registry
    host,details=local_host(root)
    if read_registry(root)['model_hosts'].get(spec['key'])!=host:
        raise ValueError('Processor check belongs to a different execution host')
    environment=validate_environment(spec)
    validate_local_checkpoint(spec)
    return {'host':host,'hostname':details['hostname'],**environment}


def validate_runtime(root, spec, model, cards):
    """Local execution admission plus portable native proof admission."""
    receipt = validate_execution_runtime(root, spec, model, cards)
    validate_native_runtime_files(root, spec, model)
    if receipt['host']=='6403':validate_resource_runtime(root,spec)
    return receipt


def validate_native_runtime_files(root,spec,model):
    """Pure file-identity portion of runtime admission, reusable before scheduling."""
    import json
    from pathlib import Path
    from .io import within,file_hash
    check = spec.get("interface_verification", {})
    if not isinstance(check, dict) or check.get("status") != "passed":
        raise ValueError("Native 16-image interface verification has not passed")
    result = json.loads(within(root, check["record"]).read_text())
    if not result.get("passed") or result.get("completed") != 16 or result.get("expected") != 16:
        raise ValueError("Incomplete native-interface evidence")
    verified = result.get("spec", {})
    fields = ("key", "factory", "kwargs", "environment_python", "versions", "dtype", "thinking_mode", "processor", "weights")
    if any(verified.get(field) != spec.get(field) for field in fields):
        raise ValueError("Model, processor or environment changed after interface verification")
    if result["manifest_sha256"] != file_hash(Path(root) / "data/current/interface16.jsonl"):
        raise ValueError("Native-interface sample identity changed")
    dependencies = {"hf.py", "backbone.py"}
    if model in {"minicpm26", "minicpm45", "phi35"}:
        dependencies.add("remote.py")
    if model == "internvl35_8b":
        dependencies.add("internvl_preprocessing.py")
    if spec.get("factory", "").partition(":")[0] == "kdm.models.internvl_dual":
        dependencies.add("internvl_dual.py")
    recorded_sources = result.get("runtime_adapter_sha256", {})
    allowed_sources = {"hf.py", "backbone.py", "sid.py", "remote.py", "internvl_preprocessing.py", "internvl_dual.py"}
    if not dependencies <= set(recorded_sources) or not set(recorded_sources) <= allowed_sources:
        raise ValueError("Native-interface evidence omits required adapter source identities")
    # Native six-condition checks never invoke SID. Its separate proof is checked below.
    for filename in sorted(dependencies):
        if file_hash(Path(root) / "src/kdm/models" / filename) != recorded_sources[filename]:
            raise ValueError("Adapter changed after native-interface verification: " + filename)
    from .execution import REGISTRY, read_registry, validate_execution_receipt
    if (Path(root) / REGISTRY).is_file() and read_registry(root)['model_hosts'].get(model) == '6403':
        validate_execution_receipt(root, result.get('execution'), model, spec.get('gpu_count'))


def validate_local_checkpoint(spec):
    """Only the execution host checks its own registered checkpoint files."""
    from pathlib import Path
    for weight in spec["weights"]:
        path = Path(spec["kwargs"]["model_path"]) / weight["filename"]
        if not path.is_file() or path.stat().st_size != weight["size_bytes"]:
            raise ValueError("Missing or incomplete registered model weight: " + str(path))


def validate_method_runtime(root, spec, methods):
    """Require actual projection or official SID evidence for methods that use it."""
    import json
    from pathlib import Path
    from .io import within, file_hash
    methods=set(methods)
    if methods & {"dola", "deco"}:
        native=json.loads(within(root,spec["interface_verification"]["record"]).read_text())
        projection=native.get("layer_projection_check",{})
        errors=[projection.get(k) for k in ("head_final_error","raw_projection_error","norm_projection_error")]
        if projection.get("status")!="passed" or errors!=[0.,0.,0.] or not projection.get("raw_layers") or not projection.get("normalized_layers"):
            raise ValueError("DoLa/DeCo require verified native layer projection evidence")
    if "sid" not in methods:
        return
    declaration=spec.get("mechanism_validation",{}).get("sid_reference")
    if not isinstance(declaration,dict) or declaration.get("status")!="passed":
        raise ValueError("SID official reference verification has not passed for this architecture")
    proof=json.loads(within(root,declaration["record"]).read_text())
    if proof.get("passed") is not True or proof.get("reference_commit")!="127dd412fa6b61ab1c9babf6979ec4da98002438":
        raise ValueError("SID proof does not establish the fixed official reference")
    required_checks={"official_selection","official_reference_logits","causal_mask_preserved","interleaved_sessions","nonmonotonic_prefix"}
    if any(proof.get("checks",{}).get(k) is not True for k in required_checks):
        raise ValueError("SID matched-prefix and session-isolation checks are incomplete")
    fields=("key","factory","kwargs","environment_python","versions","dtype","thinking_mode","processor","weights")
    if any(proof.get("spec",{}).get(field)!=spec.get(field) for field in fields):
        raise ValueError("SID checkpoint or environment changed after its numerical verification")
    dependencies={"hf.py","backbone.py","sid.py"}
    if spec["key"] in {"minicpm26","minicpm45","phi35"}:dependencies.add("remote.py")
    if spec["key"]=="internvl35_8b":dependencies.add("internvl_preprocessing.py")
    if spec.get("factory", "").partition(":")[0] == "kdm.models.internvl_dual":dependencies.add("internvl_dual.py")
    from .execution import REGISTRY, read_registry, validate_execution_receipt
    if (Path(root) / REGISTRY).is_file() and read_registry(root)['model_hosts'].get(spec['key']) == '6403':
        validate_execution_receipt(root, proof.get('execution'), spec['key'], spec.get('gpu_count'))
        first=next(read_jsonl(Path(root)/'data/current/interface16.jsonl'))
        registry=read_registry(root);catalog=json.loads(within(root,registry['image_catalog']).read_text())
        if proof.get('sample_id')!=first['id'] or proof.get('image_sha256')!=catalog['images'][first['image_path']]['sha256']:
            raise ValueError('Remote SID proof does not use the original fixed Food image')
    sources=proof.get("runtime_adapter_sha256",{})
    if not dependencies <= set(sources):raise ValueError("SID source identity is incomplete")
    for name in sorted(dependencies):
        if file_hash(Path(root)/"src/kdm/models"/name)!=sources[name]:
            raise ValueError("SID implementation changed after numerical verification: "+name)



def validate_resource_runtime(root, spec):
    """Check the registered three finite-prefix maximum-input proofs, without loading weights."""
    import json
    from pathlib import Path
    from .io import within, file_hash
    from .execution import validate_execution_receipt
    declaration=spec.get('resource_verification',{})
    stages={'layer','instruction_vcd','cda'}
    if declaration.get('status')!='passed' or set(declaration.get('records',{}))!=stages:
        raise ValueError('Migrated runtime needs all three maximum-input resource proofs')
    if spec.get('visual_count_verification',{}).get('status')!='passed':
        raise ValueError('Migrated runtime requires current native processor counts')
    required={'verification/max_input_resource_check.py','verification/check_full_visual_counts.py'}
    fields=('key','factory','kwargs','environment_python','versions','dtype','thinking_mode','processor','weights')
    for stage,relative in declaration['records'].items():
        path=within(root,relative);required.add(str(path.relative_to(root)))
        proof=json.loads(path.read_text())
        if proof.get('passed') is not True or proof.get('stage')!=stage or proof.get('phase')!='complete':
            raise ValueError('Maximum-input resource evidence is incomplete')
        if any(proof.get('spec',{}).get(k)!=spec.get(k) for k in fields):
            raise ValueError('Maximum-input checkpoint/processor/environment identity changed')
        if proof.get('script_sha256')!=file_hash(Path(root)/'verification/max_input_resource_check.py'):
            raise ValueError('Maximum-input resource check source changed')
        if proof.get('manifest_sha256')!=file_hash(Path(root)/'data/current/all.jsonl'):
            raise ValueError('Maximum-input proof has a different full manifest')
        if proof['count_record']!=spec['visual_count_verification']['record']:
            raise ValueError('Resource proof uses an unregistered visual-count record')
        countpath=within(root,proof['count_record']);required.add(str(countpath.relative_to(root)))
        if proof.get('count_record_sha256')!=file_hash(countpath):
            raise ValueError('Maximum visual-count identity changed')
        counts=json.loads(countpath.read_text())
        if counts.get('manifest_sha256')!=proof['manifest_sha256'] or counts.get('boundary_checks_passed') is not True or counts.get('total_images')!=9167:
            raise ValueError('Maximum visual-count proof is incomplete')
        if any(counts.get('spec',{}).get(k)!=spec.get(k) for k in fields):
            raise ValueError('Visual-count processor/environment changed')
        from .execution import read_registry
        registry=read_registry(root);host=registry['model_hosts'][spec['key']]
        expected_processor={'host':host,'hostname':registry['hosts'][host]['hostname'],'environment_python':spec['environment_python'],'versions':spec['versions']}
        if counts.get('processor_execution')!=expected_processor:raise ValueError('Native count environment was not verified on its target host')
        if counts.get('script_sha256')!=file_hash(Path(root)/'verification/check_full_visual_counts.py'):
            raise ValueError('Native visual-count check source changed')
        headerpath=within(root,counts['header_record']);required.add(str(headerpath.relative_to(root)))
        if counts.get('header_record_sha256')!=file_hash(headerpath):raise ValueError('Image-header record changed')
        maximum=max(g['maximum'] for g in counts['groups'] if g['dataset']=='vizwiz')
        if proof.get('expected_visual_tokens')!=maximum:
            raise ValueError('Resource evidence does not use the maximum actual VizWiz input')
        samples={sample['id']:sample for sample in read_jsonl(Path(root)/'data/current/all.jsonl')}
        sample=proof.get('sample',{});original=samples.get(sample.get('id'),{})
        if not original or sample!={k:original[k] for k in ('id','dataset','split','image_path')} or sample['dataset']!='vizwiz':
            raise ValueError('Resource sample differs from the original VizWiz manifest')
        if not any(row.get('id')==sample['id'] and row.get('formula_tokens')==maximum and row.get('equal') is True for row in counts['boundary_checks']):
            raise ValueError('Resource sample lacks a matching maximum native count')
        visits=proof.get('visits',[]);branches={'layer':['main_need_layers'],'instruction_vcd':['main','noise_reference','neutral'],'cda':['prior_text','context_image','abstention_image','null_prior_text','null_context_image']}[stage]
        if len(visits)!=len(branches)*3 or {(v.get('prefix_length'),v.get('branch')) for v in visits}!={(length,branch) for length in (0,1,2) for branch in branches} or any(v.get('logits_finite') is not True or v.get('all_layer_logits_finite') is not True for v in visits):
            raise ValueError('Resource evidence lacks the fixed finite-prefix branch checks')
        validate_execution_receipt(root,proof.get('execution'),spec['key'],spec.get('gpu_count'))
        for name in ('hf.py','backbone.py'):
            if proof.get('runtime_adapter_sha256',{}).get(name)!=file_hash(Path(root)/'src/kdm/models'/name):
                raise ValueError('Resource adapter identity changed')
    return required


def validate_freeze(root):
    """Validate one immutable protocol identity before formal census/experiment work."""
    import json
    from pathlib import Path
    from .io import within,file_hash
    from .reports import validated_method_plan
    root=Path(root).resolve()
    receipt_path=root/'outputs/records/preregistration_freeze.json'
    freeze=json.loads(receipt_path.read_text())
    if freeze.get('status')!='frozen' or not isinstance(freeze.get('files'),dict):
        raise ValueError('Protocol has not been frozen with file identities')
    for relative,expected in freeze['files'].items():
        path=within(root,relative)
        if path==receipt_path:raise ValueError('Freeze receipt must not reference itself')
        if file_hash(path)!=expected:raise ValueError('Frozen identity changed: '+relative)
    panel=json.loads((root/'configs/kdm/models.json').read_text())
    keys=[row['key'] for row in panel]
    if len(keys)!=16 or len(set(keys))!=16:raise ValueError('Freeze requires the fixed sixteen-model inventory')
    required={'docs/current/PREREGISTER.md','data/current/all.jsonl','data/current/interface16.jsonl',
        'configs/kdm/models.json','configs/kdm/food_aliases.json','configs/kdm/method_plan.json',
        'outputs/records/protocol_user_decisions_20260919.json','configs/runtime/semantic_judge.json',
        'scripts/run_census_panel.py','scripts/worker.sh','scripts/verify_complete.py',
        'configs/runtime/hosts.json','outputs/records/image_content_catalog.json'}
    samples=list(read_jsonl(root/'data/current/all.jsonl'))
    if len(samples)!=9167 or len({row['id'] for row in samples})!=9167 or Counter(row['dataset'] for row in samples)!={'food101':4848,'vizwiz':4319}:
        raise ValueError('Freeze requires all original 9167 Food-101/VizWiz samples')
    catalog=json.loads((root/'outputs/records/image_content_catalog.json').read_text())
    if catalog.get('schema')!=1 or catalog.get('manifest_sha256')!=file_hash(root/'data/current/all.jsonl') or set(catalog.get('images',{}))!={row['image_path'] for row in samples}:
        raise ValueError('Image content catalog must cover the exact full original manifest')
    if any(not isinstance(v.get('sha256'),str) or len(v['sha256'])!=64 or type(v.get('size_bytes')) is not int or v['size_bytes']<=0 for v in catalog['images'].values()):
        raise ValueError('Image content catalog lacks complete byte identities')
    plan=json.loads((root/'configs/kdm/method_plan.json').read_text())
    methods=validated_method_plan(plan,[(key,dataset) for key in keys for dataset in ('food101','vizwiz')])
    from .execution import read_registry
    registry=read_registry(root)
    if set(registry['model_hosts'])!=set(keys):raise ValueError('Host registry must assign all sixteen models exactly once')
    specs={}
    for key in keys:
        relative=f'configs/runtime/{key}.json';required.add(relative)
        spec=json.loads((root/relative).read_text());specs[key]=spec
        if spec.get('key')!=key or spec.get('availability')!='resolved':raise ValueError('Unresolved frozen runtime: '+key)
        declaration=spec.get('interface_verification',{})
        if not isinstance(declaration,dict) or declaration.get('status')!='passed':raise ValueError('Native interface is incomplete: '+key)
        proof_path=within(root,declaration['record']);required.add(str(proof_path.relative_to(root)))
        proof=json.loads(proof_path.read_text())
        if proof.get('passed') is not True or proof.get('completed')!=16 or proof.get('expected')!=16:
            raise ValueError('Native interface is incomplete: '+key)
        requested=set(method for values in methods[key].values() for method in values)
        if 'sid' in requested:
            sid=spec.get('mechanism_validation',{}).get('sid_reference')
            if not isinstance(sid,dict) or sid.get('status')!='passed':raise ValueError('Frozen SID method proof is incomplete: '+key)
            required.add(str(within(root,sid['record']).relative_to(root)))
    for key,spec in specs.items():
        if registry['model_hosts'][key]=='6403':
            required.update(validate_resource_runtime(root,spec))
    if not required<=set(freeze['files']):
        raise ValueError('Freeze receipt omits required protocol/data/runtime/proof identities: '+', '.join(sorted(required-set(freeze['files']))))
    sources=code_identity(root)
    if freeze.get('source_blobs')!=sources:raise ValueError('Active source identity differs from frozen source blobs')
    for key,spec in specs.items():
        validate_native_runtime_files(root,spec,key)
        validate_method_runtime(root,spec,{method for values in methods[key].values() for method in values})
    return freeze
