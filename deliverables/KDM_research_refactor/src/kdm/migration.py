"""Archive stage scripts and extract model construction without old experiment code."""
from __future__ import annotations
import ast,hashlib,json,shutil,subprocess
from pathlib import Path
from .io import atomic_json,file_hash,within

REQUIRED_BACKBONE_BLOB='fc94a965cfc9b3bbba94ca2cc63aea3018c36fd6'
KEEP_METHODS={'FamilyModel':{'__init__','_patch_conv3d','build'},
              'InternVLModel':{'__init__','build','_prefill','_step'}}


def extract_backbone(source):
    tree=ast.parse(source);items=[];found=set()
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.ImportFrom) and node.module=='engine':continue
            items.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NORM_CHAINS' for t in node.targets):items.append(node)
        elif isinstance(node,ast.ClassDef) and node.name in KEEP_METHODS:
            node.body=[x for x in node.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in KEEP_METHODS[node.name]]
            names={x.name for x in node.body}
            if names!=KEEP_METHODS[node.name]:raise ValueError('Backbone class contract changed')
            items.append(node);found.add(node.name)
        elif isinstance(node,ast.FunctionDef) and node.name=='get_engine':items.append(node);found.add(node.name)
    if found!={'FamilyModel','InternVLModel','get_engine'}:raise ValueError('Missing original model constructors')
    for item in items:
        if isinstance(item,ast.ClassDef) and item.name=='FamilyModel':
            init=next(x for x in item.body if x.name=='__init__')
            init.args.args.extend([ast.arg(arg='device_map'),ast.arg(arg='max_memory')])
            init.args.defaults.extend([ast.Constant(None),ast.Constant(None)])
            for call in ast.walk(init):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='from_pretrained':
                    for kw in call.keywords:
                        if kw.arg=='device_map':
                            kw.value=ast.IfExp(test=ast.Compare(ast.Name('device_map',ast.Load()),[ast.IsNot()],[ast.Constant(None)]),body=ast.Name('device_map',ast.Load()),orelse=ast.Name('device',ast.Load()))
                            call.keywords.append(ast.keyword(arg='max_memory',value=ast.Name('max_memory',ast.Load())))
        if isinstance(item,ast.FunctionDef) and item.name=='get_engine':
            item.args.args.extend([ast.arg(arg='device_map'),ast.arg(arg='max_memory')])
            item.args.defaults.extend([ast.Constant(None),ast.Constant(None)])
            for call in ast.walk(item):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='FamilyModel':
                    call.keywords.extend([ast.keyword('device_map',ast.Name('device_map',ast.Load())),ast.keyword('max_memory',ast.Name('max_memory',ast.Load()))])
    module=ast.Module(body=items,type_ignores=[]);ast.fix_missing_locations(module)
    output='"""Model construction extracted from the KDM source recorded in migration.json."""\n'+ast.unparse(module)+'\n'
    compile(output,'backbone.py','exec');return output


def migrate(project,package,apply=False,allow_source_change=False):
    root=Path(project).resolve();package=Path(package).resolve()
    if not (root/'.git').exists():raise ValueError('Target must be an existing Git repository')
    if not (root/'code/stage3_engine.py').is_file():raise FileNotFoundError('Original stage3_engine.py is required')
    status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
    if status.strip():raise RuntimeError('Commit or explicitly resolve current working-tree changes before migration')
    src=(root/'code/stage3_engine.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest()
    if blob!=REQUIRED_BACKBONE_BLOB and not allow_source_change:raise ValueError('Source revision differs; record a reviewed change before applying')
    backbone=extract_backbone(src.decode())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    archive=root/'archive'/('pre_refactor_'+commit[:12])
    if archive.exists():raise FileExistsError(archive)
    # Validate every destination before moving anything.
    for sub in ('src','tests','scripts','paper','verification','data_example','agent_prompts','docs/current','configs/kdm','source_materials/kdm'):
        if (root/sub).exists():raise FileExistsError(f'Existing {sub}; inspect before migration')
    for sub in ('src','tests','scripts','paper','verification','data_example','agent_prompts','docs','configs','source_materials'):
        if not (package/sub).is_dir():raise FileNotFoundError(package/sub)
    for name in ('README.md','AGENTS.md','pyproject.toml'):
        if not (package/name).is_file():raise FileNotFoundError(package/name)
    payload={'source_commit':commit,'source_blob':blob,'archive':str(archive.relative_to(root)),
             'moves':['code','README.md'],'copy':['src','tests','scripts','configs','docs','paper','pyproject.toml'],
             'keeps':['outputs/raw','outputs/tables','outputs/figures','data','git history']}
    if not apply:return payload
    archive.mkdir(parents=True)
    shutil.move(root/'code',archive/'code')
    if (root/'README.md').exists():shutil.move(root/'README.md',archive/'README.md')
    # Preserve all existing documents and configs; place new material in named folders.
    for sub in ('src','tests','scripts','paper','verification','data_example','agent_prompts'):
        shutil.copytree(package/sub,root/sub)
    shutil.copytree(package/'docs',root/'docs'/'current')
    shutil.copytree(package/'configs',root/'configs'/'kdm')
    shutil.copytree(package/'source_materials',root/'source_materials'/'kdm')
    shutil.copy2(package/'pyproject.toml',root/'pyproject.toml')
    (root/'README.md').write_text((package/'README.md').read_text().replace('`docs/', '`docs/current/'))
    if (root/'AGENTS.md').exists():shutil.copy2(root/'AGENTS.md',archive/'AGENTS.md')
    shutil.copy2(package/'AGENTS.md',root/'AGENTS.md')
    (root/'src/kdm/models/backbone.py').write_text(backbone)
    atomic_json(root/'docs/current/MIGRATION_RECORD.json',payload)
    return payload
