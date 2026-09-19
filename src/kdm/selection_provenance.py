"""Consume a selection only with its complete, frozen original census provenance."""
import json
from pathlib import Path
from .io import file_hash, read_jsonl, within
from .provenance import validate_census_inputs
from .protocol import selected_samples


def _source_relative(root, path):
    """Normalize registered project roots for comparison, never rewrite saved rows."""
    value=Path(path)
    if not value.is_absolute() or value.is_relative_to(root):
        return str(within(root,value).relative_to(root))
    from .execution import read_registry
    roots=[Path(host['root']) for host in read_registry(root)['hosts'].values()]
    matches=[str(value.relative_to(base)) for base in roots if value.is_relative_to(base)]
    if len(matches)!=1 or '..' in value.parts:
        raise ValueError('Selection source is outside registered project roots')
    return str(within(root,matches[0]).relative_to(root))


def validate_selection(root, selection_path, manifest_path, freeze):
    """Caller supplies validate_freeze(root); no new selection rule or human judgment.

    The original select receipt binds the selection bytes and annotation SHA.
    Every census source and identity sidecar is checked again against the complete
    frozen panel. The annotation SHA is retained as a recorded identity; this does
    not reperform or pretend to revalidate the original human review.
    """
    root=Path(root).resolve()
    path=within(root,selection_path);manifest=within(root,manifest_path)
    sidecar=path.with_suffix('.sources.json')
    if not sidecar.is_file():raise ValueError('Selection requires its original source receipt')
    selection_sha=file_hash(path);sidecar_sha=file_hash(sidecar)
    receipt=json.loads(sidecar.read_text())
    if receipt.get('operation')!='select' or receipt.get('output_sha256')!=selection_sha:
        raise ValueError('Selection output differs from its original select source receipt')
    if receipt.get('complete_panel') is not True or receipt.get('formal_evidence') is not True:
        raise ValueError('Selection requires the complete formal census panel')
    annotation_sha=receipt.get('annotations_sha256')
    if not isinstance(annotation_sha,str) or len(annotation_sha)!=64 or any(c not in '0123456789abcdef' for c in annotation_sha):
        raise ValueError('Selection receipt lacks its original annotation identity')
    sources=receipt.get('sources')
    if not isinstance(sources,list) or not sources:
        raise ValueError('Selection source receipt lacks census inputs')
    paths=[within(root,source['path']) for source in sources]
    checked=validate_census_inputs(root,paths,manifest,freeze,require_complete_panel=True)
    if any(receipt.get(key)!=value for key,value in checked.items()):
        raise ValueError('Selection source receipt differs from its revalidated census provenance')
    selection=json.loads(path.read_text());samples=list(read_jsonl(manifest))
    if not isinstance(selection,list) or any(not isinstance(row,dict) for row in selection):
        raise ValueError('Selection must be a list of model/dataset decisions')
    models=set(checked['models'])
    if {row.get('model') for row in selection}!=models:
        raise ValueError('Selection model coverage differs from the complete census panel')
    for model in sorted(models):
        selected_samples(samples,selection,model)
    for row in selection:
        values=row.get('selection_sources')
        if not isinstance(values,list) or [_source_relative(root,value) for value in values]!=checked['models'][row['model']]['paths']:
            raise ValueError('Selection model sources differ from original census source receipt')
    if file_hash(path)!=selection_sha or file_hash(sidecar)!=sidecar_sha:
        raise ValueError('Selection changed during source verification')
    return {'selection':selection,'provenance':{'schema':'kdm_selection_input_provenance_v1',
        'selection_path':str(path.relative_to(root)),'selection_sha256':selection_sha,
        'source_receipt_path':str(sidecar.relative_to(root)),'source_receipt_sha256':sidecar_sha,
        'recorded_annotations_sha256':annotation_sha,'census':checked}}
