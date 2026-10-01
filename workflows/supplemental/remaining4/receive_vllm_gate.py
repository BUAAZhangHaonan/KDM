"""Receive explicit actual Viz engine gate derivatives without touching producer paths."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--portable-manifest', required=True)
parser.add_argument('--archive', required=True)
parser.add_argument('--target', required=True)
args = parser.parse_args()
portable_path, archive_path, target = [ROOT / name for name in
    (args.portable_manifest, args.archive, args.target)]
target.resolve().relative_to(ROOT / 'outputs/supplemental/remaining4/dispatch_20261001_1600')
if (target / 'received_manifest.json').exists():
    raise ValueError('An existing received event cannot be overwritten')
portable = json.loads(portable_path.read_text())
if portable['rows'] != 160 or portable['new_GPU_regeneration_calls'] != 0 or not portable['raw_fields_exactly_preserved']:
    raise ValueError('Only the two explicit actual eighty-key CPU recording derivatives may be received')
mapping = {}
with tarfile.open(archive_path, 'r:gz') as archive:
    for item in portable['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('An explicit source path escapes its original project')
        member = archive.getmember(str(relative))
        if not member.isfile() or member.size != item['bytes']:
            raise ValueError('An explicit source archive member differs')
        data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('An actual source SHA differs after transfer')
        received = target / 'sources' / relative
        received.parent.mkdir(parents=True, exist_ok=True)
        with received.open('xb') as stream:
            stream.write(data)
        mapping[str(relative)] = {'path': received, 'sha256': item['sha256']}
source_relative = next(name for name in mapping if name.endswith('/combined_source_manifest.json'))
source = json.loads(mapping[source_relative]['path'].read_text())
if (source['rows'] != 160 or source['active_raw_opened'] != 0
        or not source['original_sources_preserved'] or not source['original_failed_sidecars_still_failed']):
    raise ValueError('The explicit recording derivative source provenance differs')
parts, verification, all_keys = [], [], set()
for part in source['parts']:
    received_part = {**part, 'files': []}
    by_kind = {}
    for item in part['files']:
        relative = str(Path(item['path']).relative_to(part['source_root']))
        received = mapping[relative]
        if received['sha256'] != item['sha256']:
            raise ValueError('The actual part member differs from its portable source proof')
        path = str(received['path'].relative_to(ROOT))
        received_part['files'].append({**item, 'received_path': path})
        by_kind[item['kind']] = received['path']
    complete = json.loads(by_kind['complete'].read_text())
    meta = json.loads(by_kind['identity'].read_text())
    if (not complete['generation_complete'] or complete['rows'] != 80
            or complete['raw_sha256'] != part['raw_sha256'] or complete['identity_sha256'] != part['identity_sha256']):
        raise ValueError('The actual CPU-sealed source receipt differs')
    raw_rows = list(map(json.loads, by_kind['raw'].read_text().splitlines()))
    if len(raw_rows) != 80 or any(row['identity'] != meta['identity'] for row in raw_rows):
        raise ValueError('The actual received eighty-row derivative is incomplete')
    ownership = {row['key'] for row in map(json.loads, by_kind['keys'].read_text().splitlines())}
    if {row['key'] for row in raw_rows} != ownership or len(ownership) != 80 or ownership & all_keys:
        raise ValueError('The actual source/owner eighty keys differ or overlap')
    all_keys.update(ownership)
    received_part.update(received_raw_path=str(by_kind['raw'].relative_to(ROOT)),
        received_identity_path=str(by_kind['identity'].relative_to(ROOT)),
        received_receipt_path=str(by_kind['complete'].relative_to(ROOT)),
        source_part_generation_complete=True)
    verification.append({'model': part['model'], 'rows': 80, 'actual_owned_keys': 80,
        'part_sha_sources_verified': True, 'source_kind': complete['source_kind'],
        'new_GPU_regeneration_calls': 0, 'original_sidecar_failure_preserved': True})
    parts.append(received_part)
result = {'schema': 'kdm_selected4_registered_received_manifest_v1',
    'local_source_sha_verified': True, 'active_raw_opened': 0, 'gpu_queries': 0,
    'original_sources_preserved': True, 'received_utc': datetime.now(timezone.utc).isoformat(),
    'rows': 160, 'parts': parts, 'copied_unique_files': len(mapping),
    'source_delta_manifest': str(mapping[source_relative]['path'].relative_to(ROOT)),
    'source_delta_manifest_sha256': mapping[source_relative]['sha256'],
    'source_cohort': 'vllm_separate_cohort_not_numpy_draw_equivalence',
    'original_failed_sidecars_still_failed': True, 'new_GPU_regeneration_calls': 0}
with (target / 'received_manifest.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
with (target / 'received_source_verification.json').open('x') as stream:
    json.dump({'passed': True, 'actual_unique_keys': len(all_keys), 'parts': verification,
        'portable_manifest_sha256': hashlib.sha256(portable_path.read_bytes()).hexdigest(),
        'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'actual_files_sha_verified': len(mapping), 'original_sources_preserved': True,
        'producer_paths_written': 0}, stream, indent=2)
    stream.write('\n')
print(json.dumps({'rows': 160, 'unique_keys': len(all_keys), 'actual_files_sha_verified': len(mapping),
    'received_manifest': str((target / 'received_manifest.json').relative_to(ROOT))}), flush=True)
