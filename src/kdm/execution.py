"""Registered host admission and content-preserving local image resolution."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import socket
import subprocess
from .io import file_hash, within

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'configs/runtime/hosts.json'


def read_registry(root):
    data = json.loads((Path(root) / REGISTRY).read_text())
    if data.get('schema') != 1 or not data.get('hosts') or not data.get('model_hosts'):
        raise ValueError('Invalid execution host registry')
    for name, host in data['hosts'].items():
        cards = host.get('allowed_gpus', [])
        if not Path(host['root']).is_absolute() or not host.get('hostname') or not cards or len(cards) != len(set(cards)):
            raise ValueError('Invalid registered host/root/GPU declaration')
        if set(host.get('gpu_uuids', {})) != {str(c) for c in cards}:
            raise ValueError('Every authorized physical GPU needs a registered UUID')
    if not set(data['model_hosts'].values()) <= set(data['hosts']):
        raise ValueError('Unknown assigned execution host')
    for source, destination in data.get('image_prefixes', {}).items():
        if not Path(source).is_absolute() or Path(destination).is_absolute() or '..' in Path(destination).parts:
            raise ValueError('Invalid image path prefix mapping')
    return data


def local_host(root, expected=None):
    root = Path(root).resolve()
    registry = read_registry(root)
    matches = [(key, host) for key, host in registry['hosts'].items()
               if host['hostname'] == socket.gethostname() and Path(host['root']) == root]
    if len(matches) != 1 or (expected is not None and matches[0][0] != expected):
        raise ValueError('Execution hostname/project root does not match the registered host')
    return matches[0]


def validate_host(root, cards, model=None, expected=None):
    name, host = local_host(root, expected)
    cards = [str(c) for c in cards]
    if not cards or len(cards) != len(set(cards)) or not set(cards) <= {str(c) for c in host['allowed_gpus']}:
        raise ValueError('Unauthorized or duplicate physical GPU on this host')
    if model is not None and read_registry(root)['model_hosts'].get(model) != name:
        raise ValueError('Model is assigned to a different execution host')
    rows = subprocess.check_output(['nvidia-smi', '-i', ','.join(cards),
        '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
    observed = dict(tuple(field.strip() for field in row.split(',')) for row in rows)
    if observed != {c: host['gpu_uuids'][c] for c in cards}:
        raise ValueError('Physical GPU UUID differs from the registered host identity')
    return name, host


def execution_receipt(root, cards, model):
    name, host = validate_host(root, cards, model)
    data = read_registry(root)
    receipt = {'host': name, 'hostname': host['hostname'], 'project_root': host['root'],
        'physical_gpus': [str(c) for c in cards],
        'gpu_uuids': {str(c): host['gpu_uuids'][str(c)] for c in cards},
        'registry_sha256': file_hash(Path(root) / REGISTRY),
        'resolver_sha256': file_hash(Path(root) / 'src/kdm/execution.py')}
    if name == '6403':
        receipt['image_catalog_sha256'] = file_hash(within(root, data['image_catalog']))
    return receipt


def validate_execution_receipt(root, receipt, model, gpu_count=None):
    """Portable evidence check: never query the consumer host's GPUs or weights."""
    data = read_registry(root)
    name = data['model_hosts'].get(model)
    host = data['hosts'][name]
    if not isinstance(receipt, dict):
        raise ValueError('Missing registered execution host receipt')
    cards = receipt.get('physical_gpus', [])
    if not cards or len(cards) != len(set(cards)) or not set(cards) <= {str(c) for c in host['allowed_gpus']}:
        raise ValueError('Evidence contains unauthorized physical GPUs')
    if gpu_count is not None and len(cards) != gpu_count:
        raise ValueError('Evidence physical GPU count differs from model spec')
    expected = {'host': name, 'hostname': host['hostname'], 'project_root': host['root'],
        'physical_gpus': cards, 'gpu_uuids': {c: host['gpu_uuids'][c] for c in cards},
        'registry_sha256': file_hash(Path(root) / REGISTRY),
        'resolver_sha256': file_hash(Path(root) / 'src/kdm/execution.py')}
    if name == '6403':
        expected['image_catalog_sha256'] = file_hash(within(root, data['image_catalog']))
    if receipt != expected:
        raise ValueError('Execution host/path resolver identity changed')


@lru_cache(maxsize=16)
def _cached_hash(path, signature):
    return file_hash(path)


def unchanged_file_hash(path):
    path=Path(path);s=path.stat()
    return _cached_hash(str(path),(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))


@lru_cache(maxsize=4)
def _catalog(path, signature):
    return json.loads(Path(path).read_text())


_verified_images = {}


def resolve_image_path(logical, root=None):
    """Keep manifest/sample paths untouched; map only at the local file-open edge."""
    root = Path(root or ROOT).resolve()
    original = Path(logical)
    # Explicit project-contained software fixtures do not represent formal inputs.
    if original.is_absolute() and any(original.resolve().is_relative_to(root / part)
            for part in ('outputs/verification', 'cache/pytest')):
        return original.resolve()
    data = read_registry(root)
    name, _ = local_host(root)
    matches = [(source, destination) for source, destination in data['image_prefixes'].items()
               if original.is_relative_to(Path(source))]
    if len(matches) != 1 or '..' in original.parts:
        raise ValueError('Image is outside the registered logical source prefixes')
    source, destination = matches[0]
    target_dir = within(root, destination)
    target = (target_dir / original.relative_to(source)).resolve()
    if not target.is_relative_to(target_dir) or not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Mapped image is missing or escapes the registered project image directory')
    if name == '6403':
        catalog_path = within(root, data['image_catalog'])
        stat = catalog_path.stat()
        catalog = _catalog(str(catalog_path), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        if catalog.get('schema') != 1 or catalog.get('manifest_sha256') != unchanged_file_hash(root / 'data/current/all.jsonl'):
            raise ValueError('Image content catalog differs from the full original manifest')
        expected = catalog.get('images', {}).get(str(original))
        if not isinstance(expected, dict):
            raise ValueError('Mapped image lacks original content identity')
        stat = target.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cache_key = (str(target), expected.get('sha256'), expected.get('size_bytes'))
        if _verified_images.get(cache_key) != signature:
            if stat.st_size != expected.get('size_bytes') or file_hash(target) != expected.get('sha256'):
                raise ValueError('Mapped image content differs from the original input')
            after = target.stat()
            if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Mapped image changed during content validation')
            _verified_images[cache_key] = signature
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cards', required=True)
    args = parser.parse_args()
    validate_host(args.root, args.cards.split(','))
