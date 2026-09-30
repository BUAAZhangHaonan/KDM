"""Copy registered dependency bytes into an exclusive supplemental environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import tarfile


def bundle(output):
    output.mkdir(parents=True, exist_ok=False)
    packages = {'numpy': '2.2.4', 'Pillow': '11.1.0', 'scipy': '1.17.0', 'requests': '2.32.5'}
    files = {}
    for name, version in packages.items():
        dist = metadata.distribution(name)
        if dist.version != version:
            raise ValueError('Source dependency differs from its frozen version')
        base = Path(dist.locate_file('')).resolve()
        for item in dist.files:
            source = Path(dist.locate_file(item)).resolve()
            if not source.is_file() or not source.is_relative_to(base):
                continue
            relative = str(source.relative_to(base))
            if relative in files and files[relative] != source:
                raise ValueError('Dependency bundle member collision')
            files[relative] = source
    archive = output / 'registered_dependencies.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as target:
        for name, source in sorted(files.items()):
            target.add(source, arcname=name, recursive=False)
    receipt = {'packages': packages, 'files': len(files), 'archive': str(archive),
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'source': 'original registered central tf553 environment bytes',
               'created_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'bundle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def prepare(archive, receipt, target, base):
    record = json.loads(receipt.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != record['archive_sha256']:
        raise ValueError('Transferred dependency archive differs')
    if target.exists():
        raise FileExistsError('Preserve existing supplemental environment')
    subprocess.run([str(base / 'bin/python'), '-m', 'venv', str(target)], check=True)
    version = subprocess.check_output([str(target / 'bin/python'), '-c',
        'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    if version != 'python3.11':
        raise ValueError('Registered dependency bytes require Python 3.11')
    site = target / 'lib' / version / 'site-packages'
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('Dependency archive member escapes or is nonregular')
            destination = site / member.name
            if destination.exists():
                raise ValueError('Dependency member would overwrite an existing file')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, destination.open('xb') as output:
                output.write(stream.read())
    # Retain the already-verified torch/transformers versions from the local
    # tf553 environment; the four registered dependency copies take precedence.
    (site / 'remaining11_registered_base.pth').write_text(
        str(base / 'lib' / version / 'site-packages') + '\n')
    observed = json.loads(subprocess.check_output([str(target / 'bin/python'), '-c',
        'import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ["numpy","Pillow","scipy","requests","torch","transformers"]}))'], text=True))
    if any(observed[name] != value for name, value in record['packages'].items()):
        raise ValueError('Isolated dependencies do not match the source versions')
    saved = {'base': str(base), 'target': str(target), 'bundle': record,
             'observed': observed, 'prepared_utc': datetime.now(timezone.utc).isoformat()}
    (target / 'remaining11_preparation.json').write_text(json.dumps(saved, indent=2) + '\n')
    print(json.dumps(saved, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('bundle').add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'bundle':
        bundle(args.output)
    else:
        prepare(args.archive, args.receipt, args.target, args.base)


if __name__ == '__main__':
    main()
