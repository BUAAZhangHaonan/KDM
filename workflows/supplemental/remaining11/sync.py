"""Explicit Windows SSH/scp-3 synchronization of completed supplemental parts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CENTRAL = '/home/g203-4028/projects/knowledge-deficit-mitigation'
ALIASES = {'6403': '6403', '4029': '4029', 'k100': 'RTX_Pro_6000'}
TOOL = 'workflows/supplemental/remaining11/snapshot.py'


def ssh(alias, command):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', alias, shlex.join(command)],
                            text=True, encoding='utf-8', stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    return result.stdout


def scp(source, target):
    subprocess.run(['scp', '-3', '-o', 'BatchMode=yes', source, target], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=tuple(ALIASES), required=True)
    parser.add_argument('--run', default='run_20260930_140337')
    parser.add_argument('--snapshot-id')
    parser.add_argument('--max-parts', type=int)
    args = parser.parse_args()
    snapshot_id = args.snapshot_id or datetime.now(timezone.utc).strftime('snapshot_%Y%m%d_%H%M%S')
    registry = json.loads(ssh('4028-root', ['cat', CENTRAL + '/workflows/supplemental/remaining11/host_registry.json']))
    source_root = registry['hosts'][args.host]['root']
    alias = ALIASES[args.host]
    capture = json.loads(ssh(alias, ['python3', source_root + '/' + TOOL, 'capture',
                                    '--host', args.host, '--run', args.run, '--snapshot-id', snapshot_id]))
    local = ROOT / 'cache/remaining11_sync' / args.host / snapshot_id
    local.mkdir(parents=True, exist_ok=False)
    scp(alias + ':' + capture['snapshot_path'], str(local / 'host_snapshot.json'))
    snapshot = json.loads((local / 'host_snapshot.json').read_text(encoding='utf-8'))
    python = CENTRAL + '/.environments/mprisk-tf553/bin/python'
    known = set(json.loads(ssh('4028-root', [python, CENTRAL + '/' + TOOL, 'known', '--run', args.run]))['part_ids'])
    parts = [item for item in snapshot['completed_parts'] if item['part_id'] not in known]
    if args.max_parts is not None:
        if args.max_parts < 0:
            raise ValueError('A bounded transfer requires a nonnegative part limit; zero captures metadata only')
        parts = parts[:args.max_parts]
    transfer_path = local / 'transfer_manifest.json'
    transfer_path.write_text(json.dumps({'run': args.run, 'host': args.host,
                                         'part_ids': [item['part_id'] for item in parts]}, indent=2) + '\n', encoding='utf-8')
    incoming = CENTRAL + '/outputs/supplemental/remaining11/' + args.run + '/remote_completed/' + args.host + '/incoming/' + snapshot_id
    directories = {incoming}
    for item in parts:
        for kind in ('raw', 'identity', 'complete'):
            relative = PurePosixPath(item[kind + '_relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Snapshot has an escaping source path')
            directories.add(incoming + '/files/' + str(relative.parent))
    ssh('4028-root', ['mkdir', '-p', *sorted(directories)])
    scp(str(local / 'host_snapshot.json'), '4028-root:' + incoming + '/host_snapshot.json')
    scp(str(transfer_path), '4028-root:' + incoming + '/transfer_manifest.json')
    for item in parts:
        for kind in ('raw', 'identity', 'complete'):
            source = alias + ':' + item[kind + '_source_path']
            destination = '4028-root:' + incoming + '/files/' + item[kind + '_relative_path']
            scp(source, destination)
    received = json.loads(ssh('4028-root', [python, CENTRAL + '/' + TOOL, 'receive',
                                           '--run', args.run, '--incoming', incoming]))
    (local / 'sync_receipt.json').write_text(json.dumps(received, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'snapshot': capture, 'receive': received, 'local_receipt': str(local / 'sync_receipt.json')}, indent=2))


if __name__ == '__main__':
    main()
