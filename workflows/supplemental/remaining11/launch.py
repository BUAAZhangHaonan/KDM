"""Run a bounded, explicit host queue; every dispatch checks inputs before Popen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    jobs = queue['jobs']
    if queue.get('schema') != 'kdm_remaining11_bounded_launch_v1' or not 1 <= len(jobs) <= 4:
        raise ValueError('Launch requires one through four explicitly registered jobs')
    for job in jobs:
        command = [sys.executable, str(ROOT / 'workflows/supplemental/remaining11/dispatch.py')]
        for key, value in job.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
