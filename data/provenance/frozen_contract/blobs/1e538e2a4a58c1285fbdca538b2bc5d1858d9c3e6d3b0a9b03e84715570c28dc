#!/usr/bin/env python3
"""Run from the unpacked bundle; --apply requires a clean target tree."""
from pathlib import Path
import sys,argparse,json
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from kdm.migration import migrate
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--apply',action='store_true')
p.add_argument('--package',default=str(HERE));p.add_argument('--reviewed-source-change',action='store_true');a=p.parse_args()
print(json.dumps(migrate(a.project,a.package,a.apply,a.reviewed_source_change),indent=2))
