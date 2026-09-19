"""Unit check: ported SID selection vs the venv_sid official reference outputs
(PRE-REGISTER_STAGE7 sec 2.1: element-wise identical keep sets, tol 1e-9).

Usage: ./venv/bin/python code/stage7_sid_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from stage7_sid import sid_keep_indices  # noqa: E402

ok = True
ref = json.load(open(ROOT / 'outputs/raw/stage7_sid_reference.json'))
print('reference source:', ref['source'], '| sha256', ref['sha256'][:16],
      '| commit', ref['repo_commit'])
for i, c in enumerate(ref['cases']):
    att = np.load(ROOT / 'outputs/raw/stage7_sid_reference.npz')[f'case{i}']
    keep_port = sid_keep_indices(torch.from_numpy(att)[0],  # drop batch dim for the port
                                 c['sys_len'], c['img_len'], c['k']).tolist()
    same = keep_port == c['keep_local']
    # mask agreement over the visual span
    mask_ref = np.array(c['mask_image_span'], dtype=int).astype(bool)
    mask_port = np.zeros(c['img_len'], dtype=bool)
    mask_port[keep_port] = True
    same_mask = bool((mask_ref == mask_port).all())
    print(f'case {i}: keep set identical={same} | mask identical={same_mask}')
    ok = ok and same and same_mask
print('ALL PASS' if ok else 'MISMATCH - SID port fails the unit check')
sys.exit(0 if ok else 1)
