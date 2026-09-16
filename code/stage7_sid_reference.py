"""Generate SID official-selection reference outputs inside venv_sid.

Runs the VERBATIM selection lines extracted from the official repository
(huofushuo/SID @ 127dd41,
 sid/transformers/src/transformers/models/llama/modeling_llama.py,
 'FastV Token Rerank, Attention Mask Implementation' branch, around L743-790)
on random attention tensors and stores the kept visual-token indices.

The source-file sha256 is recorded to prove the extraction target is unchanged.
Run with: ./venv_sid/bin/python code/stage7_sid_reference.py
"""
import hashlib
import json

import numpy as np
import torch

SRC = ('reference_repos/sid/transformers/src/transformers/models/'
       'llama/modeling_llama.py')


def official_selection(last_layer_attention, SYS_LENGTH, IMAGE_TOKEN_LENGTH,
                       ATTENTION_RANK, batch_size, seq_length_with_past, device):
    # ---- verbatim lines from the official file (mask implementation) ----
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    last_layer_attention_avg_last_tok_image = \
        last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH + IMAGE_TOKEN_LENGTH]
    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(
        ATTENTION_RANK, largest=False).indices + SYS_LENGTH
    gen_attention_mask = torch.ones(
        (batch_size, seq_length_with_past), dtype=torch.bool, device=device)
    gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH + IMAGE_TOKEN_LENGTH] = False
    gen_attention_mask[:, top_attention_rank_index] = True
    # ---- end verbatim ----
    keep_local = (top_attention_rank_index - SYS_LENGTH).sort().values
    return keep_local, gen_attention_mask


def main():
    sha = hashlib.sha256(open(SRC, 'rb').read()).hexdigest()
    torch.manual_seed(7)
    rng = np.random.RandomState(7)
    cases = []
    for ci in range(12):
        heads = int(rng.choice([16, 32]))
        sys_len = int(rng.choice([10, 40, 200]))
        img_len = int(rng.choice([256, 576, 1024]))
        suffix = int(rng.choice([20, 60]))
        k = int(rng.choice([100]))
        q, kv = 1, sys_len + img_len + suffix
        att = torch.rand(1, heads, q, kv)  # official layout: (batch, heads, q, kv)
        keep_local, mask = official_selection(
            att, sys_len, img_len, k, 1, kv, 'cpu')
        cases.append({'heads': heads, 'sys_len': sys_len, 'img_len': img_len,
                      'suffix': suffix, 'k': k,
                      'att': att.numpy().astype(np.float32),
                      'keep_local': keep_local.numpy(),
                      'mask_image_span': mask[0, sys_len:sys_len + img_len].numpy()})
    np.savez('outputs/raw/stage7_sid_reference.npz',
             meta=json.dumps({'source': SRC, 'sha256': sha,
                              'repo_commit': '127dd41'}),
             **{f'case{i}': c['att'] for i, c in enumerate(cases)})
    meta_out = [{'keep_local': c['keep_local'].tolist(),
                 'mask_image_span': c['mask_image_span'].astype(int).tolist(),
                 **{kk: c[kk] for kk in ('heads', 'sys_len', 'img_len', 'suffix', 'k')}}
                for c in cases]
    json.dump({'source': SRC, 'sha256': sha, 'repo_commit': '127dd41',
               'cases': meta_out},
              open('outputs/raw/stage7_sid_reference.json', 'w'))
    print('reference saved; source sha256', sha[:16], '| cases', len(cases))


if __name__ == '__main__':
    main()
