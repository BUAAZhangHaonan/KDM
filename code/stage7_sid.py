"""Stage-7 SID: ported introspective branch + unit check + runner.

Faithful reimplementation of the official SID selection logic (huofushuo/SID
@ 127dd41, sid/transformers modeling_llama.py 'FastV Token Rerank, Attention
Mask Implementation'): at decoder layer index 2, rank visual tokens by the
previous layer's attention received from the LAST token (head-averaged), keep
the 100 LEAST attended, mask the rest for layers >= 2. Combination formula is
identical to VCD (alpha=1, beta=0.1 plausibility truncation on the clean
branch, greedy) per the official SID repo (vcd_sample.py).

Only models with attention matrices at layers 0-1 are eligible (llava16,
internvl4b); the two Qwen3.5 models are architecturally excluded (first three
layers are linear attention).
All results are labeled 'faithful reimplementation from official code'.
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))

SID_HP = {'agg_layer': 2, 'attention_rank': 100, 'alpha': 1.0, 'beta': 0.1}


def sid_keep_indices(att_last_layer, sys_len, img_len, k=SID_HP['attention_rank']):
    """Ported official selection: input = last layer's attention
    (heads, q, kv); uses the last query token's head-averaged row over the
    visual span; returns sorted local indices of the kept (least attended)
    visual tokens."""
    avg = att_last_layer.mean(dim=0)             # (q, kv) head-averaged
    row = avg[-1]                                # last query token
    img_att = row[sys_len:sys_len + img_len]
    return img_att.topk(min(k, img_len), largest=False).indices.sort().values


def build_keep_mask(kv_len, sys_len, img_len, keep_local, device, dtype,
                    q_len=1):
    """Additive 4D mask (1,1,q,kv): 0 where allowed, -inf where blocked.
    Non-visual columns always allowed; visual columns only if kept. Causal
    structure for q>1 is handled by the model's own causal logic when q_len==1;
    for full-sequence passes causal is combined here."""
    mask = torch.zeros(1, 1, q_len, kv_len, device=device, dtype=dtype)
    blocked = torch.ones(kv_len, dtype=torch.bool, device=device)
    blocked[:sys_len] = False
    blocked[sys_len + img_len:] = False
    blocked[sys_len + keep_local] = False
    mask[..., :].masked_fill_(blocked.view(1, 1, 1, -1), float('-inf'))
    if q_len > 1:  # combine with causality for full-sequence passes
        causal = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool,
                                       device=device))
        mask = mask.masked_fill(~causal.view(1, 1, q_len, kv_len), float('-inf'))
    return mask


class SidDecoder:
    """Wraps S6Model with the SID two-branch greedy decoder."""

    def __init__(self, s6):
        self.s6 = s6
        self.em = s6.em
        self.model = s6.model
        self._hooks = []
        self._flag = {'active': False}
        self._captured = None
        layers = self._decoder_layers()
        for idx, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_pre_hook(
                self._make_hook(idx), with_kwargs=True))
        # capture layer-(agg_layer-1) attention weights: eager attention always
        # returns them; model-level output_attentions stays OFF so no layer's
        # weights are retained beyond this stash
        attn1 = layers[SID_HP['agg_layer'] - 1].self_attn
        orig_fwd = attn1.forward

        cfg_ref = attn1.config

        def capture(*a, **kw):
            prev = cfg_ref._attn_implementation
            cfg_ref._attn_implementation = 'eager'  # eager for THIS layer only
            try:
                out = orig_fwd(*a, **kw)
            finally:
                cfg_ref._attn_implementation = prev
            w = out[1]
            if w is not None:
                self._captured = w.detach()[0]  # (heads, q, kv)
            return out
        attn1.forward = capture

    def _decoder_layers(self):
        m = self.model
        for chain in ('language_model.model.layers', 'model.language_model.layers',
                      'model.layers'):
            obj = m
            ok = True
            for a in chain.split('.'):
                obj = getattr(obj, a, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                return list(obj)
        raise RuntimeError('decoder layers not found')

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self._flag['active'] or idx < SID_HP['agg_layer']:
                return args, kwargs
            kwargs['attention_mask'] = self._current_mask
            return args, kwargs
        return hook

    def image_span(self, inputs, model_type):
        ids = inputs['input_ids'][0]
        if model_type == 'internvl_chat':
            tok = self.s6.em.img_context_token_id
        else:  # llava_next
            tok = getattr(self.model.config, 'image_token_id', None)
            if tok is None:
                tok = self.s6.tokenizer.convert_tokens_to_ids('<image>')
        pos = (ids == tok).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            return None
        return int(pos.min().item()), int(pos.max().item()) + 1  # [start, end)

    @torch.no_grad()
    def decode(self, inputs, max_new_tokens):
        mt = getattr(self.em, 'mt', '')
        span = self.image_span(inputs, mt)
        if span is None:
            raise RuntimeError('no visual tokens found in input')
        sys_len, img_len = span
        t0 = time.time()
        # main (clean) branch prefill; the layer-1 capture stash fills itself
        out_m = self.s6._fwd(inputs=inputs)
        att1 = self._captured                              # (heads, q, kv)
        kv_len = att1.shape[-1]
        keep = sid_keep_indices(att1, sys_len, img_len)
        self._current_mask = build_keep_mask(
            kv_len, sys_len, img_len, keep, inputs['input_ids'].device,
            next(self.model.parameters()).dtype, q_len=1)
        # introspective branch prefill (same inputs, masked layers >= 2)
        self._flag['active'] = True
        out_s = self.s6._fwd(inputs=inputs)
        self._flag['active'] = False
        n_fw = 2
        tokens, step_probs = [], []
        first = {}
        while len(tokens) < max_new_tokens:
            z_m = out_m.logits[:, -1, :].float()
            z_s = out_s.logits[:, -1, :].float()
            z = (1 + SID_HP['alpha']) * z_m - SID_HP['alpha'] * z_s
            cutoff = math.log(SID_HP['beta']) + z_m.max(-1, keepdim=True).values
            z = z.masked_fill(z_m < cutoff, float('-inf'))
            p = F.softmax(z, -1)
            tok = int(z.argmax(-1))
            if not tokens:
                first = {'entropy': float(-(p * (p + 1e-12).log()).sum()),
                         'maxp': float(p.max())}
            step_probs.append(float(p[0, tok]))
            tokens.append(tok)
            if tok in self.s6.eos:
                break
            tk = torch.tensor([[tok]], device=inputs['input_ids'].device)
            out_m = self.s6._fwd(tok=tok, pkv=out_m.past_key_values)
            att1 = self._captured
            keep = sid_keep_indices(att1, sys_len, img_len)
            kv_len = att1.shape[-1]
            self._current_mask = build_keep_mask(
                kv_len, sys_len, img_len, keep, tk.device,
                next(self.model.parameters()).dtype, q_len=1)
            self._flag['active'] = True
            out_s = self.s6._fwd(tok=tok, pkv=out_s.past_key_values)
            self._flag['active'] = False
            n_fw += 2
        text = self.s6.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        return {'text': text, 'tokens': tokens, 'step_probs': step_probs,
                'answer_maxp': step_probs[0] if step_probs else None,
                'first_entropy': first.get('entropy'),
                'first_maxp': first.get('maxp'),
                'n_forwards': n_fw,
                'n_prefill_tokens': int(inputs['input_ids'].shape[1]),
                'wall_s': round(time.time() - t0, 3),
                'layers_selected': [SID_HP['agg_layer']],
                'method': 'sid'}

    def close(self):
        for h in self._hooks:
            h.remove()
