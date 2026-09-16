"""Stage-6 faithful decoding engine: VCD / M3ID / DoLa / DeCo per official sources.

Official bases (details in IMPLEMENTATION_MAPPING_STAGE6.md):
  VCD   DAMO-NLP-SG/VCD @ d6568ff  vcd_utils/vcd_sample.py + vcd_add_noise.py
  M3ID  arXiv:2403.14003 Algorithm 1 / Eq.(3)-(4) (no official code exists)
  DoLa  voidism/DoLa @ 805230e  transformers-4.28.1 generation/utils.py dola_greedy_decode
  DeCo  zjunlp/DeCo @ c1a9129  transformers/generation/utils.py deco_greedy_search
SID is blocked (fork transformers 4.29.2 vs project 5.17.0; mechanism needs attention
matrices at layers 0-2, which Qwen3.5's linear-attention layers do not produce).

All methods decode greedily (deterministic paired measurement; official VCD/DeCo
sample by default - recorded as deviation). answer_maxp / step_probs are taken
under the FINAL distribution the config actually used: after contrast, after any
truncation mask, renormalized. n_forwards counts transformer forwards only.
"""
import math
import time

import torch
import torch.nn.functional as F

from stage3_engine import get_engine

HP6 = {
    'vcd': {'alpha': 1.0, 'beta': 0.1, 'noise_step': 500},
    'm3id': {'lam': 0.02, 'alpha': 0.3},          # lambda=0.02, tau=0.3 (paper defaults)
    'dola': {'relative_top': 0.1},
    'deco': {'alpha': 0.6, 'top_p': 0.9, 'top_k': 20},
}
METHODS = ['direct', 'vcd', 'm3id', 'dola', 'deco']


def add_diffusion_noise_official(image_tensor, noise_step, gen=None):
    """Official VCD add_diffusion_noise: sigmoid beta schedule, applied to the
    processor's (normalized) pixel tensor, NOT clamped back to [0,1]."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device,
                           dtype=torch.float64)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    a_bar = alphas_prod[noise_step].sqrt()
    one_minus = (1 - alphas_prod[noise_step]).sqrt()
    noise = torch.randn(image_tensor.shape, device=image_tensor.device,
                        dtype=torch.float64, generator=gen)
    return a_bar * image_tensor.double() + one_minus * noise


def jsd_logp(lp1, lp2):
    """Jensen-Shannon divergence between two distributions given as log-probs."""
    p1, p2 = lp1.exp(), lp2.exp()
    m = 0.5 * (p1 + p2)
    lm = torch.log(m + 1e-12)
    kl1 = (p1 * (lp1 - lm)).sum()
    kl2 = (p2 * (lp2 - lm)).sum()
    v = float(0.5 * kl1 + 0.5 * kl2)
    return v if v == v else -1.0


class S6Model:
    """Wraps stage3_engine models with the four faithful stage-6 decoders."""

    def __init__(self, model_path, device, attn_implementation=None):
        self.em = get_engine(model_path, device,
                             attn_implementation=attn_implementation)
        self.model = self.em.model
        self.proc = self.em.proc
        self.device = self.em.device
        self.eos = self.em.eos
        self.lm_head = self.em.lm_head
        self.norm = self.em.norm
        self.tokenizer = self.proc.tokenizer
        cfg = getattr(self.em, 'cfg', None)
        tc = getattr(cfg, 'text_config', None) if cfg is not None else None
        self.n_layers = getattr(tc, 'num_hidden_layers', None) if tc is not None else None
        if self.n_layers is None:
            try:  # internvl_chat: layers live on the inner language model
                self.n_layers = self.model.language_model.config.num_hidden_layers
            except AttributeError:
                self.n_layers = self.model.config.num_hidden_layers
        # DeCo candidate layers: ceil(0.625L)..floor(0.875L) (official llama-7b
        # default range(20,29) == [20,28] at L=32).
        lo = math.ceil(0.625 * self.n_layers)
        hi = math.floor(0.875 * self.n_layers)
        self.deco_layers = list(range(lo, hi + 1))
        self._internvl = (getattr(self.em, 'mt', '') == 'internvl_chat')
        self._dola_candidates_override = None  # stage7: official-subset candidates

    # ---------------- forwards (InternVL routes through its adapter) --------
    def _fwd(self, inputs=None, tok=None, pkv=None, ohs=False, text_only=False,
             oa=False):
        if self._internvl:
            if inputs is not None:
                if text_only:  # M3ID prior branch: no pixel_values to inject
                    return self.model.language_model(
                        **inputs, output_hidden_states=ohs,
                        output_attentions=oa, use_cache=True)
                return self.em._prefill(inputs, ohs=ohs, oa=oa)
            return self.em._step(tok, pkv, ohs=ohs, oa=oa)
        if inputs is not None:
            return self.model(**inputs, output_hidden_states=ohs,
                              output_attentions=oa)
        return self.model(input_ids=torch.tensor([[tok]], device=self.device),
                          past_key_values=pkv, output_hidden_states=ohs,
                          output_attentions=oa)

    # ---------------- inputs ----------------
    def build(self, pil_img, text):
        return self.em.build(pil_img, text)

    def build_text_only(self, text):
        """M3ID prior branch: same prompt, visual conditioning dropped."""
        mt = getattr(self.em, 'mt', '')
        if mt == 'internvl_chat':
            query = (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
            enc = self.tokenizer(query, return_tensors='pt')
            return {'input_ids': enc['input_ids'].to(self.device),
                    'attention_mask': enc['attention_mask'].to(self.device)}
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                  return_tensors='pt')
        if mt in ('qwen3_5', 'qwen3_vl', 'glm4v'):
            kw['enable_thinking'] = False
        try:
            inputs = self.proc.apply_chat_template(msgs, **kw)
        except TypeError:
            kw.pop('enable_thinking', None)
            inputs = self.proc.apply_chat_template(msgs, **kw)
        return {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}

    def noised_inputs(self, inputs, seed, method='vcd', alpha=None):
        """VCD distorted branch: clone inputs, replace pixel_values with the
        officially scheduled diffusion noise on the processed pixel tensor."""
        out = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in inputs.items()}
        g = torch.Generator(device=inputs['pixel_values'].device)
        g.manual_seed(seed)
        noisy = add_diffusion_noise_official(
            inputs['pixel_values'].float(), HP6['vcd']['noise_step'], gen=g)
        out['pixel_values'] = noisy.to(inputs['pixel_values'].dtype)
        return out

    # ---------------- single-forward methods: direct / dola / deco ----------------
    @torch.no_grad()
    def decode_single(self, inputs, max_new_tokens, method='direct'):
        assert method in ('direct', 'dola', 'deco')
        t0 = time.time()
        n_fw = 0
        ohs = method in ('dola', 'deco')
        out = self._fwd(inputs=inputs, ohs=ohs)
        n_fw += 1
        n_prefill = inputs['input_ids'].shape[1]
        tokens, step_probs, layers_sel = [], [], []
        first = {}
        while len(tokens) < max_new_tokens:
            z = out.logits[:, -1, :].float()
            if method == 'dola':
                z, lyr = self._dola_adjust(out)
                layers_sel.append(lyr)
            elif method == 'deco':
                z, lyr = self._deco_adjust(out)
                layers_sel.append(lyr)
            p = F.softmax(z, -1)
            if not tokens:
                first = {'probs': p[0].clone(),
                         'entropy': float(-(p * (p + 1e-12).log()).sum()),
                         'maxp': float(p.max())}
            tok = int(z.argmax(-1))
            step_probs.append(float(p[0, tok]))
            tokens.append(tok)
            if tok in self.eos:
                break
            out = self._fwd(tok=tok, pkv=out.past_key_values, ohs=ohs)
            n_fw += 1
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        return self._record(text, tokens, step_probs, n_fw, n_prefill, t0,
                            layers_sel, method, first)

    def _dola_adjust(self, out):
        """Official dola_greedy_decode: candidates = hidden indices 0..L-1 (lm_head
        on RAW hidden state, no final norm); pick argmax JSD(final, candidate);
        z* = relative_top_filter(z_final, 0.1) - log_softmax(z_prem), base clamped
        to -1e3 at removed positions (official inf-inf guard)."""
        hss = out.hidden_states
        L = len(hss) - 1
        z_final = out.logits[:, -1, :].float()
        cutoff = math.log(HP6['dola']['relative_top']) + z_final.max(-1, keepdim=True).values
        remove = z_final < cutoff
        lp_final = F.log_softmax(z_final, -1)
        best_idx, best_jsd = 0, -1.0
        base = None
        # candidate premature hidden indices: all of 0..L-1, or the official-subset
        # override (even layers) set by the stage-7 runner
        cands = self._dola_candidates_override \
            if getattr(self, '_dola_candidates_override', None) is not None \
            else range(L)
        for l in cands:
            lg = self.lm_head(hss[l][:, -1, :]).float()
            lp = F.log_softmax(lg, -1)
            v = jsd_logp(lp_final, lp)
            if v > best_jsd:
                best_jsd, best_idx, base = v, l, lp.clone()
        base[0][remove[0]] = -1e3
        z_star = z_final.masked_fill(remove, float('-inf')) - base
        return z_star, best_idx

    def _deco_adjust(self, out):
        """Official deco_greedy_search: final-dist candidates = top-k truncated at
        cumulative top-p; pick (layer, token) with max early-layer prob among
        candidate layers (lm_head AFTER final norm - DeCo applies it, DoLa does
        not); z* = z_final + alpha * maxprob * z_sel, then keep candidates only."""
        hss = out.hidden_states
        z_final = out.logits[:, -1, :].float()
        p_final = F.softmax(z_final, -1)[0]
        k = min(HP6['deco']['top_k'], p_final.shape[-1])
        cand_p, cand_ids = torch.topk(p_final, k)
        cut = int(torch.searchsorted(cand_p.cumsum(0),
                  torch.tensor(HP6['deco']['top_p'],
                  device=cand_p.device), right=False).item()) + 1
        cut = min(cut, k)
        cand_ids = cand_ids[:cut]
        zs = []
        for l in self.deco_layers:
            h = self.norm(hss[l][:, -1, :])
            zs.append(self.lm_head(h).float())
        cand_probs = torch.stack([F.softmax(z, -1)[0, cand_ids] for z in zs])
        flat = int(cand_probs.argmax())
        li, ti = flat // cut, flat % cut
        maxprob = float(cand_probs[li, ti])
        z_star = z_final + HP6['deco']['alpha'] * maxprob * zs[li]
        keep = torch.ones_like(z_star, dtype=torch.bool)
        keep[0, cand_ids] = False
        z_star = z_star.masked_fill(keep, float('-inf'))
        return z_star, self.deco_layers[li]

    # ---------------- two-branch methods: vcd / m3id ----------------
    @torch.no_grad()
    def decode_two_branch(self, inputs_main, inputs_ref, method, max_new_tokens,
                          t0_sched=0, alpha_override=None):
        assert method in ('vcd', 'm3id')
        t0 = time.time()
        out_a = self._fwd(inputs=inputs_main)
        out_b = self._fwd(inputs=inputs_ref,
                          text_only=(method == 'm3id' and self._internvl))
        n_fw = 2
        n_prefill = inputs_main['input_ids'].shape[1]
        tokens, step_probs, first = [], [], {}
        if method == 'vcd':
            a = HP6['vcd']['alpha'] if alpha_override is None else alpha_override
            beta = HP6['vcd']['beta']
        lam = HP6['m3id']['lam']
        tau = HP6['m3id']['alpha']
        for t in range(max_new_tokens):
            z_a = out_a.logits[:, -1, :].float()
            z_b = out_b.logits[:, -1, :].float()
            if method == 'vcd':
                z_star = (1 + a) * z_a - a * z_b
                cutoff = math.log(beta) + z_a.max(-1, keepdim=True).values
                z_star = z_star.masked_fill(z_a < cutoff, float('-inf'))
            else:
                lp_a = F.log_softmax(z_a, -1)
                lp_b = F.log_softmax(z_b, -1)
                gate = float(lp_a.exp().max()) < tau
                gamma = math.exp(-lam * (t + t0_sched))
                w = (1 - gamma) / gamma
                z_star = lp_a + (w * (lp_a - lp_b) if gate else 0.0)
            p = F.softmax(z_star, -1)
            if not tokens:
                first = {'probs': p[0].clone(),
                         'entropy': float(-(p * (p + 1e-12).log()).sum()),
                         'maxp': float(p.max())}
            tok = int(z_star.argmax(-1))
            step_probs.append(float(p[0, tok]))
            tokens.append(tok)
            if tok in self.eos:
                break
            out_a = self._fwd(tok=tok, pkv=out_a.past_key_values)
            out_b = self._fwd(tok=tok, pkv=out_b.past_key_values)
            n_fw += 2
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        return self._record(text, tokens, step_probs, n_fw, n_prefill, t0,
                            [], method, first)

    def _record(self, text, tokens, step_probs, n_fw, n_prefill, t0w, layers_sel,
                method, first):
        return {'text': text, 'tokens': tokens, 'step_probs': step_probs,
                'answer_maxp': step_probs[0] if step_probs else None,
                'first_entropy': first.get('entropy'),
                'first_maxp': first.get('maxp'),
                'first_probs': first.get('probs'),
                'n_forwards': n_fw, 'n_prefill_tokens': n_prefill,
                'wall_s': round(time.time() - t0w, 3),
                'layers_selected': layers_sel, 'method': method}
