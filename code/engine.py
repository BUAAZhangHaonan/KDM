"""Unified decoding engine for the hallucination-mitigation study.

Implements, inside ONE greedy decoding loop, with identical inputs/prompts:
  - direct   : standard greedy decoding
  - vcd      : Visual Contrastive Decoding (Leng et al., CVPR 2024, arXiv:2311.16922)
               distorted branch = diffusion-noised image (t=500, linear schedule)
               p = softmax((1+a)*logit_clean - a*logit_noise), plausibility beta
  - mib      : Multimodal Mutual-Information Decoding (arXiv:2403.14003)
               distorted branch = Gaussian-blurred image, coefficient decays linearly
               with decoding step: a_t = a_max * (1 - t/T)
  - lcd      : layer-contrastive decoding (DoLa-style, arXiv:2309.03883)
               logits' = (1+a)*logit_final - a*logit_early   (single forward)

Also computes two training-free signals (used by judgments 2/3):
  - entropy of the first-answer-token distribution (clean image, direct condition)
  - JSD between first-answer-token distributions (clean image vs blank image)

Speed notes (verified in recon): the Qwen3.5 vision patch_embed Conv3d hits a
pathological cuDNN path; since kernel == stride it is exactly a linear projection,
so we substitute an equivalent matmul (verified max|diff| = 0.008 = bf16 rounding).
"""
import os, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np

# ----------------------------- hyperparameters -----------------------------
HP = {
    'vcd_alpha': 1.0, 'vcd_beta': 0.1, 'vcd_noise_step': 500,
    'mib_alpha': 1.0, 'mib_blur_radius': 9.0,
    'lcd_alpha': 1.0, 'lcd_early_layer_frac': 0.5,
    'max_new_tokens_naming': 12,
    'max_new_tokens_exist': 4,
}


def diffuse_noise(img_t, step=500, total=1000):
    """DDPM forward-process noise, linear beta schedule (VCD-style distortion).
    img_t: float tensor (H,W,3) in [0,1]."""
    b0, b1 = 1e-4, 0.02
    betas = torch.linspace(b0, b1, total, dtype=torch.float64)
    alpha_bar = torch.cumprod(1 - betas, 0)[step]
    noise = torch.randn_like(img_t.float())
    x = alpha_bar.sqrt() * img_t.float() + (1 - alpha_bar).sqrt() * noise
    return x.clamp(0, 1)


def make_variants(pil):
    """Return dict of image variants used by the methods (PIL images)."""
    arr = torch.from_numpy(np.asarray(pil.convert('RGB')).copy()) / 255.0
    noisy = diffuse_noise(arr, HP['vcd_noise_step'])
    noisy_pil = Image.fromarray((noisy.numpy() * 255).astype('uint8'))
    blur_pil = pil.convert('RGB').filter(ImageFilter.GaussianBlur(HP['mib_blur_radius']))
    blank_pil = Image.new('RGB', (448, 448), (128, 128, 128))
    return {'clean': pil.convert('RGB'), 'noise': noisy_pil, 'blur': blur_pil, 'blank': blank_pil}


class ExpModel:
    def __init__(self, model_path, device):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        self._patch_fast_patch_embed()
        self.lm_head = self.model.lm_head
        self.eos = [self.model.config.text_config.eos_token_id]
        self.vocab = self.model.config.text_config.vocab_size
        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.lcd_layer = max(1, int(self.n_layers * HP['lcd_early_layer_frac']))
        self._token_cache = {}

    def _patch_fast_patch_embed(self):
        vis = self.model.model.visual
        pe = vis.patch_embed
        W = pe.proj.weight.data
        self._pe_W = W.reshape(W.shape[0], -1)
        self._pe_b = pe.proj.bias.data
        pe.proj = _FastConv3dAsLinear(self._pe_W, self._pe_b)

    def build(self, pil_img, text):
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': pil_img},
            {'type': 'text', 'text': text}]}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors='pt', enable_thinking=False)
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ---------------- single-branch greedy with optional LCD ----------------
    @torch.no_grad()
    def decode_single(self, inputs, max_new_tokens, lcd=False, seed_pos=None):
        """Greedy decode from given prefilled inputs. If lcd, adjust each step's
        logits with the early-layer contrast. Returns dict(text, tokens, first_probs,
        first_entropy, first_maxp, n_forwards, n_prefill_tokens, wall_s)."""
        t0 = time.time()
        n_fw = 0
        out = self.model(**inputs, output_hidden_states=lcd)
        n_fw += 1
        n_prefill = inputs['input_ids'].shape[1]
        logits = out.logits[:, -1, :].float()
        if lcd:
            early = self.lm_head(out.hidden_states[self.lcd_layer][:, -1, :]).float()
            logits = (1 + HP['lcd_alpha']) * logits - HP['lcd_alpha'] * early
        probs = F.softmax(logits, -1)
        first = {'entropy': float(-(probs * (probs + 1e-12).log()).sum()),
                 'maxp': float(probs.max())}
        first_probs = probs[0]
        tokens = []
        for t in range(max_new_tokens):
            tok = int(logits.argmax(-1))
            tokens.append(tok)
            if tok in self.eos:
                break
            out = self.model(input_ids=torch.tensor([[tok]], device=self.device),
                             past_key_values=out.past_key_values, output_hidden_states=lcd)
            n_fw += 1
            logits = out.logits[:, -1, :].float()
            if lcd:
                early = self.lm_head(out.hidden_states[self.lcd_layer][:, -1, :]).float()
                logits = (1 + HP['lcd_alpha']) * logits - HP['lcd_alpha'] * early
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'n_forwards': n_fw, 'n_prefill_tokens': n_prefill,
                'wall_s': time.time() - t0}

    # ---------------- two-branch contrastive greedy (VCD / MIB) ----------------
    @torch.no_grad()
    def decode_contrastive(self, inputs_main, inputs_dist, method, max_new_tokens):
        assert method in ('vcd', 'mib')
        t0 = time.time()
        a = HP['vcd_alpha'] if method == 'vcd' else HP['mib_alpha']
        beta = HP['vcd_beta']  # adaptive plausibility (VCD); MIB paper does not use it -> 0
        if method == 'mib':
            beta = 0.0
        out_a = self.model(**inputs_main)
        out_d = self.model(**inputs_dist)
        n_fw = 2
        logit_a = out_a.logits[:, -1, :].float()
        logit_d = out_d.logits[:, -1, :].float()
        probs_a = F.softmax(logit_a, -1)
        first = {'entropy': float(-(probs_a * (probs_a + 1e-12).log()).sum()),
                 'maxp': float(probs_a.max())}
        first_probs = probs_a[0]
        tokens = []
        T = max_new_tokens
        for t in range(max_new_tokens):
            if method == 'mib':
                a_t = a * (1 - t / max(T - 1, 1))
            else:
                a_t = a
            adj = (1 + a_t) * logit_a - a_t * logit_d
            if beta > 0:
                plaus = probs_a >= beta * probs_a.max()
                adj = adj.masked_fill(~plaus, float('-inf'))
            tok = int(adj.argmax(-1))
            tokens.append(tok)
            if tok in self.eos:
                break
            tk = torch.tensor([[tok]], device=self.device)
            out_a = self.model(input_ids=tk, past_key_values=out_a.past_key_values)
            out_d = self.model(input_ids=tk, past_key_values=out_d.past_key_values)
            n_fw += 2
            logit_a = out_a.logits[:, -1, :].float()
            logit_d = out_d.logits[:, -1, :].float()
            probs_a = F.softmax(logit_a, -1)
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'n_forwards': n_fw, 'n_prefill_tokens': inputs_main['input_ids'].shape[1],
                'wall_s': time.time() - t0}

    def blank_condition_inputs(self, text, blank_pil):
        return self.build(blank_pil, text)


class _FastConv3dAsLinear(torch.nn.Module):
    """Exact matmul replacement for the non-overlapping Conv3d patch projection.
    Exposes `.weight`/`.bias`-like attrs the wrapper code reads (dtype only)."""
    def __init__(self, W2d, bias):
        super().__init__()
        self.W = W2d
        self.b = bias

    @property
    def weight(self):
        return self.W

    @property
    def bias(self):
        return self.b

    def forward(self, x):
        # x: (N, C*kt*kh*kw) already flattened by caller's view, or 5D
        if x.dim() == 5:
            x = x.reshape(x.shape[0], -1)
        return x @ self.W.T + self.b


def jsd(p, q, base=2):
    m = 0.5 * (p + q)
    kl = lambda u, v: float((u * ((u / (v + 1e-12)).log())).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
