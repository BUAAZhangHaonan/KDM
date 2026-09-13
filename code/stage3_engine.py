"""Stage 3: family-agnostic decoding engine for cross-model replication.

Wraps any HF VLM (qwen3_5, qwen3_vl, llava_next, glm4v, internvl_chat remote)
with the same interface as v2's engine.ExpModel:
  build(pil, text) -> inputs
  decode_single(inputs, max_new_tokens, lcd=False)
  decode_contrastive(inputs_main, inputs_dist, method, max_new_tokens)
Recording semantics identical to v2 (answer_maxp = probability of the first
generated token under the distribution the config actually used).

Family notes:
  - chat template: enable_thinking kwarg only where supported (probed).
  - LCD/DoLa needs final-norm + lm_head module discovery; if either is not
    found, lcd=True raises and the caller must record lcd unavailable.
  - the Qwen3.5 Conv3d->matmul patch is applied only for model_type qwen3_5.
"""
import time
import torch
import torch.nn.functional as F

from engine import HP, jsd, make_variants, diffuse_noise  # reuse helpers

NORM_CHAINS = ['model.model.language_model.norm', 'model.language_model.norm',
               'language_model.model.norm', 'language_model.norm',
               'model.model.norm', 'model.norm']


class FamilyModel:
    def __init__(self, model_path, device):
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.mt = self.cfg.model_type
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, low_cpu_mem_usage=True).eval()

        tok = self.proc.tokenizer
        self.eos = set()
        tc = getattr(self.cfg, 'text_config', None)
        e = getattr(tc, 'eos_token_id', None) if tc is not None else getattr(self.cfg, 'eos_token_id', None)
        if isinstance(e, int):
            self.eos.add(e)
        elif e:
            self.eos.update(e)
        if tok.eos_token_id is not None:
            self.eos.add(tok.eos_token_id)

        if self.mt in ('qwen3_5', 'qwen3_vl'):
            self._patch_conv3d()
        self.lm_head = getattr(self.model, 'lm_head', None)
        self.norm = None
        for chain in NORM_CHAINS:
            obj = self.model
            for a in chain.split('.'):
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, 'weight'):
                self.norm, self.norm_path = obj, chain
                break
        self.lcd_ready = self.lm_head is not None and self.norm is not None

    # ---------------- qwen3.5 speed patch (from v2 engine) ----------------
    def _patch_conv3d(self):
        class _FastConv3dAsLinear(torch.nn.Module):
            def __init__(self, W, b):
                super().__init__()
                self.register_buffer('_W', W)
                self.register_buffer('_b', b)

            @property
            def weight(self):
                return self._W

            @property
            def bias(self):
                return self._b

            def forward(self, x):
                # Conv3d(kernel==stride) on pre-blocked input == linear on the
                # flattened block; works for 2-D (N, C*k^3) and N-D layouts.
                xf = x.reshape(x.shape[0], -1)
                return xf @ self._W.T + self._b

        pe = self.model.model.visual.patch_embed
        W = pe.proj.weight.data
        pe.proj = _FastConv3dAsLinear(W.reshape(W.shape[0], -1), pe.proj.bias.data)

    # ---------------- inputs ----------------
    def build(self, pil_img, text):
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': pil_img},
            {'type': 'text', 'text': text}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                  return_tensors='pt')
        if self.mt in ('qwen3_5', 'qwen3_vl', 'glm4v'):
            kw['enable_thinking'] = False
        try:
            inputs = self.proc.apply_chat_template(msgs, **kw)
        except TypeError:
            kw.pop('enable_thinking', None)
            inputs = self.proc.apply_chat_template(msgs, **kw)
        if not hasattr(inputs, 'items'):
            raise RuntimeError('apply_chat_template returned no mapping')
        return {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}

    # ---------------- decoding ----------------
    @torch.no_grad()
    def decode_single(self, inputs, max_new_tokens, lcd=False):
        if lcd and not self.lcd_ready:
            raise RuntimeError(f'lcd unavailable for model_type={self.mt}')
        t0 = time.time()
        n_fw = 0
        out = self.model(**inputs, output_hidden_states=lcd)
        n_fw += 1
        n_prefill = inputs['input_ids'].shape[1]
        logits = out.logits[:, -1, :].float()
        if lcd:
            logits = self._lcd_adjust(out)
        probs = F.softmax(logits, -1)
        first = {'entropy': float(-(probs * (probs + 1e-12).log()).sum()),
                 'maxp': float(probs.max())}
        first_probs = probs[0]
        tokens, answer_maxp = [], None
        for t in range(max_new_tokens):
            tok = int(logits.argmax(-1))
            if t == 0:
                answer_maxp = float(probs[0, tok])
            tokens.append(tok)
            if tok in self.eos:
                break
            out = self.model(input_ids=torch.tensor([[tok]], device=self.device),
                             past_key_values=out.past_key_values, output_hidden_states=lcd)
            n_fw += 1
            logits = out.logits[:, -1, :].float()
            if lcd:
                logits = self._lcd_adjust(out)
            probs = F.softmax(logits, -1)
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'answer_maxp': answer_maxp,
                'n_forwards': n_fw, 'n_prefill_tokens': n_prefill,
                'wall_s': time.time() - t0}

    def _lcd_adjust(self, out):
        hss = out.hidden_states
        L = len(hss) - 1
        final_logits = out.logits[:, -1, :].float()
        lp_final = F.log_softmax(final_logits, -1)
        logps = []
        with torch.no_grad():
            for l in range(1, L):
                h = self.norm(hss[l][:, -1, :])
                lg = self.lm_head(h).float()
                logps.append(F.log_softmax(lg, -1))
        if len(logps) < 2:
            return final_logits
        best_idx, best = 0, -1.0
        for i in range(len(logps) - 1):
            v = jsd(logps[i].exp(), logps[i + 1].exp())
            if v > best:
                best, best_idx = v, i
        return 2 * lp_final - logps[best_idx]

    @torch.no_grad()
    def decode_contrastive(self, inputs_main, inputs_dist, method, max_new_tokens):
        assert method in ('vcd', 'mib')
        t0 = time.time()
        a = HP['vcd_alpha'] if method == 'vcd' else HP['mib_alpha']
        beta = HP['vcd_beta'] if method == 'vcd' else 0.0
        out_a = self.model(**inputs_main)
        out_d = self.model(**inputs_dist)
        n_fw = 2
        logit_a = out_a.logits[:, -1, :].float()
        logit_d = out_d.logits[:, -1, :].float()
        probs_a = F.softmax(logit_a, -1)
        first = {'entropy': float(-(probs_a * (probs_a + 1e-12).log()).sum()),
                 'maxp': float(probs_a.max())}
        first_probs = probs_a[0]
        tokens, answer_maxp = [], None
        T = max_new_tokens
        for t in range(max_new_tokens):
            a_t = a * (1 - t / max(T - 1, 1)) if method == 'mib' else a
            adj = (1 + a_t) * logit_a - a_t * logit_d
            if beta > 0:
                plaus = probs_a >= beta * probs_a.max()
                adj = adj.masked_fill(~plaus, float('-inf'))
            p_adj = F.softmax(adj, -1)
            tok = int(adj.argmax(-1))
            if t == 0:
                answer_maxp = float(p_adj[0, tok])
            tokens.append(tok)
            if tok in self.eos:
                break
            out_a = self.model(input_ids=torch.tensor([[tok]], device=self.device),
                               past_key_values=out_a.past_key_values)
            out_d = self.model(input_ids=torch.tensor([[tok]], device=self.device),
                               past_key_values=out_d.past_key_values)
            n_fw += 2
            logit_a = out_a.logits[:, -1, :].float()
            logit_d = out_d.logits[:, -1, :].float()
            probs_a = F.softmax(logit_a, -1)
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'answer_maxp': answer_maxp,
                'n_forwards': n_fw, 'n_prefill_tokens': inputs_main['input_ids'].shape[1],
                'wall_s': time.time() - t0}


# --------------------- InternVL (internvl_chat remote code) ---------------------
# The shipped remote processor is incompatible with transformers 5.17
# (expects tokenizer.start_image_token). We bypass it: manual ChatML template,
# single 448x448 square crop via the repo's CLIPImageProcessor, image_flags=1,
# prefill through InternVLChatModel.forward and incremental steps directly
# through language_model (Qwen2) with its past_key_values.
class InternVLModel:
    def __init__(self, model_path, device):
        from types import SimpleNamespace
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import torch
        # transformers 5.17 expects PreTrainedModel.all_tied_weights_keys (dict);
        # the remote InternVLChatModel predates it. lm_head weights are stored
        # explicitly in the checkpoint, so an empty dict is safe here.
        _cls = get_class_from_dynamic_module(
            'modeling_internvl_chat.InternVLChatModel', model_path)
        _cls.all_tied_weights_keys = property(lambda self: {})
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.mt = self.cfg.model_type
        self.proc = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16,
            trust_remote_code=True).to(device).eval()
        self.ip = CLIPImageProcessor.from_pretrained(model_path)
        self.num_image_token = int(self.model.num_image_token)
        tok = self.proc.tokenizer
        self.img_ctx_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        # remote code sets this only inside chat(); required by forward()
        self.model.img_context_token_id = self.img_ctx_id
        self.eos = {tok.convert_tokens_to_ids('<|im_end|>')}
        if tok.eos_token_id is not None:
            self.eos.add(tok.eos_token_id)
        try:
            self.lm_head = self.model.language_model.lm_head
            self.norm = self.model.language_model.model.norm
            self.lcd_ready = True
        except AttributeError:
            self.lm_head, self.norm, self.lcd_ready = None, None, False

    def build(self, pil_img, text):
        import torch
        px = self.ip(images=pil_img.convert('RGB'), return_tensors='pt')['pixel_values']
        img_tokens = '<img>' + '<IMG_CONTEXT>' * self.num_image_token + '</img>'
        query = (f"<|im_start|>user\n{img_tokens}\n{text}<|im_end|>\n<|im_start|>assistant\n")
        enc = self.proc.tokenizer(query, return_tensors='pt')
        return {'input_ids': enc['input_ids'].to(self.device),
                'attention_mask': enc['attention_mask'].to(self.device),
                'pixel_values': px.to(self.device, torch.bfloat16),
                'image_flags': torch.ones((1, 1), dtype=torch.long, device=self.device)}

    def _prefill(self, inputs, ohs=False):
        return self.model(pixel_values=inputs['pixel_values'], input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'], image_flags=inputs['image_flags'],
                          output_hidden_states=ohs, use_cache=True)

    def _step(self, tok, pkv, ohs=False):
        return self.model.language_model(
            input_ids=torch.tensor([[tok]], device=self.device),
            past_key_values=pkv, output_hidden_states=ohs, use_cache=True)

    @torch.no_grad()
    def decode_single(self, inputs, max_new_tokens, lcd=False):
        if lcd and not self.lcd_ready:
            raise RuntimeError('lcd unavailable for internvl_chat')
        import torch
        t0 = time.time()
        n_fw = 0
        out = self._prefill(inputs, ohs=lcd)
        n_fw += 1
        n_prefill = inputs['input_ids'].shape[1]
        logits = out.logits[:, -1, :].float()
        if lcd:
            logits = self._lcd_adjust(out)
        probs = F.softmax(logits, -1)
        first = {'entropy': float(-(probs * (probs + 1e-12).log()).sum()),
                 'maxp': float(probs.max())}
        first_probs = probs[0]
        tokens, answer_maxp = [], None
        pkv = out.past_key_values
        for t in range(max_new_tokens):
            tok = int(logits.argmax(-1))
            if t == 0:
                answer_maxp = float(probs[0, tok])
            tokens.append(tok)
            if tok in self.eos:
                break
            out = self._step(tok, pkv, ohs=lcd)
            n_fw += 1
            logits = out.logits[:, -1, :].float()
            if lcd:
                logits = self._lcd_adjust(out)
            probs = F.softmax(logits, -1)
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'answer_maxp': answer_maxp, 'n_forwards': n_fw,
                'n_prefill_tokens': n_prefill, 'wall_s': time.time() - t0}

    def _lcd_adjust(self, out):
        hss = getattr(out, 'hidden_states', None)
        if not hss:
            raise RuntimeError('internvl_chat did not return hidden_states')
        L = len(hss) - 1
        final_logits = out.logits[:, -1, :].float()
        lp_final = F.log_softmax(final_logits, -1)
        logps = []
        with torch.no_grad():
            for l in range(1, L):
                h = self.norm(hss[l][:, -1, :])
                logps.append(F.log_softmax(self.lm_head(h).float(), -1))
        if len(logps) < 2:
            return final_logits
        best_idx, best = 0, -1.0
        for i in range(len(logps) - 1):
            v = float(jsd(logps[i].exp(), logps[i + 1].exp()))
            if v > best:
                best, best_idx = v, i
        return 2 * lp_final - logps[best_idx]

    @torch.no_grad()
    def decode_contrastive(self, inputs_main, inputs_dist, method, max_new_tokens):
        import torch
        assert method in ('vcd', 'mib')
        t0 = time.time()
        a = HP['vcd_alpha'] if method == 'vcd' else HP['mib_alpha']
        beta = HP['vcd_beta'] if method == 'vcd' else 0.0
        out_a = self._prefill(inputs_main)
        out_d = self._prefill(inputs_dist)
        n_fw = 2
        logit_a = out_a.logits[:, -1, :].float()
        logit_d = out_d.logits[:, -1, :].float()
        probs_a = F.softmax(logit_a, -1)
        first = {'entropy': float(-(probs_a * (probs_a + 1e-12).log()).sum()),
                 'maxp': float(probs_a.max())}
        first_probs = probs_a[0]
        tokens, answer_maxp = [], None
        T = max_new_tokens
        for t in range(max_new_tokens):
            a_t = a * (1 - t / max(T - 1, 1)) if method == 'mib' else a
            adj = (1 + a_t) * logit_a - a_t * logit_d
            if beta > 0:
                plaus = probs_a >= beta * probs_a.max()
                adj = adj.masked_fill(~plaus, float('-inf'))
            p_adj = F.softmax(adj, -1)
            tok = int(adj.argmax(-1))
            if t == 0:
                answer_maxp = float(p_adj[0, tok])
            tokens.append(tok)
            if tok in self.eos:
                break
            out_a = self._step(tok, out_a.past_key_values)
            out_d = self._step(tok, out_d.past_key_values)
            n_fw += 2
            logit_a = out_a.logits[:, -1, :].float()
            logit_d = out_d.logits[:, -1, :].float()
            probs_a = F.softmax(logit_a, -1)
        text = self.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'text': text, 'tokens': tokens, 'first_entropy': first['entropy'],
                'first_maxp': first['maxp'], 'first_probs': first_probs,
                'answer_maxp': answer_maxp, 'n_forwards': n_fw,
                'n_prefill_tokens': inputs_main['input_ids'].shape[1],
                'wall_s': time.time() - t0}


def get_engine(model_path, device):
    """Factory: internvl_chat uses the manual adapter; everything else FamilyModel."""
    from transformers import AutoConfig
    mt = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if mt == 'internvl_chat':
        return InternVLModel(model_path, device)
    return FamilyModel(model_path, device)
