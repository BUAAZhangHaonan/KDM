"""Fixed SID reference: official 127dd41, mask mode, aggregation index 2/rank 100.

Each forward selects from its own second block attention, then restores all hooks.
Only expanded, contiguous visual tokens and full causal KV caches are supported.
"""
from __future__ import annotations
from contextlib import contextmanager
import inspect
from .hf import HFSession


def keep_visual_indices(attention, start, length, k=100):
    if length < k or start < 0 or start + length > attention.shape[-1]:
        raise ValueError('SID fixed rank 100 requires a valid expanded visual span with at least 100 tokens')
    row = attention.mean(dim=0)[-1, start:start+length]
    return row.topk(k, largest=False).indices.sort().values


def restricted_attention_mask(torch, original, hidden, kv_length, start, length, keep):
    qlen = hidden.shape[1]
    blocked = torch.zeros(kv_length, dtype=torch.bool, device=hidden.device)
    blocked[start:start+length] = True
    blocked[start+keep.to(hidden.device)] = False
    if original is not None:
        if (original.ndim != 4 or original.shape[-1] != kv_length
                or original.shape[-2] != qlen or not original.is_floating_point()):
            raise ValueError('SID requires an additive floating 4D full-KV causal mask')
        return original.masked_fill(blocked[None, None, None, :], float('-inf'))
    past = kv_length - qlen
    if past < 0:
        raise ValueError('Invalid attention lengths')
    query = torch.arange(qlen, device=hidden.device)[:, None] + past
    key = torch.arange(kv_length, device=hidden.device)
    valid = key <= query
    return torch.zeros((1, 1, qlen, kv_length), device=hidden.device, dtype=hidden.dtype).masked_fill(
        (~valid)[None, None, :, :] | blocked[None, None, None, :], float('-inf'))


class SIDControl:
    def __init__(self, backend, inputs):
        self.b = backend
        self.audit = None
        self.layers = None
        for path in ('language_model.model.layers', 'model.language_model.layers', 'model.layers'):
            obj = backend.model
            for name in path.split('.'):
                obj = getattr(obj, name, None)
            if obj is not None:
                self.layers = list(obj)
                self.layer_path = path
                break
        if self.layers is None or len(self.layers) < 3:
            raise RuntimeError('SID needs at least three decoder layers at a registered module path')
        ids = inputs.get('input_ids')
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise RuntimeError('SID supports exactly one explicit input_ids sequence')
        tok = getattr(backend.em, 'img_ctx_id', None)
        if tok is None:
            tok = getattr(backend.model.config, 'image_token_id', None)
        if getattr(backend.model.config, 'model_type', None) == 'phi3_v':
            # Native Phi3ImageEmbedding.forward indexes exactly these expanded
            # positions before clamping ids and writing projected visual vectors.
            positions = ((ids[0] < 0) & (ids[0] > -int(1e9))).nonzero(as_tuple=True)[0]
        else:
            if tok is None:
                raise RuntimeError('SID visual mapping unavailable: no explicit expanded image token id')
            positions = (ids[0] == tok).nonzero(as_tuple=True)[0]
        if not len(positions):
            raise RuntimeError('SID visual mapping unavailable: no image token positions')
        self.start = int(positions.min())
        self.length = int(positions.max()) - self.start + 1
        if self.length != len(positions):
            raise RuntimeError('SID visual mapping unsupported: image span is not contiguous')
        if self.length < 100:
            raise RuntimeError('SID fixed rank 100 requires at least 100 expanded image token positions; placeholder/Q-former mapping is unsupported')
        if self.start == 0:
            raise RuntimeError('SID requires a preceding text/BOS token to avoid fully masked causal rows')
        mask = inputs.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or not bool(mask.all())):
            raise RuntimeError('SID fixed official mask comparison requires an unpadded single sequence')
        self.attn_module = self.layers[1].self_attn
        if not hasattr(self.attn_module, 'config'):
            raise RuntimeError('SID second-layer attention has no configurable eager implementation')

    def _mask(self, original, hidden, attention):
        keep = keep_visual_indices(attention, self.start, self.length)
        return keep, restricted_attention_mask(self.b.torch, original, hidden,
                                                attention.shape[-1], self.start, self.length, keep)

    @contextmanager
    def forward_scope(self, qlen, kv_length):
        if getattr(self.b, '_sid_active_control', None) is not None:
            raise RuntimeError('SID forwards must be serialized on the shared backend')
        self.b._sid_active_control = self
        module = self.attn_module
        original_forward = module.forward
        had_override = 'forward' in module.__dict__
        old_override = module.__dict__.get('forward')
        hooks = []
        downstream_overrides = []
        attention = None
        selected = None
        # Legacy SDPA attention only delegates to eager when output_attentions=True.
        wants_output = 'output_attentions' in inspect.signature(original_forward).parameters
        def capture(*args, **kwargs):
            nonlocal attention
            cfg = module.config
            previous = cfg._attn_implementation
            cfg._attn_implementation = 'eager'
            if wants_output:
                kwargs['output_attentions'] = True
            try:
                output = original_forward(*args, **kwargs)
            finally:
                cfg._attn_implementation = previous
            if len(output) < 2 or output[1] is None:
                raise RuntimeError('SID second-layer eager attention did not return attention weights')
            weights = output[1]
            if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[-2:] != (qlen, kv_length):
                raise RuntimeError('SID attention axes do not match expanded input_ids/full KV cache')
            attention = weights.detach()[0]
            return output
        def hook(layer_index):
            def apply(module, args, kwargs):
                nonlocal selected
                if attention is None:
                    raise RuntimeError('SID reference second-layer attention was not captured in this forward')
                hidden = args[0] if args else kwargs['hidden_states']
                if hidden.shape[1] != qlen:
                    raise RuntimeError('SID hidden query length changed after visual mapping')
                original = kwargs.get('attention_mask')
                if original is None and getattr(self.attn_module.config, 'sliding_window', None):
                    raise RuntimeError('SID cannot reconstruct an implicit sliding-window attention mask')
                keep, mask = self._mask(original, hidden, attention)
                if selected is None:
                    selected = keep
                elif not self.b.torch.equal(selected, keep):
                    raise RuntimeError('SID selection changed within one forward')
                kwargs['attention_mask'] = mask
                if self.audit is not None:
                    self.audit({'layer': layer_index, 'query_length': qlen, 'kv_length': kv_length,
                                'start': self.start, 'length': self.length,
                                'attention': attention, 'selected': keep, 'original_mask': original,
                                'mask': mask})
                return args, kwargs
            return apply
        try:
            module.forward = capture
            for index, layer in enumerate(self.layers[2:], 2):
                hooks.append(layer.register_forward_pre_hook(hook(index), with_kwargs=True))
                attn = layer.self_attn
                # FA2's padding/unpadding API consumes a 2D padding mask, not
                # SID's query-specific 4D additive causal mask. Eager implements
                # that exact mask; restore dispatch immediately after the call.
                if getattr(attn.config, '_attn_implementation', None) == 'flash_attention_2':
                    saved = ('forward' in attn.__dict__, attn.__dict__.get('forward'))
                    downstream_overrides.append((attn, saved))
                    original = attn.forward
                    def eager_forward(*args, _module=attn, _forward=original, **kwargs):
                        previous = _module.config._attn_implementation
                        _module.config._attn_implementation = 'eager'
                        try:
                            return _forward(*args, **kwargs)
                        finally:
                            _module.config._attn_implementation = previous
                    attn.forward = eager_forward
            yield
        finally:
            if had_override:
                module.forward = old_override
            else:
                del module.forward
            for handle in hooks:
                handle.remove()
            for attn, (had_override, old_override) in reversed(downstream_overrides):
                if had_override:
                    attn.forward = old_override
                else:
                    del attn.forward
            self.b._sid_active_control = None

    def close(self):
        # Hooks exist only within forward_scope; no global session state to close.
        pass


class SIDSession(HFSession):
    def __init__(self, backend, inputs):
        super().__init__(backend, inputs, False, False)
        self.control = SIDControl(backend, inputs)
        self._sid_kv_length = 0

    def _call(self, **kwargs):
        if kwargs.get('inputs') is not None:
            qlen = int(self.inputs['input_ids'].shape[1])
            kv_length = qlen
        else:
            qlen = 1
            kv_length = self._sid_kv_length + 1
        with self.control.forward_scope(qlen, kv_length):
            result = super()._call(**kwargs)
        self._sid_kv_length = kv_length
        return result
