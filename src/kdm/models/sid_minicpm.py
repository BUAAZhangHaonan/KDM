"""SID mask for MiniCPM's actual, separated resampler insertion intervals.

The native checkpoint scatters each slice's 64 embeddings into image_bound.
Image and slice delimiters are text tokens and are never eligible for pruning.
This strict mapping is not registered for evaluation and has no GPU admission.
"""
from .hf import HFSession
from .sid import SIDControl, SIDSession


def minicpm_visual_positions(torch, inputs, feature_size):
    ids = inputs.get('input_ids')
    bounds = inputs.get('image_bound')
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('MiniCPM SID requires one explicit input sequence')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 1:
        raise ValueError('MiniCPM SID requires one native image_bound entry')
    bound = bounds[0]
    if bound.ndim != 2 or bound.shape[1] != 2 or not len(bound):
        raise ValueError('MiniCPM SID image_bound must contain native slice intervals')
    if (not bool((bound[:, 0] > 0).all())
            or not bool((bound[:, 1] <= ids.shape[1]).all())
            or not bool(((bound[:, 1] - bound[:, 0]) == feature_size).all())):
        raise ValueError('MiniCPM SID intervals disagree with native resampler features')
    positions = torch.cat([torch.arange(int(start), int(end), device=ids.device)
                           for start, end in bound])
    if not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError('MiniCPM SID image intervals overlap or are out of sequence')
    pixels = inputs.get('pixel_values')
    if not isinstance(pixels, list) or len(pixels) != 1 or len(pixels[0]) != len(bound):
        raise ValueError('MiniCPM SID native slice count differs from image_bound')
    return positions


def restricted_positions_mask(torch, original, hidden, kv_length, positions, selected):
    """Apply the official visual-key exclusion on exact native positions."""
    qlen = hidden.shape[1]
    positions = positions.to(hidden.device)
    selected = selected.to(hidden.device)
    if int(positions.max()) >= kv_length:
        raise ValueError('MiniCPM SID visual mapping exceeds the actual KV sequence')
    blocked = torch.zeros(kv_length, dtype=torch.bool, device=hidden.device)
    blocked[positions] = True
    blocked[selected] = False
    if original is not None:
        if (original.ndim != 4 or original.shape[-1] != kv_length
                or original.shape[-2] != qlen or not original.is_floating_point()):
            raise ValueError('MiniCPM SID requires an additive full-KV causal mask')
        return original.masked_fill(blocked[None, None, None, :], float('-inf'))
    past = kv_length - qlen
    if past < 0:
        raise ValueError('MiniCPM SID invalid query/KV lengths')
    query = torch.arange(qlen, device=hidden.device)[:, None] + past
    key = torch.arange(kv_length, device=hidden.device)
    valid = key <= query
    return torch.zeros((1, 1, qlen, kv_length), device=hidden.device,
                       dtype=hidden.dtype).masked_fill(
        (~valid)[None, None, :, :] | blocked[None, None, None, :], float('-inf'))


class MiniCPMSIDControl(SIDControl):
    """Keep the 100 least attended actual visual tokens at aggregation index 2."""
    def __init__(self, backend, inputs):
        self.b = backend
        self.audit = None
        self.layers = list(backend.model.llm.model.layers)
        self.layer_path = 'llm.model.layers'
        if len(self.layers) < 3:
            raise ValueError('MiniCPM SID requires at least three decoder layers')
        feature_size = int(backend.model.config.query_num)
        self.positions = minicpm_visual_positions(backend.torch, inputs, feature_size)
        self.image_bounds = inputs['image_bound'][0].detach().cpu().tolist()
        self.start = int(self.positions.min())
        self.length = len(self.positions)
        if self.length < 100:
            raise ValueError('MiniCPM native visual token count is below fixed SID keep100')
        self.keep_count = 100
        mask = inputs.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or not bool(mask.all())):
            raise ValueError('MiniCPM SID requires an unpadded single sequence')
        self.attn_module = self.layers[1].self_attn
        if not hasattr(self.attn_module, 'config'):
            raise ValueError('MiniCPM second-layer attention must support eager weights')

    def _mask(self, original, hidden, attention):
        positions = self.positions.to(attention.device)
        visual_attention = attention.mean(dim=0)[-1, positions]
        selected = positions[visual_attention.topk(self.keep_count, largest=False).indices]
        selected = selected.sort().values
        mask = restricted_positions_mask(self.b.torch, original, hidden,
                                         attention.shape[-1], positions, selected)
        return selected, mask


class MiniCPMSIDSession(SIDSession):
    def __init__(self, backend, inputs):
        HFSession.__init__(self, backend, inputs, False, False)
        self.control = MiniCPMSIDControl(backend, inputs)
        self._sid_kv_length = 0
