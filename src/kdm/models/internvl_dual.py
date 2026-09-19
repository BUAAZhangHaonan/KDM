"""Explicit two-GPU InternVL constructor; input and forward methods are inherited.

This is a separate registered factory, never an automatic fallback from single GPU.
"""
from __future__ import annotations
from types import SimpleNamespace
from .backbone import InternVLModel
from .hf import HFBackend


def validate_dual_placement(device_map, max_memory, device, dtype):
    expected = {'vision_model': 0, 'mlp1': 0,
                'language_model.model.embed_tokens': 0,
                'language_model.model.rotary_emb': 0,
                'language_model.model.norm': 1, 'language_model.lm_head': 1}
    expected.update({f'language_model.model.layers.{i}': 0 if i < 18 else 1 for i in range(36)})
    if device != 'cuda:0' or dtype != 'bfloat16':
        raise ValueError('Registered InternVL dual placement requires cuda:0 inputs and bfloat16')
    if not isinstance(device_map, dict) or device_map != expected:
        raise ValueError('InternVL dual placement must explicitly cover the approved 18/18 split, with no CPU/disk offload')
    memory = {int(k): v for k, v in (max_memory or {}).items()}
    if memory != {0: '22GiB', 1: '22GiB'}:
        raise ValueError('InternVL dual placement requires explicit 22GiB limits for logical GPUs 0 and 1')
    return memory


class InternVLDualEngine(InternVLModel):
    def __init__(self, model_path, device, device_map, max_memory, dtype='bfloat16'):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        memory = validate_dual_placement(device_map, max_memory, device, dtype)
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if self.cfg.model_type != 'internvl_chat' or self.cfg.llm_config.num_hidden_layers != 36:
            raise ValueError('Approved dual layout requires the registered 36-layer InternVL checkpoint')
        self.mt = self.cfg.model_type
        remote_class = get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel', model_path)
        remote_class.all_tied_weights_keys = property(lambda self: {})
        self.proc = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
        # Same native default attention and precision as InternVLModel. Explicit
        # dispatch replaces its whole-model .to(device), with no input changes.
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            device_map=device_map, max_memory=memory, low_cpu_mem_usage=True).eval()
        actual = getattr(self.model, 'hf_device_map', None)
        if actual != device_map:
            raise RuntimeError('Loaded InternVL device map differs from the registered explicit map')
        if any(parameter.device.type != 'cuda' for parameter in self.model.parameters()):
            raise RuntimeError('InternVL dual model parameters must all reside on CUDA devices')
        self.ip = CLIPImageProcessor.from_pretrained(model_path)
        self.num_image_token = int(self.model.num_image_token)
        tok = self.proc.tokenizer
        self.img_ctx_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = self.img_ctx_id
        self.eos = {tok.convert_tokens_to_ids('<|im_end|>')}
        if tok.eos_token_id is not None:
            self.eos.add(tok.eos_token_id)
        self.lm_head = self.model.language_model.lm_head
        self.norm = self.model.language_model.model.norm
        self.lcd_ready = True


class InternVLDualBackend(HFBackend):
    def __init__(self, model_path, device='cuda:0', *, device_map, max_memory, dtype='bfloat16'):
        import torch
        validate_dual_placement(device_map, max_memory, device, dtype)
        self.em = InternVLDualEngine(model_path, device, device_map, max_memory, dtype)
        self.model = self.em.model
        self.tokenizer = self.em.proc.tokenizer
        self.device = device
        self.eos = set(self.em.eos)
        self.torch = torch
        if not self.eos:
            raise ValueError('InternVL EOS token configuration is empty')
