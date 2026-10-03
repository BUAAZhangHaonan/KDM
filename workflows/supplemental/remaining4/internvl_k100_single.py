"""Explicit user-authorized K100 map; all InternVL input/forward code is inherited."""
from __future__ import annotations

from types import SimpleNamespace

from kdm.models.backbone import InternVLModel
from kdm.models.hf import HFBackend


def expected_map():
    names = ["vision_model", "mlp1", "language_model.model.embed_tokens",
             "language_model.model.rotary_emb", "language_model.model.norm",
             "language_model.lm_head"]
    names += [f"language_model.model.layers.{i}" for i in range(36)]
    return dict.fromkeys(names, 0)


def validate_placement(device_map, max_memory, device, dtype):
    if device != "cuda:0" or dtype != "bfloat16" or device_map != expected_map():
        raise ValueError("K100 InternVL requires every original module on GPU0 in BF16")
    memory = {int(key): value for key, value in (max_memory or {}).items()}
    if memory != {0: "44GiB"}:
        raise ValueError("The original two 22GiB placement budgets must sum to 44GiB")
    return memory


class InternVLK100SingleEngine(InternVLModel):
    def __init__(self, model_path, device, device_map, max_memory, dtype="bfloat16"):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        memory = validate_placement(device_map, max_memory, device, dtype)
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if self.cfg.model_type != "internvl_chat" or self.cfg.llm_config.num_hidden_layers != 36:
            raise ValueError("The single map requires the unchanged registered 36-layer checkpoint")
        self.mt = self.cfg.model_type
        remote_class = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", model_path)
        remote_class.all_tied_weights_keys = property(lambda self: {})
        self.proc = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            device_map=device_map, max_memory=memory, low_cpu_mem_usage=True).eval()
        if getattr(self.model, "hf_device_map", None) != device_map:
            raise RuntimeError("Loaded InternVL modules differ from the explicit single map")
        if any(parameter.device.type != "cuda" or parameter.device.index != 0
               for parameter in self.model.parameters()):
            raise RuntimeError("Every InternVL parameter must reside on logical GPU0")
        self.ip = CLIPImageProcessor.from_pretrained(model_path)
        self.num_image_token = int(self.model.num_image_token)
        tokenizer = self.proc.tokenizer
        self.img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = self.img_ctx_id
        self.eos = {tokenizer.convert_tokens_to_ids("<|im_end|>")}
        if tokenizer.eos_token_id is not None:
            self.eos.add(tokenizer.eos_token_id)
        self.lm_head = self.model.language_model.lm_head
        self.norm = self.model.language_model.model.norm
        self.lcd_ready = True


class InternVLK100SingleBackend(HFBackend):
    def __init__(self, model_path, device="cuda:0", *, device_map, max_memory, dtype="bfloat16"):
        import torch

        validate_placement(device_map, max_memory, device, dtype)
        self.em = InternVLK100SingleEngine(model_path, device, device_map, max_memory, dtype)
        self.model = self.em.model
        self.tokenizer = self.em.proc.tokenizer
        self.device = device
        self.eos = set(self.em.eos)
        self.torch = torch
        if not self.eos:
            raise ValueError("InternVL EOS configuration is empty")
