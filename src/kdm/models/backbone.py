"""Model construction extracted from the KDM source recorded in migration.json."""
import time
import torch
import torch.nn.functional as F
NORM_CHAINS = ['model.model.language_model.norm', 'model.language_model.norm', 'language_model.model.norm', 'language_model.norm', 'model.model.norm', 'model.norm']

class FamilyModel:

    def __init__(self, model_path, device, attn_implementation=None, device_map=None, max_memory=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.mt = self.cfg.model_type
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kw = {'attn_implementation': attn_implementation} if attn_implementation else {}
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device_map if device_map is not None else device, trust_remote_code=True, low_cpu_mem_usage=True, **kw, max_memory=max_memory).eval()
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
                (self.norm, self.norm_path) = (obj, chain)
                break
        self.lcd_ready = self.lm_head is not None and self.norm is not None

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
                xf = x.reshape(x.shape[0], -1)
                return xf @ self._W.T + self._b
        pe = self.model.model.visual.patch_embed
        W = pe.proj.weight.data
        pe.proj = _FastConv3dAsLinear(W.reshape(W.shape[0], -1), pe.proj.bias.data)

    def build(self, pil_img, text):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': pil_img}, {'type': 'text', 'text': text}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
        if self.mt in ('qwen3_5', 'qwen3_vl', 'glm4v'):
            kw['enable_thinking'] = False
        try:
            inputs = self.proc.apply_chat_template(msgs, **kw)
        except TypeError:
            kw.pop('enable_thinking', None)
            inputs = self.proc.apply_chat_template(msgs, **kw)
        if not hasattr(inputs, 'items'):
            raise RuntimeError('apply_chat_template returned no mapping')
        return {k: v.to(self.device) for (k, v) in inputs.items() if hasattr(v, 'to')}

class InternVLModel:

    def __init__(self, model_path, device):
        from types import SimpleNamespace
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import torch
        _cls = get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel', model_path)
        _cls.all_tied_weights_keys = property(lambda self: {})
        self.device = device
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.mt = self.cfg.model_type
        self.proc = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
        self.model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        self.ip = CLIPImageProcessor.from_pretrained(model_path)
        self.num_image_token = int(self.model.num_image_token)
        tok = self.proc.tokenizer
        self.img_ctx_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = self.img_ctx_id
        self.eos = {tok.convert_tokens_to_ids('<|im_end|>')}
        if tok.eos_token_id is not None:
            self.eos.add(tok.eos_token_id)
        try:
            self.lm_head = self.model.language_model.lm_head
            self.norm = self.model.language_model.model.norm
            self.lcd_ready = True
        except AttributeError:
            (self.lm_head, self.norm, self.lcd_ready) = (None, None, False)

    def _query(self, text, patches=0):
        import importlib
        conversation=importlib.import_module(self.model.__class__.__module__.rsplit('.',1)[0]+'.conversation')
        template=conversation.get_conv_template(self.model.template)
        template.system_message=self.model.system_message
        question=('<image>\n' if patches else '')+text
        template.append_message(template.roles[0],question)
        template.append_message(template.roles[1],None)
        query=template.get_prompt()
        if patches:
            image_tokens='<img>'+'<IMG_CONTEXT>'*(self.num_image_token*patches)+'</img>'
            query=query.replace('<image>',image_tokens,1)
        return query

    def build_text(self,text):
        enc=self.proc.tokenizer(self._query(text),return_tensors='pt')
        return {k:v.to(self.device) for k,v in enc.items()}

    def build(self, pil_img, text):
        from .internvl_preprocessing import build_transform,dynamic_preprocess
        size=int(self.cfg.force_image_size or self.cfg.vision_config.image_size)
        transform=build_transform(size)
        tiles=dynamic_preprocess(pil_img.convert('RGB'),image_size=size,max_num=12,use_thumbnail=True)
        px=torch.stack([transform(tile) for tile in tiles])
        enc=self.proc.tokenizer(self._query(text,len(tiles)),return_tensors='pt')
        return {'input_ids':enc['input_ids'].to(self.device),'attention_mask':enc['attention_mask'].to(self.device),
                'pixel_values':px.to(self.device,torch.bfloat16),
                'image_flags':torch.ones((len(tiles),1),dtype=torch.long,device=self.device)}

    def _prefill(self, inputs, ohs=False, oa=False):
        return self.model(pixel_values=inputs['pixel_values'], input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], image_flags=inputs['image_flags'], output_hidden_states=ohs, output_attentions=oa, use_cache=True)

    def _step(self, tok, pkv, ohs=False, oa=False):
        return self.model.language_model(input_ids=torch.tensor([[tok]], device=self.device), past_key_values=pkv, output_hidden_states=ohs, output_attentions=oa, use_cache=True)

def get_engine(model_path, device, attn_implementation=None, device_map=None, max_memory=None):
    """Factory: internvl_chat uses the manual adapter; everything else FamilyModel."""
    from transformers import AutoConfig
    mt = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if mt == 'internvl_chat':
        return InternVLModel(model_path, device)
    return FamilyModel(model_path, device, attn_implementation=attn_implementation, device_map=device_map, max_memory=max_memory)
