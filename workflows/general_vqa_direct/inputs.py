"""Local multi-image input boundary using each loaded native processor/template.

Three single-image benchmarks retain backend.session unchanged. MMMU's numbered
image references become native multimodal message items at the same positions;
all source images are included, including an unreferenced stored image. Repeated
references retain repeated image occurrences. No collage or image selection is
performed. Registered image sizes, crop/slice limits, tokenizer and dtypes remain
properties of the original engine.
"""
from __future__ import annotations

import importlib
import re

IMAGE_REFERENCE = re.compile(r"<image\s+(\d+)>")


def parts_for_sample(sample, image_count):
    """Return exact text spans and image indices in the multimodal input order."""
    prompt = sample["prompt"]
    slots = sample.get("image_slots", list(range(1, image_count + 1)))
    if (len(slots) != image_count or len(set(slots)) != image_count
            or any(type(slot) is not int or not 1 <= slot <= 7 for slot in slots)):
        raise ValueError("MMMU image slots must identify every supplied image exactly once")
    mapping = {slot: index for index, slot in enumerate(slots)}
    declared = sample.get("image_placeholder_map")
    if declared is not None and declared != {f"<image {slot}>": index for slot, index in mapping.items()}:
        raise ValueError("MMMU placeholder map differs from the ordered source image slots")
    parts, used, offset = [], set(), 0
    for match in IMAGE_REFERENCE.finditer(prompt):
        slot = int(match.group(1))
        if slot not in mapping:
            raise ValueError("Prompt references an absent source image: " + match.group(0))
        if match.start() > offset:
            parts.append(("text", prompt[offset:match.start()]))
        index = mapping[slot]
        parts.append(("image", index))
        used.add(index)
        offset = match.end()
    if offset < len(prompt):
        parts.append(("text", prompt[offset:]))
    # The same native image-before-text pattern as the original single-image
    # wrapper covers source images that have no explicit numbered reference.
    missing = [("image", index) for index in range(image_count) if index not in used]
    return missing + parts


def family_inputs(engine, images, parts):
    content = [{"type": "image", "image": images[value]} if kind == "image"
               else {"type": "text", "text": value} for kind, value in parts]
    kwargs = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    if engine.mt in ("qwen3_5", "qwen3_vl", "glm4v"):
        kwargs["enable_thinking"] = False
    inputs = engine.proc.apply_chat_template([{"role": "user", "content": content}], **kwargs)
    if not hasattr(inputs, "items"):
        raise ValueError("Native multimodal chat processor did not return model inputs")
    return {key: value.to(engine.device) for key, value in inputs.items() if hasattr(value, "to")}


def minicpm_inputs(engine, images, parts):
    """Use the checkpoint chat() image marker and original nested-image processor."""
    from kdm.models.remote import move_inputs
    engine._configure_tokenizer()
    body = "\n".join("(<image>./</image>)" if kind == "image" else value for kind, value in parts)
    prompt = engine.proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    ordered = [images[value] for kind, value in parts if kind == "image"]
    inputs = engine.proc(text=[prompt], images=[ordered], return_tensors="pt")
    mask = inputs.get("attention_mask")
    if mask is None:
        raise ValueError("MiniCPM processor did not return attention_mask")
    if inputs.get("position_ids") is None:
        positions = mask.long().cumsum(-1) - 1
        inputs["position_ids"] = positions.masked_fill(mask == 0, 0)
    return move_inputs(dict(inputs), engine.device)


def phi_inputs(engine, images, parts):
    """Keep the original Phi user/assistant template and its loaded num_crops=4 processor."""
    from kdm.models.remote import move_inputs
    body, ordered = [], []
    for kind, value in parts:
        if kind == "image":
            ordered.append(images[value])
            body.append(f"<|image_{len(ordered)}|>\n")
        else:
            body.append(value)
    prompt = "<|user|>\n" + "".join(body) + "<|end|>\n<|assistant|>\n"
    return move_inputs(dict(engine.proc(text=prompt, images=ordered, return_tensors="pt")), engine.device)


def internvl_inputs(engine, images, parts):
    """Native chat() multi-image placeholders with the original per-image tiling."""
    import torch
    from kdm.models.internvl_preprocessing import build_transform, dynamic_preprocess
    size = int(engine.cfg.force_image_size or engine.cfg.vision_config.image_size)
    transform = build_transform(size)
    pixels, counts, question = [], [], []
    for kind, value in parts:
        if kind == "text":
            question.append(value)
            continue
        tiles = dynamic_preprocess(images[value].convert("RGB"), image_size=size,
                                   max_num=12, use_thumbnail=True)
        pixels.extend(transform(tile) for tile in tiles)
        counts.append(len(tiles))
        question.append("<image>\n")
    conversation = importlib.import_module(engine.model.__class__.__module__.rsplit(".", 1)[0] + ".conversation")
    template = conversation.get_conv_template(engine.model.template)
    template.system_message = engine.model.system_message
    template.append_message(template.roles[0], "".join(question))
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for count in counts:
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * (engine.num_image_token * count) + "</img>", 1)
    if "<image>" in query:
        raise ValueError("InternVL native image placeholders were not fully expanded")
    encoded = engine.proc.tokenizer(query, return_tensors="pt")
    return {"input_ids": encoded["input_ids"].to(engine.device),
            "attention_mask": encoded["attention_mask"].to(engine.device),
            "pixel_values": torch.stack(pixels).to(engine.device, torch.bfloat16),
            "image_flags": torch.ones((len(pixels), 1), dtype=torch.long, device=engine.device)}


def direct_session(backend, images, sample):
    if not images or len(images) != len(sample["image_paths"]):
        raise ValueError("Every frozen sample image must be supplied")
    if sample["dataset"] != "mmmu":
        if len(images) != 1:
            raise ValueError("The non-MMMU benchmark registration expects one image")
        session = backend.session(images[0], sample["prompt"], reference="clean", seed=0, need_layers=False)
        order, route = [0], "unchanged_registered_single_image_session"
    else:
        from kdm.models.hf import HFSession
        from kdm.models.backbone import FamilyModel
        parts = parts_for_sample(sample, len(images))
        order = [value for kind, value in parts if kind == "image"]
        if set(order) != set(range(len(images))):
            raise ValueError("The native input does not retain every source image")
        engine = backend.em
        if engine.mt == "minicpmv":
            inputs, route = minicpm_inputs(engine, images, parts), "native_minicpm_nested_images"
        elif engine.mt == "phi3_v":
            inputs, route = phi_inputs(engine, images, parts), "native_phi_numbered_images"
        elif engine.mt == "internvl_chat":
            inputs, route = internvl_inputs(engine, images, parts), "native_internvl_num_patches_list"
        elif isinstance(engine, FamilyModel):
            inputs, route = family_inputs(engine, images, parts), "native_processor_multimodal_message"
        else:
            raise ValueError("No native multi-image input adapter is registered for this loaded engine")
        session = HFSession(backend, inputs, False, False)
    from kdm.io import stable_hash
    input_ids = session.inputs["input_ids"].detach().cpu().tolist()
    session.general_vqa_input_evidence = {
        "route": route, "source_image_count": len(images), "source_image_indices_in_input": order,
        "image_occurrences": len(order), "input_ids_sha256": stable_hash(input_ids),
        "input_token_count": sum(len(row) for row in input_ids),
        "exact_manifest_prompt_sha256": stable_hash(sample["prompt"]),
    }
    return session
