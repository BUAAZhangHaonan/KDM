"""Observe real native VCD inputs and logits without changing the decoding chain."""
from __future__ import annotations

import hashlib
import numpy as np

from kdm.io import stable_seed
from kdm.models.hf import vcd_processed_noise
from kdm.probability import contrast, log_normalize
from kdm.prompts import task_prompt


def input_description(value):
    import torch
    if torch.is_tensor(value):
        data = value.detach().contiguous().cpu()
        return {"shape": list(data.shape), "dtype": str(data.dtype),
                "sha256": hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: input_description(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [input_description(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("Unsupported actual processor input: " + type(value).__name__)


class NativeAuditBackend:
    def __init__(self, backend, model, samples):
        self.backend, self.model, self.samples = backend, model, samples
        self.records, self.clean_logits = [], {}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        session = self.backend.session(image, prompt, reference=reference, seed=seed, need_layers=need_layers)
        if reference == "clean":
            index = len(self.records)
            sample = self.samples[index]
            expected = task_prompt(sample["question"], guided=False)
            if prompt != expected or need_layers:
                raise ValueError("Native clean prompt or layer setting differs")
            self.records.append({"sample_id": sample["id"], "main_prompt": prompt,
                                 "clean_inputs": input_description(session.inputs),
                                 "prompt_token_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                                 "step_checks": [], "clean_session": session})
        elif reference == "noise":
            index = len(self.records) - 1
            record = self.records[index]
            expected_seed = stable_seed(record["sample_id"], self.model, 0)
            if prompt != record["main_prompt"] or seed != expected_seed or need_layers:
                raise ValueError("Native reference text, seed or layer setting differs")
            clean = record.pop("clean_session")
            clean_inputs = {key: item for key, item in clean.inputs.items() if key != "pixel_values"}
            noise_inputs = {key: item for key, item in session.inputs.items() if key != "pixel_values"}
            if input_description(clean_inputs) != input_description(noise_inputs):
                raise ValueError("Native branches differ outside the actual visual tensor")
            expected_noise = vcd_processed_noise(clean.inputs["pixel_values"], seed)
            if input_description(expected_noise) != input_description(session.inputs["pixel_values"]):
                raise ValueError("Native noise differs from the registered processed-tensor VCD operator")
            record.update(reference_prompt=prompt, reference_inputs=input_description(session.inputs),
                          seed=seed, noise_step=500, only_processed_pixels_changed=True,
                          expected_noise_recreated_exactly=True)
        else:
            raise ValueError("Native audit received an unregistered reference")
        return NativeAuditSession(self, session, index, reference)

    def proof(self, rows, eos):
        by_id = {row["sample"]["id"]: row for row in rows}
        if len(self.records) != 8 or set(by_id) != {row["id"] for row in self.samples}:
            raise ValueError("Native gate does not cover all eight fixed inputs")
        for record in self.records:
            row = by_id[record["sample_id"]]
            tokens = row["tokens"]
            if len(record["step_checks"]) != len(tokens) or not tokens:
                raise ValueError("Native gate lacks a real per-token check")
            for step, token in zip(record["step_checks"], tokens):
                if token != step["official_argmax"]:
                    raise ValueError("Native real generation token differs from the official operator")
            if row["terminated"] != (tokens[-1] in eos):
                raise ValueError("Native termination flag differs from the actual last token")
            record.update(tokens=tokens, text=row["text"], terminated=row["terminated"],
                          truncated=not row["terminated"], eos_token_ids=sorted(eos),
                          wall_s=row["wall_s"], config=row["config"])
        if self.clean_logits:
            raise ValueError("Native gate has unpaired clean/reference steps")
        return {"passed": True, "completed": 8, "records": self.records,
                "tolerance": {"absolute": 1e-8, "relative": 1e-10},
                "gate_generation_wall_s": sum(row["wall_s"] for row in rows)}


class NativeAuditSession:
    def __init__(self, owner, session, index, reference):
        self.owner, self.session, self.index, self.reference = owner, session, index, reference

    def next(self, prefix):
        step = self.session.next(prefix)
        logits = np.asarray(step.logits, dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("Native actual full-vocabulary logits are not finite")
        key = (self.index, tuple(prefix))
        if self.reference == "clean":
            self.owner.clean_logits[key] = logits
        else:
            clean = self.owner.clean_logits.pop(key)
            keep = clean >= clean.max() + np.log(.1)
            official = log_normalize(np.where(keep, 2.0 * clean - logits, -np.inf))
            project, _ = contrast(log_normalize(clean), log_normalize(logits), 1.0, .1)
            if not np.array_equal(np.isfinite(official), np.isfinite(project)):
                raise ValueError("Native clean candidate support differs from official VCD")
            if not np.allclose(official[keep], project[keep], atol=1e-8, rtol=1e-10):
                raise ValueError("Native score combination differs from official VCD")
            self.owner.records[self.index]["step_checks"].append({
                "step": len(prefix), "prefix_token_ids": list(prefix),
                "support_size": int(keep.sum()), "vocab_size": len(clean),
                "official_argmax": int(np.argmax(official)), "project_argmax": int(np.argmax(project)),
                "maximum_normalized_logp_error": float(np.max(np.abs(official[keep] - project[keep]))),
                "normalization_sum": float(np.exp(project[keep]).sum()),
            })
        return step
