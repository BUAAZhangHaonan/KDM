"""Observe the first real native VCD/M3ID inputs and tokens in a claimed shard."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from kdm.decoding import distribution
from kdm.io import stable_seed
from kdm.models.hf import vcd_processed_noise
from kdm.probability import log_normalize
from kdm.prompts import task_prompt
from input_audit import input_description


class NativeAuditBackend:
    def __init__(self, backend, model, tasks, cfg):
        self.backend, self.model, self.tasks, self.cfg = backend, model, tasks, cfg
        self.records, self.clean_steps = [], {}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        session = self.backend.session(image, prompt, reference=reference, seed=seed,
                                       need_layers=need_layers)
        if reference == "clean":
            index = len(self.records)
            task = self.tasks[index]
            expected = task_prompt(task["sample"]["question"], "NONE", False)
            if prompt != expected or need_layers:
                raise ValueError("Native clean prompt or layer setting differs")
            self.records.append({"sample_id": task["sample"]["id"],
                                 "main_prompt": prompt, "clean_inputs": input_description(session.inputs),
                                 "main_prompt_token_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                                 "offset_prompt_tokens": list(self.backend.encode(prompt)),
                                 "clean_session": session, "step_checks": []})
        else:
            index = len(self.records) - 1
            record = self.records[index]
            expected_reference = "noise" if self.cfg.method == "vcd" else "text_only"
            if (reference != expected_reference or prompt != record["main_prompt"] or need_layers
                    or seed != stable_seed(record["sample_id"], self.model, 0)):
                raise ValueError("Native reference prompt, mode or seed differs")
            clean = record.pop("clean_session")
            if reference == "noise":
                clean_text = {key: value for key, value in clean.inputs.items() if key != "pixel_values"}
                ref_text = {key: value for key, value in session.inputs.items() if key != "pixel_values"}
                if input_description(clean_text) != input_description(ref_text):
                    raise ValueError("VCD branches differ outside processed pixels")
                expected_pixels = vcd_processed_noise(clean.inputs["pixel_values"], seed)
                if input_description(expected_pixels) != input_description(session.inputs["pixel_values"]):
                    raise ValueError("VCD processed noise differs from the registered operator")
                record.update(noise_step=500, only_processed_pixels_changed=True)
            else:
                if session.text_only is not True:
                    raise ValueError("M3ID reference session is not text-only")
                expected_text = self.backend._text_inputs(prompt)
                if input_description(expected_text) != input_description(session.inputs):
                    raise ValueError("M3ID reference inputs differ from the registered text adapter")
                record.update(text_only_reference=True, original_text_adapter_inputs_equal=True)
            record.update(reference_prompt=prompt, reference_mode=reference, seed=seed,
                          reference_inputs=input_description(session.inputs),
                          reference_prompt_token_ids=session.inputs["input_ids"].detach().cpu().tolist())
        return NativeAuditSession(self, session, index, reference)

    def proof(self, rows):
        by_id = {row["sample"]["id"]: row for row in rows}
        if (len(self.records) != len(self.tasks)
                or len(by_id) != len(rows)
                or set(by_id) != {task["sample"]["id"] for task in self.tasks}):
            raise ValueError("Native operator audit sample coverage differs")
        for record in self.records:
            row = by_id[record["sample_id"]]
            if (len(record["step_checks"]) != len(row["tokens"])
                    or len(row["trace"]) != len(row["tokens"]) or not row["tokens"]):
                raise ValueError("Native operator audit has incomplete real token coverage")
            for check, token, trace in zip(record["step_checks"], row["tokens"], row["trace"]):
                if token != check["pipeline_argmax"]:
                    raise ValueError("Actual native token differs from the pipeline argmax")
                if abs(check["weight"] - trace["weight"]) > 1e-12 or check["active"] != trace["active"]:
                    raise ValueError("Actual native trace differs from the registered operator")
                check.update(observed_token=token, observed_logp_gap=0.0)
                if check["official_logp_gap_at_pipeline_argmax"] > 1e-8:
                    raise ValueError("Actual native token is outside the official numerical maximum")
            if row["terminated"] != (row["tokens"][-1] in self.backend.eos):
                raise ValueError("Native termination differs from actual EOS tokens")
            if self.cfg.method == "m3id" and row["offset_prompt_tokens"] != record["offset_prompt_tokens"]:
                raise ValueError("M3ID recorded offset tokens differ from the actual native prompt")
            record.update(tokens=row["tokens"], text=row["text"], config=row["config"],
                          terminated=row["terminated"], truncated=not row["terminated"],
                          eos_token_ids=sorted(self.backend.eos), wall_s=row["wall_s"])
        if self.clean_steps:
            raise ValueError("Native audit contains unpaired branch logits")
        return {"schema": "kdm_remaining4_native_operator_audit_v1", "passed": True,
                "model": self.model, "method": self.cfg.method, "required": len(self.tasks),
                "completed": len(self.records), "records": self.records,
                "tolerance": {"absolute_logp": 1e-8, "relative_logp": 1e-10},
                "measurement": "actual claimed generation inputs and prefixes",
                "gate_generation_wall_s": sum(row["wall_s"] for row in rows)}


class NativeAuditSession:
    def __init__(self, owner, session, index, reference):
        self.owner, self.session, self.index, self.reference = owner, session, index, reference

    def next(self, prefix):
        step = self.session.next(prefix)
        logits = np.asarray(step.logits, dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("Native full-vocabulary logits are not finite")
        key = (self.index, tuple(prefix))
        if self.reference == "clean":
            self.owner.clean_steps[key] = step
        else:
            clean_step = self.owner.clean_steps.pop(key)
            clean = np.asarray(clean_step.logits, dtype=np.float64)
            cfg = self.owner.cfg
            if cfg.method == "m3id":
                cfg = replace(cfg, m3id_offset=len(self.owner.records[self.index]["offset_prompt_tokens"]))
            project, meta = distribution(clean_step, step, cfg, len(prefix))
            if cfg.method == "vcd":
                keep = clean >= clean.max() + np.log(cfg.beta)
                official = log_normalize(np.where(keep, (1 + cfg.alpha) * clean - cfg.alpha * logits, -np.inf))
            else:
                keep = np.ones(len(clean), dtype=bool)
                p, q = log_normalize(clean), log_normalize(logits)
                active = bool(np.exp(p.max()) < cfg.m3id_threshold)
                weight = float(np.expm1(cfg.m3id_lambda * (len(prefix) + cfg.m3id_offset))) if active else 0.0
                official = log_normalize((1 + weight) * p - weight * q)
                if active != meta["active"] or weight != meta["weight"]:
                    raise ValueError("M3ID activation or weight differs from its registered definition")
            if not np.array_equal(np.isfinite(project), keep):
                raise ValueError("Native candidate support differs from the registered definition")
            if not np.allclose(official[keep], project[keep], atol=1e-8, rtol=1e-10):
                raise ValueError("Native normalized scores differ from the registered definition")
            argmax = int(np.argmax(project))
            self.owner.records[self.index]["step_checks"].append({
                "step": len(prefix), "prefix_token_ids": list(prefix), "support_size": int(keep.sum()),
                "vocab_size": len(clean), "pipeline_argmax": argmax,
                "official_argmax": int(np.argmax(official)),
                "official_logp_gap_at_pipeline_argmax": float(official.max() - official[argmax]),
                "maximum_normalized_logp_error": float(np.max(np.abs(official[keep] - project[keep]))),
                "normalization_sum": float(np.exp(project[keep]).sum()),
                "weight": meta["weight"], "active": meta["active"],
            })
        return step
