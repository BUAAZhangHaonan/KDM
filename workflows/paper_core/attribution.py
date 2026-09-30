"""Finite four-view attribution on caller-supplied prefixes.

This module performs no generation and creates no semantic labels.  The caller
supplies a backend implementing ``session`` and, for complete replay, token
prefixes from frozen responses.  GPU work occurs only when the caller invokes a
session through a real backend.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from kdm.prompts import task_prompt
from kdm.models.hf import vcd_processed_noise

from four_view_math import analyze, log_probs
from native_audit import input_description


def _prompt_views(question: str, marker: str) -> dict[str, str]:
    plain = task_prompt(question, guided=False)
    guided = task_prompt(question, marker=marker, guided=True)
    return {"c": plain, "r": plain, "g": guided, "h": guided}


def _sessions(backend, image, prompts: Mapping[str, str], seed: int) -> dict[str, object]:
    return {
        "c": backend.session(image, prompts["c"], reference="clean", seed=seed),
        "r": backend.session(image, prompts["r"], reference="noise", seed=seed),
        "g": backend.session(image, prompts["g"], reference="clean", seed=seed),
        "h": backend.session(image, prompts["h"], reference="noise", seed=seed),
    }


def _marker_tokens(backend, marker: str, eos: Iterable[int]) -> dict[str, list[int]]:
    markers = ["UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it"]
    return {name: list(backend.encode(name)) for name in markers} | {"eos": sorted(set(int(x) for x in eos))}


def _support_candidates(steps: Mapping[str, object], marker_tokens: Mapping[str, Sequence[int]], top_k: int = 10) -> list[int]:
    ids: set[int] = set(marker_tokens["eos"])
    for tokens in marker_tokens.values():
        ids.update(int(x) for x in tokens[:1])
    for step in steps.values():
        logits = np.asarray(step.logits, dtype=np.float64)
        ids.update(int(x) for x in np.argsort(-logits, kind="stable")[:top_k])
    return sorted(ids)


def _scores(steps: Mapping[str, object], alpha: float, beta: float) -> dict:
    if not np.isfinite([alpha, beta]).all() or alpha < 0 or not 0 <= beta <= 1:
        raise ValueError("Invalid registered contrast parameters")
    lp = {name: log_probs(np.asarray(steps[name].logits, dtype=np.float64)) for name in ("g", "h", "c", "r")}
    if len({value.shape for value in lp.values()}) != 1:
        raise ValueError("Four actual vocabularies differ")
    sg = lp["g"] >= lp["g"].max() + np.log(beta) if beta else np.isfinite(lp["g"])
    sc = lp["c"] >= lp["c"].max() + np.log(beta) if beta else np.isfinite(lp["c"])
    native = (1 + alpha) * lp["c"] - alpha * lp["r"]
    guided = (1 + alpha) * lp["g"] - alpha * lp["h"]
    ip = lp["g"] + alpha * (lp["c"] - lp["r"])
    interaction = alpha * ((lp["g"]-lp["c"]) - (lp["h"]-lp["r"]))
    return {"logp": lp, "Sc": sc, "Sg": sg, "native": native,
            "guided": guided, "ip": ip, "interaction": interaction}


def _masked_distribution(score, support):
    if not support.any():
        return {"argmax": None, "log_normalizer": None, "logp": None}
    values = score[support]
    normalizer = float(values.max()+np.log(np.exp(values-values.max()).sum()))
    return {"argmax": int(np.argmax(np.where(support, score, -np.inf))),
            "log_normalizer": normalizer,
            "logp": np.where(support, score-normalizer, -np.inf)}


def _input_proof(sessions, seed):
    descriptions = {name: input_description(session.inputs) for name, session in sessions.items()}
    for clean, noisy in (("c", "r"), ("g", "h")):
        unmodified = {key: value for key, value in descriptions[clean].items() if key != "pixel_values"}
        reference = {key: value for key, value in descriptions[noisy].items() if key != "pixel_values"}
        if unmodified != reference:
            raise ValueError("Four-view paired branches differ outside processed visual pixels")
        expected = vcd_processed_noise(sessions[clean].inputs["pixel_values"], seed)
        if input_description(expected) != descriptions[noisy]["pixel_values"]:
            raise ValueError("Four-view reference differs from the registered processed-tensor noise")
    if descriptions["c"]["pixel_values"] != descriptions["g"]["pixel_values"] or descriptions["r"]["pixel_values"] != descriptions["h"]["pixel_values"]:
        raise ValueError("The guided and unguided views do not share the same clean/noisy image tensors")
    return {"inputs": descriptions,
            "prompt_token_ids": {name: session.inputs["input_ids"].detach().cpu().tolist()
                                 for name, session in sessions.items()},
            "same_clean_visual_tensor": True, "same_noisy_visual_tensor": True,
            "registered_noise_recreated_exactly": True}


def measure_attribution(
    backend,
    image,
    question: str,
    prefixes: Sequence[Sequence[int]],
    seed: int,
    marker: str = "UNKNOWN",
    diagnostic_positions: Sequence[int] = (),
    eos: Iterable[int] | None = None,
    source_meta: Mapping[str, object] | None = None,
    alpha: float = 1.0,
    beta: float = 0.1,
    observed_tokens: Sequence[int] | None = None,
    candidate_token_ids: Sequence[int] = (),
    candidate_pairs: Sequence[tuple[int, int]] = (),
    observed_text: str | None = None,
) -> dict:
    """Measure four views at supplied prefixes and selected diagnostic steps.

    ``prefixes`` are frozen generated token prefixes; they are teacher-forced
    exactly in every independently created view.  The function returns sparse
    candidate events, full prompt/prefix identity, support states, and
    float64 closure values from ``four_view_math.analyze``.
    """
    if not isinstance(question, str) or not question:
        raise ValueError("question must be non-empty")
    if eos is None:
        eos = getattr(backend, "eos", ())
    prompts = _prompt_views(question, marker)
    marker_ids = _marker_tokens(backend, marker, eos)
    sessions = _sessions(backend, image, prompts, seed)
    input_proof = _input_proof(sessions, seed)
    positions = sorted(set(int(x) for x in diagnostic_positions))
    events = []
    for prefix_index, prefix_value in enumerate(prefixes):
        prefix = tuple(int(x) for x in prefix_value)
        if any(x < 0 for x in prefix):
            raise ValueError("prefix token IDs must be nonnegative")
        position = len(prefix)
        if position not in positions and positions:
            continue
        steps = {name: session.next(prefix) for name, session in sessions.items()}
        scores = _scores(steps, alpha, beta)
        common = scores["Sc"] & scores["Sg"]
        distributions = {
            "native": _masked_distribution(scores["native"], scores["Sc"]),
            "guided": _masked_distribution(scores["guided"], scores["Sg"]),
            "ip": _masked_distribution(scores["ip"], scores["Sg"]),
            "native_on_guided_support": _masked_distribution(scores["native"], scores["Sg"]),
        }
        for name in ("native", "guided", "ip"):
            distributions[name+"_on_common_support"] = _masked_distribution(scores[name], common)
        for weight in (0.0, 0.5, 1.0):
            score = scores["ip"] + weight*scores["interaction"]
            distributions[f"interaction_{weight:g}"] = _masked_distribution(score, scores["Sg"])
        candidates = sorted(set(_support_candidates(steps, marker_ids)) | {int(x) for x in candidate_token_ids})
        candidates = sorted(set(candidates) | {int(token) for pair in candidate_pairs for token in pair})
        if observed_tokens is not None and position < len(observed_tokens):
            candidates = sorted(set(candidates) | {int(observed_tokens[position])})
        view_argmax = {name: distributions[name]["argmax"] for name in ("native", "guided", "ip")}
        candidates = sorted(set(candidates) | {value["argmax"] for value in distributions.values() if value["argmax"] is not None})
        if any(token < 0 or token >= len(scores["logp"]["c"]) for token in candidates):
            raise ValueError("A measured token lies outside the actual vocabulary")
        candidate_rows = []
        for token in candidates:
            candidate_rows.append({
                "token": int(token),
                "decoded": backend.decode([int(token)]),
                "logp": {name: float(scores["logp"][name][token]) for name in ("g", "h", "c", "r")},
                "probability": {name: float(np.exp(scores["logp"][name][token])) for name in ("g", "h", "c", "r")},
                "Sc_member": bool(scores["Sc"][token]), "Sg_member": bool(scores["Sg"][token]),
                "common_support": bool(scores["Sc"][token] and scores["Sg"][token]),
                "method_probability": {name: (float(np.exp(value["logp"][token]))
                    if value["logp"] is not None else None) for name, value in distributions.items()},
                "method_log_probability": {name: (float(value["logp"][token])
                    if value["logp"] is not None and np.isfinite(value["logp"][token]) else None)
                    for name, value in distributions.items()},
            })
        event = {
            "position": position,
            "prefix_tokens": list(prefix),
            "candidate_tokens": candidates,
            "observed_token": (int(observed_tokens[position]) if observed_tokens is not None and position < len(observed_tokens) else None),
            "candidate_rows": candidate_rows,
            "support": {"Sc_size": int(scores["Sc"].sum()), "Sg_size": int(scores["Sg"].sum()), "common_size": int((scores["Sc"] & scores["Sg"]).sum())},
            "argmax": view_argmax,
            "masked_method_distributions": {name: {"argmax": value["argmax"],
                "log_normalizer": value["log_normalizer"]} for name, value in distributions.items()},
            "marker_token_ids": marker_ids,
            "views": {name: {"argmax": int(np.argmax(np.asarray(step.logits))),
                "vocab_size": int(len(step.logits)), "top10_token_ids":
                [int(token) for token in np.argsort(-np.asarray(step.logits), kind="stable")[:10]]}
                for name, step in steps.items()},
            "source_meta": dict(source_meta or {}),
        }
        event["token_proxy_marker_ids"] = {name: ids[:1] for name, ids in marker_ids.items() if name != "eos"}
        event["math_pairs"] = []
        for token_a, token_b in candidate_pairs:
            if token_a == token_b or token_a not in candidates or token_b not in candidates:
                raise ValueError("candidate_pairs must contain distinct tokens in the candidate set")
            math = analyze(steps["g"].logits, steps["h"].logits, steps["c"].logits, steps["r"].logits, token_a, token_b, alpha=alpha, beta=beta)
            for key in ("decomposition_closure", "ip_difference_closure"):
                value = abs(float(math[key])); scale = max(1.0, abs(float(math["guided_pair_margin"])))
                if value > max(1e-8, 1e-10 * scale):
                    raise AssertionError(f"four-view closure exceeds tolerance: {key}={value}")
            event["math_pairs"].append(math)
        events.append(event)
    return {
        "schema": "kdm_four_view_attribution_v1",
        "code_authors": [{"agent": "/root/scoring", "model": "gpt-5.6-luna", "effort": "medium", "call_id": ""},
                         {"agent": "/root", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""}],
        "question": question,
        "prompt_texts": prompts,
        "actual_input_proof": input_proof,
        "prefixes": [list(map(int, p)) for p in prefixes],
        "seed": int(seed),
        "noise_step": 500,
        "alpha": float(alpha),
        "beta": float(beta),
        "marker": marker,
        "marker_token_ids": marker_ids,
        "observed_tokens": (list(map(int, observed_tokens)) if observed_tokens is not None else None),
        "observed_text": observed_text,
        "candidate_pairs": [[int(a), int(b)] for a, b in candidate_pairs],
        "events": events,
        "source_meta": dict(source_meta or {}),
    }


def measure_case(backend, image, question: str, response_tokens: Sequence[int], seed: int, **kwargs) -> dict:
    """Teacher-force one complete saved response, including its EOS token."""
    eos = set(getattr(backend, "eos", ()))
    tokens = [int(x) for x in response_tokens]
    if not tokens:
        raise ValueError("response_tokens must contain the saved response")
    max_tokens = int(kwargs.pop("max_tokens", 32))
    result = measure_attribution(backend, image, question, [tokens[:i] for i in range(len(tokens))], seed, eos=eos, observed_tokens=tokens, observed_text=kwargs.pop("observed_text", None), **kwargs)
    result["response_tokens"] = tokens
    result["response_text"] = result.get("observed_text")
    result["terminated"] = bool(tokens[-1] in eos)
    result["truncated"] = not result["terminated"] and len(tokens) >= max_tokens
    return result
