"""CPU-only proof of the original HF plain-question M3ID token count."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    source = Path(args.identity)
    identity = json.loads(source.read_text())
    backend = identity["definition"]["backend"]
    if backend["factory"] not in {"kdm.models.hf:HFBackend", "kdm.models.internvl_dual:InternVLDualBackend"}:
        raise ValueError("This proof is limited to the registered HFBackend.encode implementation")
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}
    if any(versions[name] != backend["versions"][name] for name in versions):
        raise ValueError("Tokenizer libraries differ from the source identity")
    checkpoint = Path(backend["kwargs"]["model_path"])
    project = Path(identity["definition"]["runtime_admission"]["execution"]["project_root"])
    prompt_source = project / "src/kdm/prompts.py"
    spec = importlib.util.spec_from_file_location("registered_historical_task_prompts", prompt_source)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    plain_prompt = prompt_module.task_prompt(args.question, guided=False)
    checked = {}
    for name, sha in backend["processor"]["files"].items():
        if "tokenizer" in name or "processor" in name:
            actual = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            if actual != sha:
                raise ValueError("Registered processor/tokenizer bytes differ: " + name)
            checked[name] = actual
    from transformers import AutoProcessor, AutoTokenizer
    import torch

    if backend["factory"] == "kdm.models.internvl_dual:InternVLDualBackend":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    else:
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        tokenizer = processor.tokenizer
    tokens = tokenizer.encode(plain_prompt, add_special_tokens=False)
    if torch.cuda.is_initialized():
        raise ValueError("The CPU tokenizer proof initialized CUDA")
    print(json.dumps({"schema": "kdm_historical_plain_tokenizer_config_proof_v1",
        "model": identity["definition"]["model"], "question": args.question,
        "plain_prompt": plain_prompt, "tokens": tokens, "offset": len(tokens),
        "registered_prompt_source": str(prompt_source),
        "registered_prompt_source_sha256": hashlib.sha256(prompt_source.read_bytes()).hexdigest(),
        "source_identity": identity["identity"], "source_identity_path": str(source),
        "source_identity_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "registered_factory": backend["factory"], "checkpoint": str(checkpoint),
        "processor_files_verified": checked, "versions": versions,
        "encode_implementation": "src/kdm/models/hf.py:tokenizer.encode(text,add_special_tokens=False)",
        "model_weights_loaded": False, "GPU_initialized": False, "new_API_calls": 0,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
