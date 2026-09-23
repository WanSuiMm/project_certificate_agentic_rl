#!/usr/bin/env python3
"""Offline load/generate smoke for the pinned survival model on CUDA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshot = snapshot_download(config["model"], revision=config["model_revision"],
                                 local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        torch_dtype=torch.bfloat16, trust_remote_code=False,
    ).to("cuda:0").eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that adds two integers."}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        output = model.generate(prompt, max_new_tokens=16, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    receipt = {"status": "smoke_passed", "model": config["model"],
               "revision": config["model_revision"], "device": torch.cuda.get_device_name(0),
               "prompt_tokens": int(prompt.shape[1]),
               "generated_tokens": int(output.shape[1] - prompt.shape[1]),
               "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
               "torch_version": torch.__version__,
               "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
