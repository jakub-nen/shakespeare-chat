#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXAMPLE OF USAGE (saved as shakespeare_lora_5ep in outputs):
python train_lora_cuda.py `
   --model_name .\models\Llama-3.2-1B-Instruct `
   --output_dir .\outputs\shakespeare_lora_5ep `
   --epochs 5 `
   --learning_rate 3e-5 `
   --max_seq_length 768 `
   --batch_size 1 `
   --grad_accum 16 `
   --save_steps 100 `
   --eval_steps 50 `
   --logging_steps 10
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "shakespeare_lora"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path} line {line_no}: {e}") from e
    return rows


def build_dataset(path: Path) -> Dataset:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    texts = []
    for r in rows:
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        texts.append(txt)

    if not texts:
        raise ValueError(f"No non-empty 'text' fields found in {path}")

    return Dataset.from_dict({"text": texts})


def print_gpu_info() -> None:
    print("=" * 80)
    print("CUDA CHECK")
    print("=" * 80)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install CUDA-enabled PyTorch and check NVIDIA driver."
        )
    print("cuda version from torch:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("vram GB:", round(props.total_memory / 1024**3, 2))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("=" * 80)


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    def tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(
        tok,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )


def guess_lora_target_modules(model_name: str) -> list[str]:
    name = model_name.lower()

    # Llama, Mistral, Qwen2, Qwen2.5, Gemma-like decoder modules normally use these names.
    # If a model has different names, print_trainable_parameters/load will fail and you can override.
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def create_training_args(args: argparse.Namespace, use_bf16: bool) -> TrainingArguments:
    """
    Build TrainingArguments in a version-tolerant way.

    Transformers 4.x and 5.x differ in some argument names and accepted fields.
    This function maps what it can and drops unsupported keys instead of crashing.
    """
    raw_kwargs = dict(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim=args.optim,
        lr_scheduler_type="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=not use_bf16,
        bf16=use_bf16,
        dataloader_num_workers=0,  # safer on Windows
        remove_unused_columns=False,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    allowed = set(sig.keys())

    # evaluation strategy naming changed between versions.
    if "eval_strategy" in allowed:
        raw_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in allowed:
        raw_kwargs["evaluation_strategy"] = "steps"

    if "save_strategy" in allowed:
        raw_kwargs["save_strategy"] = "steps"

    # Keep only arguments accepted by the installed Transformers version.
    kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed}

    dropped = sorted(set(raw_kwargs) - set(kwargs))
    if dropped:
        print("TrainingArguments: dropped unsupported args:", dropped)

    return TrainingArguments(**kwargs)



def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--train_file", type=str, default="lora_train_v2.jsonl")
    ap.add_argument("--val_file", type=str, default="lora_val_v2.jsonl")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    ap.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Use Qwen/Qwen2.5-1.5B-Instruct if you do not have Llama access.",
    )

    ap.add_argument("--max_seq_length", type=int, default=768)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    # paged_adamw_8bit is memory friendly when bitsandbytes works.
    # adamw_torch is safer if bitsandbytes is not installed.
    ap.add_argument("--optim", type=str, default="adamw_torch")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print_gpu_info()

    train_path = args.data_dir / args.train_file
    val_path = args.data_dir / args.val_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("DATA")
    print("train:", train_path)
    print("val:", val_path)
    print("output:", args.output_dir)
    print()

    raw_train = build_dataset(train_path)
    raw_val = build_dataset(val_path)

    print("records train:", len(raw_train))
    print("records val:", len(raw_val))
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            print("Loading model in 4-bit QLoRA mode.")
        except Exception as e:
            raise RuntimeError(
                "You used --load_in_4bit, but bitsandbytes quantization could not be configured. "
                "Install bitsandbytes or rerun without --load_in_4bit."
            ) from e
    else:
        model_kwargs["torch_dtype"] = compute_dtype
        print("Loading model in normal LoRA mode:", compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = guess_lora_target_modules(args.model_name)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = tokenize_dataset(raw_train, tokenizer, args.max_seq_length)
    val_ds = tokenize_dataset(raw_val, tokenizer, args.max_seq_length)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = create_training_args(args, use_bf16=use_bf16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)
    trainer.train()

    print("=" * 80)
    print("FINAL EVAL")
    print("=" * 80)
    metrics = trainer.evaluate()
    print(metrics)

    print("=" * 80)
    print("SAVING ADAPTER")
    print("=" * 80)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = args.output_dir / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["use_bf16"] = use_bf16
    run_config["torch_version"] = torch.__version__
    run_config["cuda_version_from_torch"] = torch.version.cuda
    run_config["gpu_name"] = torch.cuda.get_device_name(0)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)

    print("Done.")
    print("Adapter saved to:", args.output_dir)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
