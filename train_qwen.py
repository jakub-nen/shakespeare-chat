import argparse
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

"""
EXAMPLE OF USAGE:
python train_qwen.py `
  --model .\models\Qwen2.5-3B-Instruct `
  --txt .\data\shakespeare.txt `
  --out_dir .\out\lora_shakespeare_qwen3b_light

"""

def strip_gutenberg_boilerplate(text: str) -> str:
    start_pat = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pat = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"

    m1 = re.search(start_pat, text, flags=re.IGNORECASE | re.DOTALL)
    if m1:
        text = text[m1.end():]

    m2 = re.search(end_pat, text, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        text = text[:m2.start()]

    return text.strip()


def make_style_chunks(text: str, max_chars: int = 1200, min_chars: int = 200):
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    cur = []
    cur_len = 0

    for p in paras:
        if len(p) < 20:
            continue

        if cur_len + len(p) + 2 > max_chars and cur_len >= min_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2

    if cur and cur_len >= min_chars:
        chunks.append("\n\n".join(cur))

    return chunks


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)

    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--min_chars", type=int, default=200)

    ap.add_argument("--qlora", type=int, default=0)

    args = ap.parse_args()

    txt_path = Path(args.txt)

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing TXT file: {txt_path}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    raw = strip_gutenberg_boilerplate(raw)

    chunks = make_style_chunks(
        raw,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )

    system = (
        "You are an assistant who answers helpfully and accurately. "
        "When asked to write creatively, you may use a light Shakespearean tone. "
        "Do not write stage directions, character lists, or scene headers."
    )

    user_prompt = (
        "Rewrite the passage in a light Shakespearean tone, as a single narrator. "
        "Do not add scene headings or character labels.\n"
        "(Passage follows.)"
    )

    examples = [
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": c},
            ]
        }
        for c in chunks
    ]

    ds = Dataset.from_list(examples)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def formatting_func(example):
        return tok.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if args.qlora == 1:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map="auto",
        )

        optim = "paged_adamw_8bit"

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        optim = "adamw_torch"

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out_dir,
        max_length=args.max_seq_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        optim=optim,
        max_grad_norm=0.0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora,
        args=cfg,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    print(f"Done. Adapter saved to: {args.out_dir}")


if __name__ == "__main__":
    main()