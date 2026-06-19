import argparse
import os
import re
from pathlib import Path

import numpy as np
import faiss
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = PROJECT_ROOT / "out"


def set_cwd_to_script_dir():
    os.chdir(SCRIPT_DIR)


def load_chunks(chunks_path: Path):
    s = chunks_path.read_text(encoding="utf-8")
    return s.split("\n\n---CHUNK---\n\n")


def retrieve(index, chunks, embedder, query: str, top_k: int):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    scores, ids = index.search(q_emb, top_k)

    ids = ids[0].tolist()
    scores = scores[0].tolist()

    out = []
    for i, sc in zip(ids, scores):
        if 0 <= i < len(chunks):
            out.append((sc, chunks[i]))
    return out


def build_prompt(tok, question: str, contexts: list[str]):
    system = (
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the CONTEXT does not contain the answer, say: \"I don't know from the provided text.\".\n"
        "Write in a Shakespearean / Early Modern English flavor: mild archaisms, rhythm, and figurative phrasing.\n"
        "But keep it clear, factual, and concise (1–3 sentences).\n"
        "Do NOT write play dialogue, character names, stage directions, or scene headings.\n"
        "Do NOT claim you are William Shakespeare; speak as an assistant.\n"
    )

    context_block = "\n\n".join([f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(contexts)])

    user = (
        f"Question: {question}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "Answer now (concise, factual, Shakespearean flavor)."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    set_cwd_to_script_dir()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir", default=str(OUT_DIR / "lora_shakespeare_qwen3b_light"))

    ap.add_argument("--rag_dir", default=str(OUT_DIR / "rag_wiki_shakespeare"))
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=180)

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.08)

    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    rag_dir = Path(args.rag_dir)
    index_path = rag_dir / "faiss.index"
    chunks_path = rag_dir / "chunks.txt"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing RAG files. Run: python scripts/build_rag_index.py\n"
            f"Expected: {index_path} and {chunks_path}"
        )

    index = faiss.read_index(str(index_path))
    chunks = load_chunks(chunks_path)
    embedder = SentenceTransformer(args.embed_model)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    print("Shakespeare chat. Ctrl+C to exit.\n")

    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            hits = retrieve(index, chunks, embedder, q, top_k=args.top_k)
            ctxs = [c for _, c in hits]

            prompt = build_prompt(tok, q, ctxs)
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = out[0][prompt_len:]
            answer = tok.decode(new_tokens, skip_special_tokens=True).strip()

            answer = re.sub(r"^\s*(assistant|Assistant)\s*[:\n]+", "", answer).strip()


            print("\n" + answer + "\n")

        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
