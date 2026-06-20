import argparse
import hashlib
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

import faiss
import numpy as np
import torch

from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
EXAMPLE OF USAGE:
python chat_qwen.py `
  --base_model .\models\Qwen2.5-3B-Instruct `
  --adapter_dir .\out\lora_shakespeare_qwen3b_light `
  --rag_file .\data\rag_chunks_v3.jsonl 
  
## YOU CAN ADD --show_fact i --show_rag to check what model "think" during anserws
"""
SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_CWD = Path.cwd()


def resolve_path(path):
    p = Path(path)

    if p.is_absolute():
        return p

    for base in [ORIGINAL_CWD, SCRIPT_DIR, SCRIPT_DIR.parent]:
        candidate = base / p
        if candidate.exists():
            return candidate

    return ORIGINAL_CWD / p


def load_jsonl_rag(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            content = str(obj.get("content") or obj.get("text") or "").strip()

            if not content:
                continue

            rows.append({
                "id": obj.get("id", f"chunk_{i:06d}"),
                "section": obj.get("section", ""),
                "article": obj.get("article", "William Shakespeare"),
                "content": content,
            })

    return rows


def load_txt_rag(path):
    text = path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n---CHUNK---\n\n") if c.strip()]

    rows = []

    for i, chunk in enumerate(chunks):
        rows.append({
            "id": f"chunk_{i:06d}",
            "section": "",
            "article": "William Shakespeare",
            "content": chunk,
        })

    return rows


def load_rag(path):
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_rag(path)

    return load_txt_rag(path)


def row_to_retrieval_text(row):
    return (
        f"Article: {row['article']}\n"
        f"Section: {row['section']}\n"
        f"Content:\n{row['content']}"
    ).strip()


def row_to_prompt_text(row):
    if row["section"]:
        return f"Section: {row['section']}\n{row['content']}"

    return row["content"]


def cache_paths(rag_file, embed_model):
    cache_dir = rag_file.parent / ".rag_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_text = f"{rag_file.resolve()}|{rag_file.stat().st_mtime_ns}|{embed_model}|simple_style_v1"
    key = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:16]

    return (
        cache_dir / f"{rag_file.stem}_{key}.faiss.index",
        cache_dir / f"{rag_file.stem}_{key}.json",
    )


def load_or_build_index(rag_file, embedder, embed_model):
    rows = load_rag(rag_file)

    if not rows:
        raise RuntimeError(f"No RAG chunks found in: {rag_file}")

    retrieval_texts = [row_to_retrieval_text(r) for r in rows]
    prompt_texts = [row_to_prompt_text(r) for r in rows]

    index_path, meta_path = cache_paths(rag_file, embed_model)

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        return index, rows, prompt_texts

    print("Building FAISS index from RAG...")

    embeddings = embedder.encode(
        retrieval_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    meta_path.write_text(
        json.dumps(
            {
                "rag_file": str(rag_file),
                "embed_model": embed_model,
                "chunks": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return index, rows, prompt_texts


def retrieve(index, rows, prompt_texts, embedder, question, top_k):
    q_emb = embedder.encode([question], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    scores, ids = index.search(q_emb, min(top_k, len(prompt_texts)))

    hits = []

    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if 0 <= idx < len(prompt_texts):
            hits.append({
                "score": float(score),
                "idx": idx,
                "row": rows[idx],
                "text": prompt_texts[idx],
            })

    return hits


def build_context(hits):
    parts = []

    for i, hit in enumerate(hits, start=1):
        parts.append(f"[CONTEXT {i}]\n{hit['text']}")

    return "\n\n".join(parts)


def is_factual_question(question):
    q = question.lower().strip()

    factual_starts = (
        "who ", "what ", "where ", "when ", "why ", "how ",
        "did ", "does ", "do ", "was ", "were ", "is ", "are ",
        "tell me about ", "summarize ", "compare ",
    )

    if q.startswith(factual_starts):
        return True

    factual_words = (
        "shakespeare", "sonnet", "sonnets", "tragedy", "tragedies",
        "hamlet", "othello", "macbeth", "king lear", "anne hathaway",
        "stratford", "globe", "folio", "bard",
    )

    return any(w in q for w in factual_words)


def make_extract_prompt(tokenizer, question, context):
    system = (
        "You extract factual answers from context.\n"
        "Use ONLY the provided CONTEXT.\n"
        "If the answer is not present, output exactly: NOT_FOUND\n"
        "If the answer is present, output a short plain-English answer.\n"
        "For counting questions, infer the number from phrases in the context, for example 'first of their three children' means 3 children.\n"
        "For yes/no questions, answer yes or no and include the key fact.\n"
        "Copy names, dates, numbers, places, titles, and relationships exactly.\n"
        "Do not write creatively.\n"
        "Do not write poetry.\n"
        "Do not explain your reasoning."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Extract the factual answer."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_style_prompt(tokenizer, factual_answer):
    system = (
        "You rewrite answers in a Shakespearean / Early Modern English flavor.\n"
        "Always produce a stylized answer.\n"
        "Use archaic rhythm, mild ornament, and words like hath, doth, thy, yet keep it understandable.\n"
        "Preserve the factual meaning.\n"
        "Do not intentionally change names, dates, numbers, places, titles, or relationships.\n"
        "Do not say 'Thus speaks the provided text'.\n"
        "Do not mention the context.\n"
        "Do not write stage directions or scene headings.\n"
        "A little poetic style is allowed."
    )

    user = (
        "Rewrite this answer in Shakespearean style.\n\n"
        f"ANSWER:\n{factual_answer}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_unknown_style_prompt(tokenizer, question):
    system = (
        "You answer in a Shakespearean / Early Modern English flavor.\n"
        "The provided text did not contain the factual answer.\n"
        "Say that you know it not from the given text, but say it in a stylized way.\n"
        "Do not say 'Thus speaks the provided text'.\n"
        "Do not be dry."
    )

    user = (
        f"Question:\n{question}\n\n"
        "Give a short stylized refusal."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_creative_prompt(tokenizer, question):
    system = (
        "You are a helpful assistant writing in Shakespearean / Early Modern English flavor.\n"
        "Use archaic rhythm, metaphor, and rich phrasing.\n"
        "Keep the answer understandable.\n"
        "Do not write scene headings unless asked."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    repetition_penalty,
    disable_adapter=False,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    adapter_context = (
        model.disable_adapter()
        if disable_adapter and hasattr(model, "disable_adapter")
        else nullcontext()
    )

    with adapter_context:
        with torch.inference_mode():
            output = model.generate(**kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def clean_answer(text):
    text = text.strip()

    text = re.sub(r"^\s*(assistant|Assistant)\s*[:\n]+", "", text).strip()

    text = re.sub(
        r"^\s*(SCENE|ACT)\s+[IVXLC\d]+.*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()

    text = re.sub(
        r"^\s*(Enter|Exit|Exeunt)\b.*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()

    return text.strip()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir", default="out/lora_shakespeare_qwen3b_light")
    ap.add_argument("--rag_file", default="data/rag_chunks_v3.jsonl")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    ap.add_argument("--top_k", type=int, default=5)

    ap.add_argument("--fact_tokens", type=int, default=100)
    ap.add_argument("--style_tokens", type=int, default=220)
    ap.add_argument("--creative_tokens", type=int, default=260)

    ap.add_argument("--style_temperature", type=float, default=0.55)
    ap.add_argument("--creative_temperature", type=float, default=0.80)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--repetition_penalty", type=float, default=1.08)

    ap.add_argument("--show_rag", action="store_true")
    ap.add_argument("--show_fact", action="store_true")

    args = ap.parse_args()

    adapter_dir = resolve_path(args.adapter_dir)
    rag_file = resolve_path(args.rag_file)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    if not rag_file.exists():
        raise FileNotFoundError(f"Missing RAG file: {rag_file}")

    print("Loading embedder...")
    embedder = SentenceTransformer(args.embed_model)

    index, rows, prompt_texts = load_or_build_index(
        rag_file,
        embedder,
        args.embed_model,
    )

    print(f"RAG chunks loaded: {len(rows)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    print("\nQwen Shakespeare chat ready. Type exit/quit to stop.\n")

    while True:
        try:
            question = input("> ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                break

            if not is_factual_question(question):
                prompt = make_creative_prompt(tokenizer, question)

                answer = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.creative_tokens,
                    do_sample=True,
                    temperature=args.creative_temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    disable_adapter=False,
                )

                print("\n" + clean_answer(answer) + "\n")
                continue

            hits = retrieve(
                index=index,
                rows=rows,
                prompt_texts=prompt_texts,
                embedder=embedder,
                question=question,
                top_k=args.top_k,
            )

            if args.show_rag:
                print("\n[RAG]")
                for i, hit in enumerate(hits, start=1):
                    row = hit["row"]
                    preview = hit["text"].replace("\n", " ")[:450]

                    print(
                        f"{i}. score={hit['score']:.3f} "
                        f"| section={row['section']} | id={row['id']}"
                    )
                    print(f"   {preview}...")
                print()

            context = build_context(hits)

            extract_prompt = make_extract_prompt(
                tokenizer=tokenizer,
                question=question,
                context=context,
            )

            factual_answer = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=extract_prompt,
                max_new_tokens=args.fact_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                disable_adapter=True,
            )

            factual_answer = clean_answer(factual_answer)

            if args.show_fact:
                print(f"[FACT]\n{factual_answer}\n")

            if "NOT_FOUND" in factual_answer.upper() or not factual_answer:
                style_prompt = make_unknown_style_prompt(tokenizer, question)
            else:
                style_prompt = make_style_prompt(tokenizer, factual_answer)

            answer = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=style_prompt,
                max_new_tokens=args.style_tokens,
                do_sample=True,
                temperature=args.style_temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                disable_adapter=False,
            )

            print("\n" + clean_answer(answer) + "\n")

        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()