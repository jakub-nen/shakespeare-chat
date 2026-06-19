import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

"""
EXAMPLE OF USAGE:
python chat_console.py `
  --base_model .\models\Llama-3.2-1B-Instruct `
  --adapter_dir .\outputs\shakespeare_lora_5ep_weak `
  --rag_file .\data\rag_chunks_v2.jsonl `
  --show_rag

"""
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, device, prefix, batch_size=8):
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[i:i + batch_size]]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        vectors.append(emb.cpu())

    return torch.cat(vectors, dim=0)


def tokenize_words(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower()))


def lexical_score(query, document):
    q = tokenize_words(query)
    d = tokenize_words(document)

    if not q or not d:
        return 0.0

    return len(q & d) / max(1, len(q))


def retrieve(question, chunks, chunk_embeddings, emb_tokenizer, emb_model, device, top_k):
    q_emb = embed_texts(
        [question],
        emb_tokenizer,
        emb_model,
        device,
        prefix="query: ",
    )

    dense_scores = (q_emb @ chunk_embeddings.T).squeeze(0)

    retrieval_texts = [
        c.get("retrieval_text") or c.get("prompt_text") or c.get("content") or c.get("text") or ""
        for c in chunks
    ]

    lexical_scores = torch.tensor(
        [lexical_score(question, text) for text in retrieval_texts],
        dtype=torch.float32,
    )

    # Hybrid retrieval: mostly semantic, slightly keyword-based.
    # This helps questions such as "How many children did Shakespeare have?"
    final_scores = (0.85 * dense_scores.cpu()) + (0.15 * lexical_scores)

    top = torch.topk(final_scores, k=min(top_k, len(chunks)))

    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        results.append((score, chunks[idx]))

    return results


def build_messages(system_prompt, history, context, user_text):
    messages = [{"role": "system", "content": system_prompt}]

    for old_user, old_answer in history[-3:]:
        messages.append({"role": "user", "content": old_user})
        messages.append({"role": "assistant", "content": old_answer})

    user_prompt = (
        "Use the RAG context below to answer the question.\n"
        "If the answer is not present in the context, say that you lack enough information.\n"
        "Do not invent specific facts, names, dates, numbers, or events.\n\n"
        "RAG CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{user_text}"
    )

    messages.append({"role": "user", "content": user_prompt})
    return messages


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--rag_file", default=r".\data\rag_chunks_v3_labeled.jsonl")
    parser.add_argument("--embedding_model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--show_rag", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print("Loading RAG chunks...")
    chunks = load_jsonl(args.rag_file)

    if not chunks:
        raise RuntimeError("No RAG chunks loaded.")

    retrieval_texts = [
        c.get("retrieval_text") or c.get("prompt_text") or c.get("content") or c.get("text") or ""
        for c in chunks
    ]

    print(f"Loaded RAG chunks: {len(chunks)}")

    print("Loading embedding model...")
    emb_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    emb_model = AutoModel.from_pretrained(args.embedding_model).to(device)
    emb_model.eval()

    print("Embedding RAG chunks...")
    chunk_embeddings = embed_texts(
        retrieval_texts,
        emb_tokenizer,
        emb_model,
        device,
        prefix="passage: ",
        batch_size=8,
    )

    print("Loading Llama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base Llama model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    system_prompt = (
        "You are a Shakespeare for machine chatbot.\n"
        "Answer FACTUAL questions using ONLY the provided CONTEXT.\n"
        "If the CONTEXT does not contain the answer, say: "
        "\"I cannot find that in the provided text.\" and stop.\n"
        "Write in a light Shakespearean tone (archaic flavor), but keep clarity and do not invent facts.\n"
        "Do NOT write play dialogue, character lists, or scene headings.\n"
    )

    print("\nRAG + LoRA chat ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    history = []

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(
            user_text,
            chunks,
            chunk_embeddings,
            emb_tokenizer,
            emb_model,
            device,
            args.top_k,
        )

        context_parts = []

        print("\n[RAG selected chunks]")
        for i, (score, chunk) in enumerate(retrieved, start=1):
            article = chunk.get("article", "unknown")
            section = chunk.get("section", "unknown")
            chunk_no = chunk.get("chunk", "unknown")
            chunk_id = chunk.get("id", "unknown")

            print(f"{i}. score={score:.3f} | article={article} | section={section} | chunk={chunk_no} | id={chunk_id}")

            if args.show_rag:
                preview = (chunk.get("content") or "")[:500].replace("\n", " ")
                print(f"   {preview}...")

            context_parts.append(
                chunk.get("prompt_text")
                or (
                    f"[RAG SOURCE {i}]\n"
                    f"Article: {article}\n"
                    f"Section: {section}\n"
                    f"Chunk: {chunk_no}\n"
                    f"Content:\n{chunk.get('content', '')}"
                )
            )

        context = "\n\n---\n\n".join(context_parts)

        messages = build_messages(system_prompt, history, context, user_text)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print("\nModel:", answer, "\n")

        history.append((user_text, answer))


if __name__ == "__main__":
    main()
