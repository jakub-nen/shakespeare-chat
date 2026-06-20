# Shakespeare Qwen Chat

Small minimal local project for a Shakespeare-style chatbot with LoRA fine tuning and RAG.
The project uses:

- `Qwen/Qwen2.5-3B-Instruct` as the base model
- a LoRA adapter for Shakespeare-like style
- RAG chunks for facts about Shakespeare
- FAISS for searching the RAG text during chat

The base model is not included in this repo. You have to download it yourself.

---


## Install requirements

```powershell
pip install -r requirements.txt
```

If Torch installs without CUDA, install CUDA Torch first:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Download Qwen

```powershell
pip install -U huggingface_hub
mkdir models
hf download Qwen/Qwen2.5-3B-Instruct --local-dir .\models\Qwen2.5-3B-Instruct
```

After this, the model should be here:

```text
models/Qwen2.5-3B-Instruct/
```

---

## Training

Training is done with:

```text
train_qwen.py
```

It reads:

```text
data/shakespeare.txt
```

and creates a LoRA adapter:

```text
out/lora_shakespeare_qwen3b_light/
```

Run:

```powershell
python train_qwen.py `
  --model .\models\Qwen2.5-3B-Instruct `
  --txt .\data\shakespeare.txt `
  --out_dir .\out\lora_shakespeare_qwen3b_light
```

The training is only for style. Facts are not stored in the LoRA adapter.

You do not need to train again these files already exist:

```text
out/lora_shakespeare_qwen3b_light/adapter_config.json
out/lora_shakespeare_qwen3b_light/adapter_model.safetensors
```

---

## Chat

Chat is done with:

```text
chat_qwen.py
```

Run:

```powershell
python chat_qwen.py `
  --base_model .\models\Qwen2.5-3B-Instruct `
  --adapter_dir .\out\lora_shakespeare_qwen3b_light `
  --rag_file .\data\rag_chunks_v3.jsonl
```


---

## How it works

During chat:

```text
question
  -> FAISS searches data/rag_chunks_v3.jsonl
  -> Qwen extracts a plain factual answer
  -> LoRA rewrites it in Shakespeare-like style
```

FAISS cache is created automatically in:

```text
data/.rag_cache/
```


---

## Example results

Some answers are good:

```text
> How many children Shakespeare have?

Three Children doth thus speak, O most assured truth!
```

```text
> What are some of Shakespeare's most famous tragedies?

Hamlet, Othello, King Lear, and Macbeth,
These are the plays of mighty wits that have
Shaped their actions to the world’s applause.
```

It can also refuse unknown facts:

```text
> What was Shakespeare's favorite food?

I cannot thus dish out his diet,
Nor will I tell what meats he did prefer.
My sources are not so impertinent,
To utter such a paltry and unprofitable item.
```

Some answers are weaker. For example, it sometimes gets the main fact right but adds invented details after it. So this is a working RAG + LoRA prototype, not a perfect factual assistant. Check example_dialog.txt or try yourself.


## Stop

Type:

```text
exit
```

or:

```text
quit
```
