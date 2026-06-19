# Shakespeare Chat

Minimal LoRA fine-tuning setup for `meta-llama/Llama-3.2-1B-Instruct` with .

## 1. Download Llama model

First accept access to the model on Hugging Face:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Then login:

```powershell
hf auth login
```

Run from the project folder:

```powershell
mkdir models
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir .\models\Llama-3.2-1B-Instruct
```

Model will be saved in:

```text
models\Llama-3.2-1B-Instruct
```

## 2. Install requirements

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_lora_cuda.txt
```

## 3. Train LoRA

```powershell
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
```

Output adapter:

```text
outputs\shakespeare_lora_5ep
```

## 4. Chat with trained model

```powershell
python chat_lora_console.py `
  --base_model .\models\Llama-3.2-1B-Instruct `
  --adapter_dir .\outputs\shakespeare_lora_5ep
```

Type `exit` or `quit` to stop chat.