from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

MODEL_PATH = Path("models/gemma-3-4b-it")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA در دسترس نیست. ابتدا نصب PyTorch و GPU را بررسی کنید.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Gemma 3 4B in 4-bit mode...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    local_files_only=True,
).eval()

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "تو یک مدرس علوم و مهندسی پلیمر هستی. "
                    "به زبان فارسی، دقیق، آموزشی و قابل‌فهم پاسخ بده."
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "تفاوت پلیمرهای ترموپلاستیک و ترموست را توضیح بده.",
            }
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_length = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
    )

answer = processor.decode(
    output[0][input_length:],
    skip_special_tokens=True,
).strip()

print("\n--- Gemma Baseline Answer ---\n")
print(answer)

Path("results/baseline_gemma3.txt").write_text(
    answer + "\n",
    encoding="utf-8",
)

print("\nSaved to: results/baseline_gemma3.txt")
