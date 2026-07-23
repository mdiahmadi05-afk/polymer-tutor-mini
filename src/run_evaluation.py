import json
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

MODEL_PATH = Path("models/gemma-3-4b-it")
QUESTIONS_PATH = Path("data/evaluation/polymer_baseline_fa.jsonl")
OUTPUT_PATH = Path("results/baseline_30_gemma3_v2.jsonl")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA در دسترس نیست.")

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

questions = []
with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            questions.append(json.loads(line))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

completed_ids = set()
if OUTPUT_PATH.exists():
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                completed_ids.add(json.loads(line)["id"])

print(f"Total questions: {len(questions)}")
print(f"Already completed: {len(completed_ids)}")

for number, item in enumerate(questions, start=1):
    if item["id"] in completed_ids:
        print(f"[{number}/{len(questions)}] Skipped: {item['id']}")
        continue

    print(f"\n[{number}/{len(questions)}] {item['id']}")
    print(item["prompt"])

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "تو یک مدرس علوم و مهندسی پلیمر هستی. "
                        "به فارسی، مستقیم، دقیق و متناسب با سطح سؤال پاسخ بده. "
                        "از مقدمه‌های اضافی، معرفی خود و تکرار سؤال خودداری کن. "
                        "پاسخ را کامل و ترجیحاً در کمتر از ۲۰۰ کلمه ارائه کن. "
                        "مثال‌ها و اصطلاحات علمی را با دقت بررسی کن."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": item["prompt"],
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
            max_new_tokens=500,
            do_sample=False,
        )

    answer = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    result = {
        **item,
        "model": "google/gemma-3-4b-it",
        "stage": "baseline_before_finetuning",
        "answer": answer,
    }

    with OUTPUT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Saved.")

print(f"\nEvaluation completed.")
print(f"Results: {OUTPUT_PATH}")
