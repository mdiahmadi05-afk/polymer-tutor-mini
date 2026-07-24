from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_PATH = Path("models/gemma-3-4b-it")
ADAPTER_PATH = Path("outputs/gemma3-polymer-pilot-qlora")

SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم و مهندسی پلیمر هستی. "
    "پاسخ را مستقیم، علمی، روان و بدون مقدمه اضافی ارائه کن."
)

QUESTIONS = [
    "چرا افزایش وزن مولکولی پلیمر معمولاً استحکام مذاب را افزایش می‌دهد؟",
    "این جمله را اصلاح کن: پلیمر آمورف دارای دمای ذوب مشخص است.",
    "اگر Mw برابر ۹۰۰۰۰ و Mn برابر ۴۵۰۰۰ باشد، پراکندگی را محاسبه و تفسیر کن.",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"مدل اصلی پیدا نشد: {MODEL_PATH}")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"آداپتور پیدا نشد: {ADAPTER_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA در دسترس نیست.")

    print("در حال بارگذاری مدل اصلی به‌صورت 4-bit...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    print("در حال نصب آداپتور LoRA...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()
    model.config.use_cache = True

    for number, question in enumerate(QUESTIONS, start=1):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
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
        )

        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.1,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]

        answer = processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        print("\n" + "=" * 70)
        print(f"سؤال {number}: {question}")
        print("-" * 70)
        print(answer)

    print("\n" + "=" * 70)
    print("آزمایش آداپتور تمام شد.")


if __name__ == "__main__":
    main()
