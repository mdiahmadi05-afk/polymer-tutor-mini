import argparse
import re
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)


MODEL_PATH = Path("models/gemma-3-4b-it")


def clean_query(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    prefixes = [
        "English query:",
        "Technical query:",
        "Search query:",
        "Query:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text.strip(' "\'')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="تبدیل سؤال فارسی پلیمر به عبارت فنی انگلیسی"
    )

    parser.add_argument(
        "question",
        type=str,
        help="سؤال فارسی کاربر",
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"مدل محلی پیدا نشد: {MODEL_PATH}"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("در حال بارگذاری Gemma 3 در حالت 4-bit...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You rewrite Persian polymer-science questions "
                        "as concise English technical search queries for "
                        "retrieving passages from scientific textbooks. "
                        "Preserve formulas, symbols, polymer names, and "
                        "scientific meaning. Add standard English polymer "
                        "terminology when useful. Do not answer the question. "
                        "Output only one English search query."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": args.question.strip(),
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

    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.05,
        )

    generated_ids = generated_ids[0][input_length:]

    technical_query = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    technical_query = clean_query(technical_query)

    print("\n" + "=" * 72)
    print("سؤال فارسی:")
    print(args.question)
    print("\nعبارت فنی انگلیسی:")
    print(technical_query)


if __name__ == "__main__":
    main()
