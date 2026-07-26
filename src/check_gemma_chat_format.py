import json
from pathlib import Path

from transformers import AutoProcessor


MODEL_PATH = "models/gemma-3-4b-it"

DATASET_PATHS = [
    Path(
        "data/training/main/splits/"
        "train_verified_24.jsonl"
    ),
    Path(
        "data/training/main/splits/"
        "validation_verified_6.jsonl"
    ),
]


def load_jsonl(path: Path) -> list[dict]:
    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در {path}، خط {line_number}: "
                    f"{error}"
                ) from error

    return samples


def main() -> None:
    print("در حال بارگذاری Processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    tokenizer = processor.tokenizer

    all_lengths = []

    for path in DATASET_PATHS:
        samples = load_jsonl(path)

        print("\n" + "=" * 72)
        print("فایل:", path)
        print("تعداد نمونه‌ها:", len(samples))

        for index, sample in enumerate(samples, start=1):
            sample_id = sample["metadata"]["id"]
            messages = sample["messages"]

            try:
                formatted_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as error:
                raise RuntimeError(
                    f"خطا در قالب چت نمونه {sample_id}: {error}"
                ) from error

            token_ids = tokenizer(
                formatted_text,
                add_special_tokens=False,
            )["input_ids"]

            token_count = len(token_ids)
            all_lengths.append(
                (
                    token_count,
                    sample_id,
                    path.name,
                )
            )

            print(
                f"{index:02d}. "
                f"{sample_id}: "
                f"{token_count} توکن"
            )

    all_lengths.sort(reverse=True)

    print("\n" + "=" * 72)
    print("نتیجه نهایی")
    print("تعداد کل نمونه‌ها:", len(all_lengths))
    print("کمترین طول:", min(item[0] for item in all_lengths))
    print("بیشترین طول:", max(item[0] for item in all_lengths))

    average_length = sum(
        item[0]
        for item in all_lengths
    ) / len(all_lengths)

    print(f"میانگین طول: {average_length:.1f}")

    print("\nپنج نمونه بلندتر:")

    for token_count, sample_id, filename in all_lengths[:5]:
        print(
            f"- {sample_id}: "
            f"{token_count} توکن "
            f"({filename})"
        )

    if max(item[0] for item in all_lengths) > 1024:
        raise ValueError(
            "حداقل یک نمونه بیشتر از ۱۰۲۴ توکن است."
        )

    print("\nهمه نمونه‌ها با قالب چت Gemma سازگارند.")
    print("هیچ نمونه‌ای از سقف ۱۰۲۴ توکن عبور نکرده است.")


if __name__ == "__main__":
    main()
