import json
import os
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)


MODEL_PATH = Path("models/gemma-3-4b-it")

ADAPTER_PATH = Path(
    "outputs/qlora_verified_30_pilot/"
    "final_adapter"
)

VALIDATION_PATH = Path(
    "data/training/main/splits/"
    "validation_verified_6.jsonl"
)

OUTPUT_DIR = Path(
    "outputs/comparisons/"
    "verified_pilot"
)

JSONL_OUTPUT_PATH = OUTPUT_DIR / "base_vs_adapter_validation.jsonl"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "base_vs_adapter_validation.md"

MAX_NEW_TOKENS = 180


def check_paths() -> None:
    required_paths = [
        MODEL_PATH,
        ADAPTER_PATH,
        VALIDATION_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"مسیر لازم پیدا نشد: {path}"
            )

    adapter_config = ADAPTER_PATH / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(
            f"فایل تنظیمات آداپتور پیدا نشد: {adapter_config}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA در دسترس نیست."
        )


def load_validation_samples() -> list[dict]:
    samples = []

    with VALIDATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: {error}"
                ) from error

    if len(samples) != 6:
        raise ValueError(
            f"تعداد Validation باید ۶ باشد، "
            f"اما {len(samples)} نمونه پیدا شد."
        )

    return samples


def load_processor():
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return processor


def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("در حال بارگذاری مدل پایه به‌صورت 4-bit...")

    base_model = (
        Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
    )

    base_model.config.use_cache = True

    print("در حال بارگذاری آداپتور QLoRA...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    return model


def prepare_inputs(
    processor,
    messages: list[dict],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    formatted_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=formatted_text,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    messages: list[dict],
) -> str:
    device = next(model.parameters()).device

    inputs = prepare_inputs(
        processor=processor,
        messages=messages,
        device=device,
    )

    input_length = inputs["input_ids"].shape[-1]

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    answer_ids = generated_ids[
        0,
        input_length:,
    ]

    answer = processor.tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    )

    return answer.strip()


def split_user_content(
    user_content: str,
) -> tuple[str, str]:
    source_marker = "متن منبع:\n"
    question_marker = "\n\nسؤال:\n"

    if (
        source_marker in user_content
        and question_marker in user_content
    ):
        body = user_content.split(
            source_marker,
            1,
        )[1]

        source_text, question = body.split(
            question_marker,
            1,
        )

        return source_text.strip(), question.strip()

    return "", user_content.strip()


def write_jsonl(results: list[dict]) -> None:
    with JSONL_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_markdown(results: list[dict]) -> None:
    lines = [
        "# مقایسه مدل پایه و آداپتور QLoRA",
        "",
        f"تعداد نمونه‌ها: {len(results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## {index}. {result['id']}",
                "",
                "### سؤال",
                "",
                result["question"],
                "",
                "### پاسخ مرجع",
                "",
                result["reference_answer"],
                "",
                "### پاسخ مدل پایه",
                "",
                result["base_answer"],
                "",
                "### پاسخ مدل دارای آداپتور",
                "",
                result["adapter_answer"],
                "",
                "---",
                "",
            ]
        )

    MARKDOWN_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def print_memory() -> None:
    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    maximum = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print("\nمصرف حافظه GPU:")
    print(f"- تخصیص‌یافته: {allocated:.2f} GB")
    print(f"- رزروشده: {reserved:.2f} GB")
    print(f"- بیشترین مصرف: {maximum:.2f} GB")


def main() -> None:
    check_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_validation_samples()
    processor = load_processor()
    model = load_model()

    results = []

    print("\n" + "=" * 72)
    print("شروع مقایسه مدل پایه و آداپتور")
    print("=" * 72)

    for index, sample in enumerate(samples, start=1):
        sample_id = sample["metadata"]["id"]
        messages = sample["messages"]

        prompt_messages = messages[:-1]
        reference_answer = messages[-1]["content"]

        user_content = prompt_messages[-1]["content"]

        source_text, question = split_user_content(
            user_content
        )

        print(
            f"\n[{index}/{len(samples)}] "
            f"نمونه {sample_id}"
        )

        print("- تولید پاسخ مدل پایه...")

        with model.disable_adapter():
            base_answer = generate_answer(
                model=model,
                processor=processor,
                messages=prompt_messages,
            )

        print("- تولید پاسخ مدل دارای آداپتور...")

        adapter_answer = generate_answer(
            model=model,
            processor=processor,
            messages=prompt_messages,
        )

        results.append(
            {
                "id": sample_id,
                "source_text": source_text,
                "question": question,
                "reference_answer": reference_answer,
                "base_answer": base_answer,
                "adapter_answer": adapter_answer,
            }
        )

        print("  پاسخ پایه:", base_answer)
        print("  پاسخ آداپتور:", adapter_answer)

    write_jsonl(results)
    write_markdown(results)
    print_memory()

    print("\n" + "=" * 72)
    print("مقایسه با موفقیت تمام شد.")
    print("فایل JSONL:")
    print(JSONL_OUTPUT_PATH)
    print("فایل خوانا:")
    print(MARKDOWN_OUTPUT_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()
