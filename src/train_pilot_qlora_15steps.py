import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from trl import SFTConfig, SFTTrainer


MODEL_PATH = Path("models/gemma-3-4b-it")
DATASET_PATH = Path("data/training/pilot_train_fa.jsonl")
OUTPUT_PATH = Path("outputs/gemma3-polymer-pilot-qlora-15steps")

MAX_LENGTH = 512
SEED = 42


def normalize_messages(messages):
    """Convert text messages to Gemma 3 multimodal chat format."""
    normalized = []

    for message in messages:
        normalized.append(
            {
                "role": message["role"],
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                    }
                ],
            }
        )

    return normalized


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"مدل پیدا نشد: {MODEL_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"دیتاست پیدا نشد: {DATASET_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA در دسترس نیست.")

    torch.manual_seed(SEED)

    print("GPU:", torch.cuda.get_device_name(0))
    print("در حال آماده‌سازی پردازشگر و دیتاست...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(
        "json",
        data_files=str(DATASET_PATH),
        split="train",
    )

    def tokenize_sample(example):
        messages = normalize_messages(example["messages"])

        if len(messages) != 3:
            raise ValueError("هر نمونه باید system، user و assistant داشته باشد.")

        prompt_messages = messages[:-1]

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"طول یک نمونه {len(input_ids)} توکن است و از "
                f"حد مجاز {MAX_LENGTH} بیشتر شده است."
            )

        prompt_length = min(
            len(prompt_tokens["input_ids"]),
            len(input_ids),
        )

        labels = input_ids.copy()

        # System and user tokens do not contribute to the loss.
        labels[:prompt_length] = [-100] * prompt_length

        trainable_tokens = sum(label != -100 for label in labels)

        if trainable_tokens == 0:
            raise ValueError("هیچ توکن آموزشی در پاسخ دستیار باقی نمانده است.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sequence_length": len(input_ids),
            "trainable_tokens": trainable_tokens,
        }

    tokenized_dataset = raw_dataset.map(
        tokenize_sample,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing pilot dataset",
    )

    max_sequence_length = max(tokenized_dataset["sequence_length"])
    min_sequence_length = min(tokenized_dataset["sequence_length"])
    total_trainable_tokens = sum(tokenized_dataset["trainable_tokens"])

    print(f"تعداد نمونه‌ها: {len(tokenized_dataset)}")
    print(f"کوتاه‌ترین نمونه: {min_sequence_length} توکن")
    print(f"بلندترین نمونه: {max_sequence_length} توکن")
    print(f"مجموع توکن‌های پاسخ آموزشی: {total_trainable_tokens}")

    tokenized_dataset = tokenized_dataset.remove_columns(
        ["sequence_length", "trainable_tokens"]
    )

    print("\nدر حال بارگذاری مدل 4-bit...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        exclude_modules=r".*(vision_tower|multi_modal_projector).*",
    )

    model = get_peft_model(model, lora_config)

    vision_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            "vision_tower" in name
            or "multi_modal_projector" in name
        )
    ]

    if vision_trainable:
        raise RuntimeError("بخش بینایی به‌اشتباه قابل‌آموزش شده است.")

    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_config = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
        max_length=None,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nآموزش آزمایشی QLoRA شروع شد...")

    torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_PATH))
    processor.save_pretrained(str(OUTPUT_PATH))

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "samples": len(tokenized_dataset),
        "max_steps": training_config.max_steps,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "peak_gpu_memory_gb": round(peak_memory, 3),
    }

    summary_path = OUTPUT_PATH / "training_summary.json"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nآموزش آزمایشی تمام شد.")
    print(f"آداپتور ذخیره شد: {OUTPUT_PATH}")
    print(f"بیشترین حافظه GPU: {peak_memory:.2f} GB")
    print(f"خلاصه آموزش: {summary_path}")


if __name__ == "__main__":
    main()
