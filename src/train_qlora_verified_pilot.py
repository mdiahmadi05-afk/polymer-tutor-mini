import inspect
import os
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


MODEL_PATH = Path("models/gemma-3-4b-it")

TRAIN_PATH = Path(
    "data/training/main/splits/"
    "train_verified_24.jsonl"
)

VALIDATION_PATH = Path(
    "data/training/main/splits/"
    "validation_verified_6.jsonl"
)

OUTPUT_DIR = Path(
    "outputs/qlora_verified_30_pilot"
)

FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"

SEED = 42
MAX_LENGTH = 512
MAX_STEPS = 5


def check_environment() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"مدل محلی پیدا نشد: {MODEL_PATH}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"فایل Train پیدا نشد: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"فایل Validation پیدا نشد: {VALIDATION_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA در دسترس نیست. آموزش را بدون GPU اجرا نکن."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "GPU یا محیط فعلی از BF16 پشتیبانی نمی‌کند."
        )

    print("=" * 72)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "حافظه GPU:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
    )
    print("نسخه PyTorch:", torch.__version__)
    print("CUDA فعال:", torch.cuda.is_available())
    print("BF16 فعال:", torch.cuda.is_bf16_supported())


def load_and_prepare_datasets():
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_PATH),
            "validation": str(VALIDATION_PATH),
        },
    )

    def convert_to_prompt_completion(example: dict) -> dict:
        messages = example["messages"]

        if len(messages) != 3:
            raise ValueError(
                "هر نمونه باید دقیقاً شامل system، user و assistant باشد."
            )

        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "پیام آخر هر نمونه باید متعلق به assistant باشد."
            )

        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    train_columns = dataset["train"].column_names
    validation_columns = dataset["validation"].column_names

    train_dataset = dataset["train"].map(
        convert_to_prompt_completion,
        remove_columns=train_columns,
        desc="Preparing train dataset",
    )

    validation_dataset = dataset["validation"].map(
        convert_to_prompt_completion,
        remove_columns=validation_columns,
        desc="Preparing validation dataset",
    )

    print("\nتعداد نمونه‌های Train:", len(train_dataset))
    print(
        "تعداد نمونه‌های Validation:",
        len(validation_dataset),
    )

    print("\nنمونه ساختار تبدیل‌شده:")
    print(train_dataset[0])

    return train_dataset, validation_dataset


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_quantized_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\nدر حال بارگذاری Gemma 3 به‌صورت 4-bit...")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
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
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nپارامترهای قابل آموزش:")
    model.print_trainable_parameters()

    return model


def build_sft_config() -> SFTConfig:
    supported_parameters = inspect.signature(
        SFTConfig.__init__
    ).parameters

    config_kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": MAX_STEPS,
        "save_total_limit": 1,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },
        "optim": "paged_adamw_8bit",
        "max_grad_norm": 0.3,
        "weight_decay": 0.0,
        "packing": False,
        "dataset_num_proc": 1,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "remove_unused_columns": True,
    }

    if "max_length" in supported_parameters:
        config_kwargs["max_length"] = MAX_LENGTH
    elif "max_seq_length" in supported_parameters:
        config_kwargs["max_seq_length"] = MAX_LENGTH
    else:
        raise RuntimeError(
            "SFTConfig نه max_length دارد و نه max_seq_length."
        )

    if "eval_strategy" in supported_parameters:
        config_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported_parameters:
        config_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError(
            "پارامتر راهبرد Evaluation در SFTConfig پیدا نشد."
        )

    config_kwargs["eval_steps"] = MAX_STEPS

    if "completion_only_loss" in supported_parameters:
        config_kwargs["completion_only_loss"] = True

    return SFTConfig(**config_kwargs)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    training_args,
):
    trainer_parameters = inspect.signature(
        SFTTrainer.__init__
    ).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
    }

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "SFTTrainer پارامتر tokenizer یا processing_class ندارد."
        )

    return SFTTrainer(**trainer_kwargs)


def print_memory(title: str) -> None:
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    maximum = torch.cuda.max_memory_allocated() / 1024**3

    print("\n" + title)
    print(f"حافظه تخصیص‌یافته: {allocated:.2f} GB")
    print(f"حافظه رزروشده: {reserved:.2f} GB")
    print(f"بیشترین حافظه مصرفی: {maximum:.2f} GB")


def main() -> None:
    set_seed(SEED)
    check_environment()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset, validation_dataset = (
        load_and_prepare_datasets()
    )

    tokenizer = load_tokenizer()
    model = load_quantized_model()
    training_args = build_sft_config()

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        training_args=training_args,
    )

    print_memory("حافظه قبل از آموزش:")

    print("\n" + "=" * 72)
    print("شروع QLoRA آزمایشی")
    print("تعداد مراحل:", MAX_STEPS)
    print("حداکثر طول توکن:", MAX_LENGTH)
    print("=" * 72)

    train_result = trainer.train()

    print("\nنتیجه آموزش:")
    print(train_result)

    print_memory("حافظه پس از آموزش:")

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(FINAL_ADAPTER_DIR)
    )

    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    print("\n" + "=" * 72)
    print("آموزش آزمایشی با موفقیت تمام شد.")
    print("آداپتور ذخیره شد در:")
    print(FINAL_ADAPTER_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
