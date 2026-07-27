#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


DEFAULT_MODEL_DIR = Path("models/gemma-3-4b-it")
DEFAULT_TRAIN_FILE = Path(
    "data/pilot/odian_ch3/section_3_3/"
    "sft/sft_3_3_train_80.jsonl"
)
DEFAULT_EVAL_FILE = Path(
    "data/pilot/odian_ch3/section_3_3/"
    "sft/sft_3_3_in_domain_eval_40.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/pilot_3_3/qlora_r8_lr5e5"
)

TARGET_MODULE_PATTERN = (
    r".*language_model.*\."
    r"(q_proj|k_proj|v_proj|o_proj)$"
)


@dataclass
class RunConfig:
    model_dir: str
    train_file: str
    eval_file: str
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    save_steps: int
    save_total_limit: int
    max_steps: int
    seed: int
    train_limit: int
    eval_limit: int
    target_module_pattern: str
    quantization: str
    compute_dtype: str


class TokenizedConversationDataset(TorchDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor: Any,
        max_length: int,
        dataset_name: str,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.dataset_name = dataset_name

        overlength: list[tuple[str, int]] = []
        zero_completion: list[str] = []

        for row in rows:
            item = tokenize_prompt_completion(
                row=row,
                processor=processor,
            )

            length = int(item["input_ids"].shape[0])
            valid_labels = int(
                (item["labels"] != -100).sum().item()
            )

            if length > max_length:
                overlength.append(
                    (
                        str(
                            row.get("metadata", {}).get(
                                "qa_id",
                                "<unknown>",
                            )
                        ),
                        length,
                    )
                )
                continue

            if valid_labels == 0:
                zero_completion.append(
                    str(
                        row.get("metadata", {}).get(
                            "qa_id",
                            "<unknown>",
                        )
                    )
                )
                continue

            item["metadata"] = row.get("metadata", {})
            self.items.append(item)

        if overlength:
            preview = ", ".join(
                f"{qa_id}:{length}"
                for qa_id, length in overlength[:10]
            )
            raise RuntimeError(
                f"{dataset_name}: {len(overlength)} نمونه از "
                f"max_length={max_length} بلندتر است. "
                f"نمونه‌ها: {preview}. "
                "داده عمداً truncate نشد."
            )

        if zero_completion:
            raise RuntimeError(
                f"{dataset_name}: completion mask برای این نمونه‌ها "
                f"خالی است: {zero_completion[:10]}"
            )

        if not self.items:
            raise RuntimeError(
                f"{dataset_name}: هیچ نمونه قابل استفاده‌ای وجود ندارد."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class CompletionOnlyCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(
            int(item["input_ids"].shape[0])
            for item in batch
        )

        if self.pad_to_multiple_of:
            max_length = (
                math.ceil(
                    max_length / self.pad_to_multiple_of
                )
                * self.pad_to_multiple_of
            )

        input_ids = torch.full(
            (len(batch), max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch), max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(batch), max_length),
            fill_value=-100,
            dtype=torch.long,
        )

        for row_index, item in enumerate(batch):
            length = int(item["input_ids"].shape[0])

            input_ids[row_index, :length] = item[
                "input_ids"
            ]
            attention_mask[row_index, :length] = item[
                "attention_mask"
            ]
            labels[row_index, :length] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QLoRA کم‌ریسک Gemma 3 4B روی داده بخش 3-3 Odian."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
    )

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="۰ یعنی بر اساس epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="فقط برای smoke test؛ ۰ یعنی همه ۸۰ نمونه.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="فقط برای smoke test؛ ۰ یعنی همه ۴۰ نمونه.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help=(
            "مسیر checkpoint یا عبارت latest. "
            "برای اجرای تازه خالی بگذار."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs باید حداقل ۱ باشد.")
    if args.batch_size < 1:
        raise ValueError("--batch-size باید حداقل ۱ باشد.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "--gradient-accumulation-steps باید حداقل ۱ باشد."
        )
    if args.max_length < 128:
        raise ValueError("--max-length بیش از حد کوچک است.")
    if args.lora_r < 1:
        raise ValueError("--lora-r باید مثبت باشد.")
    if args.max_steps < 0:
        raise ValueError("--max-steps نمی‌تواند منفی باشد.")
    if args.train_limit < 0:
        raise ValueError("--train-limit نمی‌تواند منفی باشد.")
    if args.eval_limit < 0:
        raise ValueError("--eval-limit نمی‌تواند منفی باشد.")

    for path in (
        args.model_dir,
        args.train_file,
        args.eval_file,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"مسیر لازم پیدا نشد: {path}"
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در {path}، "
                    f"خط {line_number}: {error}"
                ) from error

    if not rows:
        raise RuntimeError(f"فایل داده خالی است: {path}")

    return rows


def validate_sft_rows(
    rows: list[dict[str, Any]],
    expected_count: int,
    dataset_name: str,
) -> None:
    errors: list[str] = []

    if len(rows) != expected_count:
        errors.append(
            f"تعداد نمونه‌ها باید {expected_count} باشد، "
            f"اما {len(rows)} است."
        )

    qa_ids: list[str] = []

    for index, row in enumerate(rows):
        messages = row.get("messages")
        metadata = row.get("metadata", {})

        qa_id = str(
            metadata.get("qa_id", f"row-{index}")
        )
        qa_ids.append(qa_id)

        if not isinstance(messages, list):
            errors.append(
                f"{qa_id}: messages باید list باشد."
            )
            continue

        roles = [
            message.get("role")
            for message in messages
        ]

        if roles != ["system", "user", "assistant"]:
            errors.append(
                f"{qa_id}: ترتیب roleها باید "
                "system,user,assistant باشد."
            )

        for message in messages:
            if not str(message.get("content", "")).strip():
                errors.append(
                    f"{qa_id}: content خالی وجود دارد."
                )

        if metadata.get("training_ready") is not True:
            errors.append(
                f"{qa_id}: training_ready=true نیست."
            )

    if len(qa_ids) != len(set(qa_ids)):
        errors.append("qa_id تکراری وجود دارد.")

    if errors:
        print(f"اعتبارسنجی {dataset_name} ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)


def to_processor_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for message in messages:
        content = message["content"]

        if isinstance(content, str):
            content = [
                {
                    "type": "text",
                    "text": content,
                }
            ]

        converted.append(
            {
                "role": message["role"],
                "content": content,
            }
        )

    return converted


def longest_common_prefix(
    first: torch.Tensor,
    second: torch.Tensor,
) -> int:
    limit = min(
        int(first.shape[0]),
        int(second.shape[0]),
    )

    index = 0
    while (
        index < limit
        and int(first[index]) == int(second[index])
    ):
        index += 1

    return index


def tokenize_prompt_completion(
    row: dict[str, Any],
    processor: Any,
) -> dict[str, torch.Tensor]:
    messages = to_processor_messages(row["messages"])
    prompt_messages = messages[:-1]

    prompt_encoded = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    full_encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    prompt_ids = prompt_encoded["input_ids"][0]
    full_ids = full_encoded["input_ids"][0]

    prefix_length = longest_common_prefix(
        prompt_ids,
        full_ids,
    )

    if prefix_length < int(prompt_ids.shape[0]) - 2:
        qa_id = row.get("metadata", {}).get(
            "qa_id",
            "<unknown>",
        )
        raise RuntimeError(
            f"{qa_id}: prompt و full conversation prefix "
            "سازگار ندارند."
        )

    labels = full_ids.clone()
    labels[:prefix_length] = -100

    attention_mask = torch.ones_like(
        full_ids,
        dtype=torch.long,
    )

    return {
        "input_ids": full_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "labels": labels.cpu(),
    }


def load_processor(model_dir: Path) -> Any:
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        padding_side="right",
    )

    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "tokenizer نه pad_token و نه eos_token دارد."
            )
        tokenizer.pad_token = tokenizer.eos_token

    return processor


def load_quantized_base_model(
    model_dir: Path,
) -> Any:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    kwargs = {
        "local_files_only": True,
        "device_map": {"": 0},
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }

    try:
        model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                **kwargs,
            )
        )
    except TypeError:
        model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
        )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
    )

    return model


def list_target_modules(model: Any) -> list[str]:
    pattern = re.compile(TARGET_MODULE_PATTERN)

    return sorted(
        name
        for name, module in model.named_modules()
        if pattern.fullmatch(name)
        and isinstance(module, torch.nn.Linear)
    )


def attach_or_load_adapter(
    model: Any,
    args: argparse.Namespace,
    resume_checkpoint: Path | None,
) -> Any:
    if resume_checkpoint is not None:
        model = PeftModel.from_pretrained(
            model,
            resume_checkpoint,
            is_trainable=True,
        )
        return model

    target_modules = list_target_modules(model)

    if not target_modules:
        raise RuntimeError(
            "هیچ q/k/v/o projection در language_model "
            "برای LoRA پیدا نشد."
        )

    print(
        "LoRA target modules found:",
        len(target_modules),
    )
    print("Target preview:")
    for name in target_modules[:8]:
        print("-", name)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULE_PATTERN,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


def parameter_counts(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0

    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    return trainable, total


def make_epoch_loader(
    dataset: TokenizedConversationDataset,
    collator: CompletionOnlyCollator,
    batch_size: int,
    seed: int,
    epoch_index: int,
) -> DataLoader:
    indices = list(range(len(dataset)))
    random.Random(seed + epoch_index).shuffle(indices)

    ordered_items = [
        dataset[index]
        for index in indices
    ]

    return DataLoader(
        ordered_items,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_loss(
    model: Any,
    dataset: TokenizedConversationDataset,
    collator: CompletionOnlyCollator,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    weighted_loss = 0.0
    valid_tokens = 0
    started = time.perf_counter()

    for batch in loader:
        batch = move_batch(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            outputs = model(**batch)
            loss = outputs.loss

        batch_valid_tokens = int(
            (batch["labels"] != -100).sum().item()
        )

        weighted_loss += (
            float(loss.detach().cpu())
            * batch_valid_tokens
        )
        valid_tokens += batch_valid_tokens

    elapsed = time.perf_counter() - started

    model.train()

    average_loss = (
        weighted_loss / valid_tokens
        if valid_tokens
        else float("nan")
    )

    perplexity = (
        math.exp(average_loss)
        if math.isfinite(average_loss)
        and average_loss < 20
        else float("inf")
    )

    return {
        "eval_loss": average_loss,
        "eval_perplexity": perplexity,
        "eval_valid_tokens": float(valid_tokens),
        "eval_seconds": elapsed,
    }


def checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint-step-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(
    output_dir: Path,
) -> Path | None:
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob(
                "checkpoint-step-*"
            )
            if path.is_dir()
            and (
                path / "training_state.pt"
            ).exists()
        ),
        key=checkpoint_number,
    )

    return checkpoints[-1] if checkpoints else None


def resolve_resume_checkpoint(
    args: argparse.Namespace,
) -> Path | None:
    if not args.resume_from:
        return None

    if args.resume_from == "latest":
        checkpoint = find_latest_checkpoint(
            args.output_dir
        )
        if checkpoint is None:
            raise FileNotFoundError(
                "هیچ checkpoint قابل resume پیدا نشد."
            )
        return checkpoint

    checkpoint = Path(args.resume_from)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint پیدا نشد: {checkpoint}"
        )

    return checkpoint


def prune_checkpoints(
    output_dir: Path,
    save_total_limit: int,
) -> None:
    if save_total_limit <= 0:
        return

    checkpoints = sorted(
        (
            path
            for path in output_dir.glob(
                "checkpoint-step-*"
            )
            if path.is_dir()
        ),
        key=checkpoint_number,
    )

    while len(checkpoints) > save_total_limit:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest)


def save_checkpoint(
    model: Any,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    epoch_index: int,
    next_batch_index: int,
    global_step: int,
    best_eval_loss: float,
    history: list[dict[str, Any]],
    save_total_limit: int,
) -> Path:
    checkpoint_dir = (
        output_dir
        / f"checkpoint-step-{global_step}"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )
    processor.save_pretrained(checkpoint_dir)

    state = {
        "epoch_index": epoch_index,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }

    torch.save(
        state,
        checkpoint_dir / "training_state.pt",
    )

    prune_checkpoints(
        output_dir,
        save_total_limit,
    )

    return checkpoint_dir


def prepare_output_dir(
    output_dir: Path,
    overwrite: bool,
    resume_checkpoint: Path | None,
) -> None:
    if resume_checkpoint is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        return

    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        if not overwrite:
            raise RuntimeError(
                f"پوشه خروجی خالی نیست: {output_dir}\n"
                "برای اجرای تازه --overwrite بزن یا برای ادامه "
                "--resume-from latest استفاده کن."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    resume_checkpoint = resolve_resume_checkpoint(
        args
    )

    prepare_output_dir(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        resume_checkpoint=resume_checkpoint,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA در دسترس نیست؛ این QLoRA برای GPU تنظیم شده."
        )

    torch.backends.cuda.matmul.allow_tf32 = True

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)

    validate_sft_rows(
        train_rows,
        expected_count=80,
        dataset_name="train",
    )
    validate_sft_rows(
        eval_rows,
        expected_count=40,
        dataset_name="in-domain eval",
    )

    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if args.eval_limit:
        eval_rows = eval_rows[:args.eval_limit]

    processor = load_processor(args.model_dir)

    train_dataset = TokenizedConversationDataset(
        rows=train_rows,
        processor=processor,
        max_length=args.max_length,
        dataset_name="train",
    )
    eval_dataset = TokenizedConversationDataset(
        rows=eval_rows,
        processor=processor,
        max_length=args.max_length,
        dataset_name="in-domain eval",
    )

    train_lengths = [
        int(item["input_ids"].shape[0])
        for item in train_dataset.items
    ]
    eval_lengths = [
        int(item["input_ids"].shape[0])
        for item in eval_dataset.items
    ]

    print("=" * 76)
    print("Tokenization completed.")
    print(
        "Train rows:",
        len(train_dataset),
        "| max tokens:",
        max(train_lengths),
        "| mean tokens:",
        round(
            sum(train_lengths) / len(train_lengths),
            2,
        ),
    )
    print(
        "Eval rows:",
        len(eval_dataset),
        "| max tokens:",
        max(eval_lengths),
        "| mean tokens:",
        round(
            sum(eval_lengths) / len(eval_lengths),
            2,
        ),
    )
    print("=" * 76)

    model = load_quantized_base_model(
        args.model_dir
    )

    model = attach_or_load_adapter(
        model=model,
        args=args,
        resume_checkpoint=resume_checkpoint,
    )

    trainable, total = parameter_counts(model)

    print(
        "Trainable parameters:",
        f"{trainable:,}",
    )
    print(
        "Total parameters:",
        f"{total:,}",
    )
    print(
        "Trainable percent:",
        round(100 * trainable / total, 6),
    )

    collator = CompletionOnlyCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    micro_batches_per_epoch = math.ceil(
        len(train_dataset) / args.batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / args.gradient_accumulation_steps
    )

    configured_total_steps = (
        args.max_steps
        if args.max_steps > 0
        else optimizer_steps_per_epoch
        * args.epochs
    )

    warmup_steps = max(
        1,
        round(
            configured_total_steps
            * args.warmup_ratio
        ),
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=configured_total_steps,
    )

    start_epoch = 0
    start_batch_index = 0
    global_step = 0
    best_eval_loss = float("inf")
    history: list[dict[str, Any]] = []

    if resume_checkpoint is not None:
        state_path = (
            resume_checkpoint
            / "training_state.pt"
        )

        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            state["scheduler_state_dict"]
        )

        start_epoch = int(
            state["epoch_index"]
        )
        start_batch_index = int(
            state["next_batch_index"]
        )
        global_step = int(
            state["global_step"]
        )
        best_eval_loss = float(
            state["best_eval_loss"]
        )
        history = list(
            state.get("history", [])
        )

        print(
            "Resuming from:",
            resume_checkpoint,
        )
        print(
            "Resume state:",
            {
                "epoch_index": start_epoch,
                "next_batch_index": (
                    start_batch_index
                ),
                "global_step": global_step,
            },
        )

    run_config = RunConfig(
        model_dir=str(args.model_dir),
        train_file=str(args.train_file),
        eval_file=str(args.eval_file),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        seed=args.seed,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        target_module_pattern=(
            TARGET_MODULE_PATTERN
        ),
        quantization=(
            "bitsandbytes_4bit_nf4_"
            "double_quant"
        ),
        compute_dtype="bfloat16",
    )

    (
        args.output_dir
        / "run_config.json"
    ).write_text(
        json.dumps(
            asdict(run_config),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    device = torch.device("cuda:0")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    training_started = time.perf_counter()
    stop_training = False

    for epoch_index in range(
        start_epoch,
        args.epochs,
    ):
        loader = make_epoch_loader(
            dataset=train_dataset,
            collator=collator,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch_index=epoch_index,
        )

        epoch_loss_sum = 0.0
        epoch_valid_tokens = 0
        accumulation_counter = 0

        for batch_index, batch in enumerate(loader):
            if (
                epoch_index == start_epoch
                and batch_index < start_batch_index
            ):
                continue

            batch = move_batch(batch, device)
            valid_tokens = int(
                (batch["labels"] != -100)
                .sum()
                .item()
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = (
                    raw_loss
                    / args.gradient_accumulation_steps
                )

            loss.backward()

            epoch_loss_sum += (
                float(raw_loss.detach().cpu())
                * valid_tokens
            )
            epoch_valid_tokens += valid_tokens
            accumulation_counter += 1

            is_last_batch = (
                batch_index + 1 == len(loader)
            )
            should_step = (
                accumulation_counter
                >= args.gradient_accumulation_steps
                or is_last_batch
            )

            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                args.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            accumulation_counter = 0
            global_step += 1

            current_loss = (
                epoch_loss_sum / epoch_valid_tokens
                if epoch_valid_tokens
                else float("nan")
            )

            print(
                f"epoch={epoch_index + 1}/"
                f"{args.epochs} "
                f"step={global_step}/"
                f"{configured_total_steps} "
                f"train_loss={current_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"gpu_alloc_gb="
                f"{torch.cuda.memory_allocated() / 1024**3:.3f} "
                f"gpu_reserved_gb="
                f"{torch.cuda.memory_reserved() / 1024**3:.3f}"
            )

            history.append(
                {
                    "event": "optimizer_step",
                    "epoch": epoch_index + 1,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "train_loss_running": (
                        current_loss
                    ),
                    "learning_rate": (
                        scheduler.get_last_lr()[0]
                    ),
                }
            )

            if (
                args.save_steps > 0
                and global_step % args.save_steps == 0
            ):
                checkpoint = save_checkpoint(
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=args.output_dir,
                    epoch_index=epoch_index,
                    next_batch_index=(
                        batch_index + 1
                    ),
                    global_step=global_step,
                    best_eval_loss=(
                        best_eval_loss
                    ),
                    history=history,
                    save_total_limit=(
                        args.save_total_limit
                    ),
                )
                print(
                    "Checkpoint saved:",
                    checkpoint,
                )

            if (
                args.max_steps > 0
                and global_step >= args.max_steps
            ):
                stop_training = True
                break

        eval_metrics = evaluate_loss(
            model=model,
            dataset=eval_dataset,
            collator=collator,
            batch_size=args.batch_size,
            device=device,
        )

        train_epoch_loss = (
            epoch_loss_sum / epoch_valid_tokens
            if epoch_valid_tokens
            else float("nan")
        )

        epoch_metrics = {
            "event": "epoch_end",
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "train_loss": train_epoch_loss,
            **eval_metrics,
        }
        history.append(epoch_metrics)

        print(
            "Epoch evaluation:",
            json.dumps(
                epoch_metrics,
                ensure_ascii=False,
                indent=2,
            ),
        )

        epoch_adapter_dir = (
            args.output_dir
            / f"adapter-epoch-{epoch_index + 1}"
        )
        model.save_pretrained(
            epoch_adapter_dir,
            safe_serialization=True,
        )
        processor.save_pretrained(
            epoch_adapter_dir
        )

        if (
            eval_metrics["eval_loss"]
            < best_eval_loss
        ):
            best_eval_loss = float(
                eval_metrics["eval_loss"]
            )

            best_dir = (
                args.output_dir / "best_adapter"
            )
            if best_dir.exists():
                shutil.rmtree(best_dir)

            model.save_pretrained(
                best_dir,
                safe_serialization=True,
            )
            processor.save_pretrained(best_dir)

            (
                best_dir / "best_metric.json"
            ).write_text(
                json.dumps(
                    {
                        "eval_loss": (
                            best_eval_loss
                        ),
                        "epoch": epoch_index + 1,
                        "global_step": global_step,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "Best adapter updated:",
                best_dir,
            )

        save_checkpoint(
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=args.output_dir,
            epoch_index=epoch_index + 1,
            next_batch_index=0,
            global_step=global_step,
            best_eval_loss=best_eval_loss,
            history=history,
            save_total_limit=args.save_total_limit,
        )

        start_batch_index = 0

        if stop_training:
            break

    training_seconds = (
        time.perf_counter()
        - training_started
    )

    final_dir = (
        args.output_dir / "final_adapter"
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)

    model.save_pretrained(
        final_dir,
        safe_serialization=True,
    )
    processor.save_pretrained(final_dir)

    final_summary = {
        "completed": True,
        "global_step": global_step,
        "configured_total_steps": (
            configured_total_steps
        ),
        "best_eval_loss": best_eval_loss,
        "training_seconds": training_seconds,
        "training_minutes": (
            training_seconds / 60
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": (
            100 * trainable / total
        ),
        "max_gpu_allocated_gb": (
            torch.cuda.max_memory_allocated()
            / 1024**3
        ),
        "max_gpu_reserved_gb": (
            torch.cuda.max_memory_reserved()
            / 1024**3
        ),
        "history": history,
        "best_adapter": str(
            args.output_dir / "best_adapter"
        ),
        "final_adapter": str(final_dir),
    }

    (
        args.output_dir
        / "training_summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("QLoRA training completed.")
    print("Global steps:", global_step)
    print(
        "Best eval loss:",
        best_eval_loss,
    )
    print(
        "Training minutes:",
        round(training_seconds / 60, 2),
    )
    print(
        "Max GPU allocated GB:",
        round(
            final_summary[
                "max_gpu_allocated_gb"
            ],
            3,
        ),
    )
    print(
        "Best adapter:",
        args.output_dir / "best_adapter",
    )
    print(
        "Summary:",
        args.output_dir
        / "training_summary.json",
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
