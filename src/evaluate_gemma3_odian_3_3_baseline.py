#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)


SYSTEM_PROMPT = (
    "شما مدرس علوم پلیمر هستید. پاسخ را دقیق، کوتاه، علمی و به زبان "
    "فارسی بدهید. فرمول‌ها را بدون تغییر علامت، توان یا ضریب بنویسید."
)

DEFAULT_MODEL_DIR = Path("models/gemma-3-4b-it")
DEFAULT_DATA_ROOT = Path(
    "data/final/odian_ch3/section_3_3/qa"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/pilot_3_3/baseline"
)

SPLIT_FILES = {
    "validation": "qa_3_3_final_validation.jsonl",
    "holdout": "qa_3_3_final_holdout.jsonl",
}

REFUSAL_PHRASES = (
    "اطلاعات کافی نیست",
    "منبع کافی نیست",
    "نمی‌توان پاسخ داد",
    "نمی‌توانم پاسخ دهم",
    "قابل پاسخ نیست",
    "پاسخ مشخص نیست",
    "نمی‌دانم",
)

CSV_FIELDS = [
    "qa_id",
    "split",
    "concept_group",
    "source_record_id",
    "subsection",
    "question_type",
    "difficulty",
    "question_fa",
    "reference_answer_fa",
    "prediction_fa",
    "exact_match",
    "lexical_precision",
    "lexical_recall",
    "lexical_f1",
    "canonical_formula_latex",
    "formula_present",
    "refusal_detected",
    "prompt_tokens",
    "generated_tokens",
    "generation_seconds",
    "tokens_per_second",
    "model_dir",
    "quantization",
    "manual_grade",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ارزیابی Baseline مدل Gemma 3 روی validation و holdout "
            "بخش 3-3 کتاب Odian."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--split",
        choices=("validation", "holdout", "both"),
        default="validation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="۰ یعنی همه نمونه‌ها.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"فایل داده پیدا نشد: {path}")

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
                    f"JSON نامعتبر در {path}، خط {line_number}: {error}"
                ) from error
    return rows


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = text.lower()
    text = re.sub(r"[ـ]+", "", text)
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_overlap(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(
        r"[a-z0-9_]+|[\u0600-\u06FF]+",
        normalized,
    )


def lexical_scores(
    prediction: str,
    reference: str,
) -> tuple[float, float, float]:
    pred_tokens = tokenize_for_overlap(prediction)
    ref_tokens = tokenize_for_overlap(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(
        min(pred_counts[token], ref_counts[token])
        for token in pred_counts.keys() & ref_counts.keys()
    )

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def normalize_formula(text: str) -> str:
    text = str(text or "")
    text = text.replace("$", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    return text


def formula_present(
    prediction: str,
    canonical_formula: str,
) -> bool | None:
    canonical = normalize_formula(canonical_formula)
    if not canonical:
        return None
    prediction_normalized = normalize_formula(prediction)
    return canonical in prediction_normalized


def detect_refusal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        normalize_text(phrase) in normalized
        for phrase in REFUSAL_PHRASES
    )


def build_messages(question: str) -> list[dict[str, Any]]:
    return [
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


def load_processor(model_dir: Path) -> Any:
    return AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        padding_side="left",
    )


def load_model(model_dir: Path) -> Any:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    common_kwargs = {
        "local_files_only": True,
        "device_map": "auto",
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            **common_kwargs,
        )
    except TypeError:
        # Fallback for releases that still use torch_dtype.
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            **common_kwargs,
        )

    model.eval()
    model.config.use_cache = True
    return model


def move_inputs_to_model(
    inputs: Any,
    model: Any,
) -> Any:
    try:
        return inputs.to(model.device)
    except AttributeError:
        return {
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }


@torch.inference_mode()
def generate_answer(
    model: Any,
    processor: Any,
    question: str,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    messages = build_messages(question)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = move_inputs_to_model(inputs, model)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    started = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started

    new_ids = output_ids[0, prompt_tokens:]
    generated_tokens = int(new_ids.shape[-1])

    prediction = processor.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return prediction, prompt_tokens, generated_tokens, elapsed


def selected_splits(split: str) -> list[str]:
    if split == "both":
        return ["validation", "holdout"]
    return [split]


def validate_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []

    qa_ids = [row.get("qa_id") for row in records]
    if len(qa_ids) != len(set(qa_ids)):
        errors.append("qa_id تکراری وجود دارد.")

    for row in records:
        qa_id = row.get("qa_id", "<missing>")

        if not str(row.get("question_fa", "")).strip():
            errors.append(f"{qa_id}: سؤال خالی است.")

        if not str(row.get("answer_fa", "")).strip():
            errors.append(f"{qa_id}: پاسخ مرجع خالی است.")

        if row.get("training_ready") is not True:
            errors.append(f"{qa_id}: training_ready=true نیست.")

        if row.get("answerable_from_source") is not True:
            errors.append(
                f"{qa_id}: answerable_from_source=true نیست."
            )

        if row.get("refusal_expected") is not False:
            errors.append(f"{qa_id}: refusal_expected=false نیست.")

    if errors:
        print("اعتبارسنجی داده ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)


def result_paths(
    output_dir: Path,
    split: str,
) -> tuple[Path, Path, Path]:
    stem = f"baseline_{split}"
    return (
        output_dir / f"{stem}_predictions.jsonl",
        output_dir / f"{stem}_predictions.csv",
        output_dir / f"{stem}_summary.json",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        for row in rows:
            csv_row = {
                field: row.get(field, "")
                for field in CSV_FIELDS
            }
            for field in (
                "exact_match",
                "formula_present",
                "refusal_detected",
            ):
                value = csv_row[field]
                if isinstance(value, bool):
                    csv_row[field] = str(value).lower()
                elif value is None:
                    csv_row[field] = ""
            writer.writerow(csv_row)


def build_summary(
    rows: list[dict[str, Any]],
    model: Any,
    model_dir: Path,
    split: str,
) -> dict[str, Any]:
    formula_rows = [
        row
        for row in rows
        if row["formula_present"] is not None
    ]

    total_seconds = sum(
        row["generation_seconds"]
        for row in rows
    )
    total_generated_tokens = sum(
        row["generated_tokens"]
        for row in rows
    )

    return {
        "evaluation_name": f"gemma3_4b_baseline_{split}",
        "model_dir": str(model_dir),
        "split": split,
        "sample_count": len(rows),
        "quantization": "bitsandbytes_4bit_nf4_double_quant_bf16",
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
        },
        "mean_exact_match": (
            sum(bool(row["exact_match"]) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_lexical_precision": (
            sum(row["lexical_precision"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_lexical_recall": (
            sum(row["lexical_recall"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_lexical_f1": (
            sum(row["lexical_f1"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "formula_sample_count": len(formula_rows),
        "formula_presence_rate": (
            sum(
                bool(row["formula_present"])
                for row in formula_rows
            )
            / len(formula_rows)
            if formula_rows
            else None
        ),
        "refusal_count": sum(
            bool(row["refusal_detected"])
            for row in rows
        ),
        "total_generation_seconds": total_seconds,
        "generated_tokens": total_generated_tokens,
        "overall_tokens_per_second": (
            total_generated_tokens / total_seconds
            if total_seconds > 0
            else 0.0
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "model_memory_footprint_gb": round(
            model.get_memory_footprint() / (1024 ** 3),
            3,
        ),
        "metric_note": (
            "Exact match و lexical overlap فقط شاخص‌های ماشینی سطحی‌اند "
            "و جای ارزیابی علمی دستی را نمی‌گیرند."
        ),
    }


def main() -> None:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit نمی‌تواند منفی باشد.")

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"پوشه مدل پیدا نشد: {args.model_dir}"
        )

    splits = selected_splits(args.split)
    records: list[dict[str, Any]] = []

    for split in splits:
        path = args.data_root / SPLIT_FILES[split]
        split_rows = load_jsonl(path)

        for row in split_rows:
            if row.get("split") != split:
                raise ValueError(
                    f"{row.get('qa_id')}: مقدار split با فایل تطابق ندارد."
                )

        records.extend(split_rows)

    if args.limit:
        records = records[:args.limit]

    validate_records(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path, summary_path = result_paths(
        args.output_dir,
        args.split,
    )

    if args.overwrite:
        for path in (jsonl_path, csv_path, summary_path):
            if path.exists():
                path.unlink()

    completed_ids: set[str] = set()
    results: list[dict[str, Any]] = []

    if jsonl_path.exists():
        results = load_jsonl(jsonl_path)
        completed_ids = {
            row["qa_id"]
            for row in results
        }
        print(
            f"Resume فعال است: {len(completed_ids)} نمونه قبلاً انجام شده."
        )

    remaining = [
        row
        for row in records
        if row["qa_id"] not in completed_ids
    ]

    print("=" * 76)
    print("Baseline evaluation")
    print("Model:", args.model_dir)
    print("Split:", args.split)
    print("Total selected:", len(records))
    print("Remaining:", len(remaining))
    print("Max new tokens:", args.max_new_tokens)
    print("=" * 76)

    if not remaining:
        print("همه نمونه‌های انتخاب‌شده قبلاً ارزیابی شده‌اند.")
        return

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    processor = load_processor(args.model_dir)
    model = load_model(args.model_dir)

    print(
        "Model memory footprint GB:",
        round(model.get_memory_footprint() / (1024 ** 3), 3),
    )

    with jsonl_path.open("a", encoding="utf-8") as output_file:
        for index, record in enumerate(remaining, start=1):
            print(
                f"[{index}/{len(remaining)}] "
                f"{record['qa_id']} | {record['split']}"
            )

            try:
                (
                    prediction,
                    prompt_tokens,
                    generated_tokens,
                    elapsed,
                ) = generate_answer(
                    model=model,
                    processor=processor,
                    question=record["question_fa"],
                    max_new_tokens=args.max_new_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory. اسکریپت را با "
                    "--max-new-tokens 160 دوباره اجرا کن."
                )

            reference = record["answer_fa"]
            precision, recall, f1 = lexical_scores(
                prediction,
                reference,
            )

            formula = record.get(
                "canonical_formula_latex",
                "",
            )

            result = {
                "qa_id": record["qa_id"],
                "split": record["split"],
                "concept_group": record.get(
                    "concept_group",
                    "",
                ),
                "source_record_id": record.get(
                    "source_record_id",
                    "",
                ),
                "subsection": record.get("subsection", ""),
                "question_type": record.get(
                    "question_type",
                    "",
                ),
                "difficulty": record.get("difficulty", ""),
                "question_fa": record["question_fa"],
                "reference_answer_fa": reference,
                "prediction_fa": prediction,
                "exact_match": (
                    normalize_text(prediction)
                    == normalize_text(reference)
                ),
                "lexical_precision": round(precision, 6),
                "lexical_recall": round(recall, 6),
                "lexical_f1": round(f1, 6),
                "canonical_formula_latex": formula,
                "formula_present": formula_present(
                    prediction,
                    formula,
                ),
                "refusal_detected": detect_refusal(prediction),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": round(elapsed, 4),
                "tokens_per_second": round(
                    generated_tokens / elapsed
                    if elapsed > 0
                    else 0.0,
                    4,
                ),
                "model_dir": str(args.model_dir),
                "quantization": (
                    "bitsandbytes_4bit_nf4_"
                    "double_quant_bf16"
                ),
                "manual_grade": "",
                "manual_notes": "",
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            results.append(result)

            print("Prediction:", prediction[:220])
            print(
                "lexical_f1=",
                result["lexical_f1"],
                "| refusal=",
                result["refusal_detected"],
                "| tok/s=",
                result["tokens_per_second"],
            )

    # Keep only rows selected in this invocation when resuming.
    selected_ids = {
        row["qa_id"]
        for row in records
    }
    results = [
        row
        for row in load_jsonl(jsonl_path)
        if row["qa_id"] in selected_ids
    ]

    write_csv(csv_path, results)

    summary = build_summary(
        rows=results,
        model=model,
        model_dir=args.model_dir,
        split=args.split,
    )
    summary["max_new_tokens"] = args.max_new_tokens

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("Baseline evaluation کامل شد.")
    print("Predictions JSONL:", jsonl_path)
    print("Predictions CSV:", csv_path)
    print("Summary:", summary_path)
    print("Samples:", summary["sample_count"])
    print("Mean lexical F1:", round(summary["mean_lexical_f1"], 4))
    print("Refusal count:", summary["refusal_count"])
    print(
        "Overall tokens/sec:",
        round(summary["overall_tokens_per_second"], 3),
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
