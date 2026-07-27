#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/final/odian_ch3/section_3_3/"
    "qa/qa_3_3_final_all.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/pilot/odian_ch3/section_3_3"
)

SYSTEM_PROMPT = (
    "شما مدرس علوم پلیمر هستید. پاسخ را دقیق، کوتاه، علمی و به زبان "
    "فارسی بدهید. فرمول‌ها را بدون تغییر علامت، توان یا ضریب بنویسید."
)

EXPECTED_TOTAL_QA = 147
EXPECTED_TRAIN_SOURCE_RECORDS = 40
EXPECTED_CHALLENGE_VALIDATION_QA = 12
EXPECTED_CHALLENGE_HOLDOUT_QA = 15

QUESTION_TYPE_BY_VARIANT = {
    "v1": "direct_recall",
    "v2": "reasoning_and_conditions",
    "v3": "misconception_correction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ساخت Splitهای کم‌نشت برای Pilot Training بخش 3-3 Odian."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد: {path}")

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
                    f"JSON نامعتبر در خط {line_number}: {error}"
                ) from error
    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def qa_variant(qa_id: str) -> str:
    match = re.search(r"_(v[123])$", qa_id)
    if not match:
        raise ValueError(
            f"variant از qa_id قابل استخراج نیست: {qa_id}"
        )
    return match.group(1)


def source_record_number(source_record_id: str) -> int:
    match = re.search(r"_(\d+)$", source_record_id)
    if not match:
        raise ValueError(
            "شماره source_record_id قابل استخراج نیست: "
            f"{source_record_id}"
        )
    return int(match.group(1))


def reserved_variant_for_record(
    source_record_id: str,
) -> str:
    number = source_record_number(source_record_id)
    remainder = number % 3

    if remainder == 0:
        return "v1"
    if remainder == 1:
        return "v2"
    return "v3"


def to_sft_row(
    qa_row: dict[str, Any],
    pilot_split: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": qa_row["question_fa"],
            },
            {
                "role": "assistant",
                "content": qa_row["answer_fa"],
            },
        ],
        "metadata": {
            "qa_id": qa_row["qa_id"],
            "pilot_split": pilot_split,
            "original_split": qa_row["split"],
            "concept_group": qa_row["concept_group"],
            "source_record_id": qa_row["source_record_id"],
            "question_type": qa_row["question_type"],
            "printed_pages": qa_row["printed_pages"],
            "pdf_pages": qa_row["pdf_pages"],
            "training_ready": qa_row["training_ready"],
            "domain_expert_verified": qa_row[
                "domain_expert_verified"
            ],
        },
    }


def validate_source(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []

    if len(rows) != EXPECTED_TOTAL_QA:
        errors.append(
            f"تعداد QA باید {EXPECTED_TOTAL_QA} باشد، "
            f"اما {len(rows)} است."
        )

    qa_ids = [row.get("qa_id") for row in rows]
    if len(qa_ids) != len(set(qa_ids)):
        errors.append("qa_id تکراری وجود دارد.")

    question_hashes = [
        sha256_text(str(row.get("question_fa", "")).strip())
        for row in rows
    ]
    if len(question_hashes) != len(set(question_hashes)):
        errors.append("سؤال دقیقاً تکراری وجود دارد.")

    per_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_record[row["source_record_id"]].append(row)

        if row.get("training_ready") is not True:
            errors.append(
                f"{row.get('qa_id')}: training_ready=true نیست."
            )

        if row.get("answerable_from_source") is not True:
            errors.append(
                f"{row.get('qa_id')}: answerable_from_source=true نیست."
            )

        if row.get("refusal_expected") is not False:
            errors.append(
                f"{row.get('qa_id')}: refusal_expected=false نیست."
            )

        if not str(row.get("question_fa", "")).strip():
            errors.append(
                f"{row.get('qa_id')}: سؤال خالی است."
            )

        if not str(row.get("answer_fa", "")).strip():
            errors.append(
                f"{row.get('qa_id')}: پاسخ خالی است."
            )

        variant = qa_variant(row["qa_id"])
        expected_type = QUESTION_TYPE_BY_VARIANT[variant]
        if row.get("question_type") != expected_type:
            errors.append(
                f"{row['qa_id']}: question_type با variant "
                "تطابق ندارد."
            )

    for record_id, record_rows in per_record.items():
        if len(record_rows) != 3:
            errors.append(
                f"{record_id}: باید دقیقاً ۳ QA داشته باشد، "
                f"اما {len(record_rows)} دارد."
            )

        variants = {
            qa_variant(row["qa_id"])
            for row in record_rows
        }
        if variants != {"v1", "v2", "v3"}:
            errors.append(
                f"{record_id}: variantها کامل نیستند: "
                f"{sorted(variants)}"
            )

        original_splits = {
            row["split"]
            for row in record_rows
        }
        if len(original_splits) != 1:
            errors.append(
                f"{record_id}: original split ناسازگار است."
            )

    if errors:
        print("اعتبارسنجی ورودی ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)


def build_splits(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    train_original = [
        row for row in rows
        if row["split"] == "train"
    ]
    challenge_validation = [
        row for row in rows
        if row["split"] == "validation"
    ]
    challenge_holdout = [
        row for row in rows
        if row["split"] == "holdout"
    ]

    source_record_ids = sorted({
        row["source_record_id"]
        for row in train_original
    })

    if len(source_record_ids) != EXPECTED_TRAIN_SOURCE_RECORDS:
        raise RuntimeError(
            "تعداد source recordهای train باید "
            f"{EXPECTED_TRAIN_SOURCE_RECORDS} باشد، اما "
            f"{len(source_record_ids)} است."
        )

    sft_train_qa: list[dict[str, Any]] = []
    in_domain_eval_qa: list[dict[str, Any]] = []

    for row in sorted(
        train_original,
        key=lambda item: item["qa_id"],
    ):
        reserved = reserved_variant_for_record(
            row["source_record_id"]
        )

        if qa_variant(row["qa_id"]) == reserved:
            in_domain_eval_qa.append(row)
        else:
            sft_train_qa.append(row)

    return {
        "sft_train_qa": sft_train_qa,
        "in_domain_eval_qa": in_domain_eval_qa,
        "challenge_validation_qa": sorted(
            challenge_validation,
            key=lambda item: item["qa_id"],
        ),
        "challenge_holdout_qa": sorted(
            challenge_holdout,
            key=lambda item: item["qa_id"],
        ),
    }


def validate_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    train = splits["sft_train_qa"]
    in_domain = splits["in_domain_eval_qa"]
    challenge_validation = splits[
        "challenge_validation_qa"
    ]
    challenge_holdout = splits[
        "challenge_holdout_qa"
    ]

    expected_counts = {
        "sft_train_qa": 80,
        "in_domain_eval_qa": 40,
        "challenge_validation_qa": (
            EXPECTED_CHALLENGE_VALIDATION_QA
        ),
        "challenge_holdout_qa": (
            EXPECTED_CHALLENGE_HOLDOUT_QA
        ),
    }

    for name, expected in expected_counts.items():
        actual = len(splits[name])
        if actual != expected:
            errors.append(
                f"{name}: باید {expected} نمونه باشد، "
                f"اما {actual} است."
            )

    qa_id_sets = {
        name: {row["qa_id"] for row in split_rows}
        for name, split_rows in splits.items()
    }

    names = list(qa_id_sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = qa_id_sets[first] & qa_id_sets[second]
            if overlap:
                errors.append(
                    f"هم‌پوشانی qa_id میان {first} و {second}: "
                    f"{sorted(overlap)}"
                )

    train_records = {
        row["source_record_id"]
        for row in train
    }
    in_domain_records = {
        row["source_record_id"]
        for row in in_domain
    }
    challenge_validation_records = {
        row["source_record_id"]
        for row in challenge_validation
    }
    challenge_holdout_records = {
        row["source_record_id"]
        for row in challenge_holdout
    }

    if train_records != in_domain_records:
        errors.append(
            "source recordهای train و in-domain eval باید "
            "دقیقاً یکسان باشند."
        )

    if train_records & challenge_validation_records:
        errors.append(
            "نشت source record میان train و challenge validation."
        )

    if train_records & challenge_holdout_records:
        errors.append(
            "نشت source record میان train و challenge holdout."
        )

    if (
        challenge_validation_records
        & challenge_holdout_records
    ):
        errors.append(
            "نشت source record میان challenge validation "
            "و challenge holdout."
        )

    per_train_record = Counter(
        row["source_record_id"]
        for row in train
    )
    per_eval_record = Counter(
        row["source_record_id"]
        for row in in_domain
    )

    if set(per_train_record.values()) != {2}:
        errors.append(
            "هر source record باید دقیقاً ۲ نمونه train داشته باشد."
        )

    if set(per_eval_record.values()) != {1}:
        errors.append(
            "هر source record باید دقیقاً ۱ نمونه in-domain eval "
            "داشته باشد."
        )

    in_domain_types = Counter(
        row["question_type"]
        for row in in_domain
    )

    if min(in_domain_types.values()) < 12:
        warnings.append(
            "توزیع question_type در in-domain eval "
            "کاملاً متوازن نیست."
        )

    report = {
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            name: len(split_rows)
            for name, split_rows in splits.items()
        },
        "source_record_counts": {
            "train": len(train_records),
            "in_domain_eval": len(in_domain_records),
            "challenge_validation": len(
                challenge_validation_records
            ),
            "challenge_holdout": len(
                challenge_holdout_records
            ),
        },
        "question_type_counts": {
            name: dict(sorted(Counter(
                row["question_type"]
                for row in split_rows
            ).items()))
            for name, split_rows in splits.items()
        },
        "evaluation_design": {
            "in_domain_eval": (
                "همان facts موجود در train با صورت سؤال رزروشده؛ "
                "برای سنجش یادگیری facts و تعمیم به paraphrase."
            ),
            "challenge_validation": (
                "concept group دیده‌نشده در train؛ "
                "برای تنظیمات کم‌ریسک و سنجش انتقال."
            ),
            "challenge_holdout": (
                "concept group کاملاً دیده‌نشده و دست‌نخورده؛ "
                "فقط برای ارزیابی نهایی."
            ),
        },
        "important_note": (
            "نسخه قبلی هر سه variant یک source record را در train "
            "می‌گذاشت و فقط conceptهای کاملاً دیده‌نشده را ارزیابی "
            "می‌کرد. این نسخه دو variant را train و یک variant را "
            "برای in-domain eval رزرو می‌کند؛ challenge splitها نیز "
            "بدون نشت source record باقی می‌مانند."
        ),
    }

    if errors:
        print("اعتبارسنجی Splitها ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    return report


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)

    validate_source(rows)
    splits = build_splits(rows)
    report = validate_splits(splits)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qa_dir = args.output_dir / "qa"
    sft_dir = args.output_dir / "sft"
    qa_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)

    qa_paths = {
        "in_domain_eval_qa": (
            qa_dir / "qa_3_3_in_domain_eval_40.jsonl"
        ),
        "challenge_validation_qa": (
            qa_dir / "qa_3_3_challenge_validation_12.jsonl"
        ),
        "challenge_holdout_qa": (
            qa_dir / "qa_3_3_challenge_holdout_15.jsonl"
        ),
    }

    for key, path in qa_paths.items():
        write_jsonl(path, splits[key])

    sft_train = [
        to_sft_row(row, "train")
        for row in splits["sft_train_qa"]
    ]
    sft_in_domain_eval = [
        to_sft_row(row, "in_domain_eval")
        for row in splits["in_domain_eval_qa"]
    ]

    sft_train_path = (
        sft_dir / "sft_3_3_train_80.jsonl"
    )
    sft_eval_path = (
        sft_dir / "sft_3_3_in_domain_eval_40.jsonl"
    )

    write_jsonl(sft_train_path, sft_train)
    write_jsonl(sft_eval_path, sft_in_domain_eval)

    audit_path = (
        args.output_dir / "pilot_split_audit.json"
    )
    audit_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 76)
    print("Pilot splitها ساخته و اعتبارسنجی شدند.")
    print("SFT train:", sft_train_path)
    print("SFT in-domain eval:", sft_eval_path)
    print(
        "Challenge validation:",
        qa_paths["challenge_validation_qa"],
    )
    print(
        "Challenge holdout:",
        qa_paths["challenge_holdout_qa"],
    )
    print("Audit:", audit_path)
    print()
    print("Counts:", report["counts"])
    print(
        "In-domain question types:",
        report["question_type_counts"][
            "in_domain_eval_qa"
        ],
    )
    print("Errors:", report["error_count"])
    print("Warnings:", report["warning_count"])
    print("=" * 76)


if __name__ == "__main__":
    main()
