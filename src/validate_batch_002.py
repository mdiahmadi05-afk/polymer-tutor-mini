import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(
    "data/training/main/batch_002/"
    "molecular_weight_verified_fa.jsonl"
)

EXPECTED_IDS = {
    f"mw_{number:03d}"
    for number in range(1, 11)
}

REQUIRED_ROLES = [
    "system",
    "user",
    "assistant",
]

REQUIRED_METADATA_FIELDS = {
    "id",
    "topic",
    "subtopic",
    "type",
    "difficulty",
    "language",
    "verified",
    "source",
}

REQUIRED_SOURCE_FIELDS = {
    "book",
    "author",
    "edition",
    "printed_pages",
}

FORBIDDEN_ASSISTANT_PHRASES = [
    "Mw مجموع وزن مولکولی همه زنجیرها بر تعداد زنجیرها",
    "Mw از تقسیم مجموع وزن مولکولی همه زنجیرها بر تعداد زنجیرها",
    "Mn میانگین تعداد مول‌ها",
    "η = K * t",
    "گرونایزن",
    "Grüneisen",
]

REQUIRED_ANSWER_CONTENT = {
    "mw_001": [
        "Mn = ΣNiMi / ΣNi",
    ],
    "mw_002": [
        "Mw = ΣNiMi² / ΣNiMi",
    ],
    "mw_003": [
        "Mw ≥ Mn",
    ],
    "mw_004": [
        "Đ = Mw/Mn",
        "1",
    ],
    "mw_005": [
        "20000 گرم بر مول",
    ],
    "mw_006": [
        "30000 گرم بر مول",
    ],
    "mw_007": [
        "[η] = K Mv^a",
        "1 دسی‌لیتر بر گرم",
    ],
    "mw_008": [
        "پلیمر",
        "حلال",
        "دما",
    ],
    "mw_009": [
        "Mn ≤ Mv ≤ Mw",
    ],
    "mw_010": [
        "Mw = ΣNiMi² / ΣNiMi",
    ],
}


def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل دیتاست پیدا نشد: {path}"
        )

    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: {error}"
                ) from error

            sample["_line_number"] = line_number
            samples.append(sample)

    return samples


def validate_sample(
    sample: dict,
    seen_ids: set[str],
) -> list[str]:
    errors = []
    line_number = sample["_line_number"]

    messages = sample.get("messages")

    if not isinstance(messages, list):
        return [
            f"خط {line_number}: messages باید لیست باشد."
        ]

    if len(messages) != 3:
        errors.append(
            f"خط {line_number}: تعداد پیام‌ها باید دقیقاً ۳ باشد."
        )
        return errors

    roles = [
        message.get("role")
        for message in messages
    ]

    if roles != REQUIRED_ROLES:
        errors.append(
            f"خط {line_number}: ترتیب نقش‌ها نادرست است: {roles}"
        )

    for index, message in enumerate(messages, start=1):
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            errors.append(
                f"خط {line_number}: پیام شماره {index} خالی است."
            )

    metadata = sample.get("metadata")

    if not isinstance(metadata, dict):
        errors.append(
            f"خط {line_number}: metadata نامعتبر است."
        )
        return errors

    missing_metadata = (
        REQUIRED_METADATA_FIELDS - metadata.keys()
    )

    if missing_metadata:
        errors.append(
            f"خط {line_number}: فیلدهای metadata ناقص‌اند: "
            f"{sorted(missing_metadata)}"
        )

    sample_id = metadata.get("id")

    if not isinstance(sample_id, str):
        errors.append(
            f"خط {line_number}: شناسه نمونه نامعتبر است."
        )
    elif sample_id in seen_ids:
        errors.append(
            f"خط {line_number}: شناسه تکراری است: {sample_id}"
        )
    else:
        seen_ids.add(sample_id)

    if metadata.get("verified") is not True:
        errors.append(
            f"خط {line_number}: verified باید True باشد."
        )

    if metadata.get("language") != "fa":
        errors.append(
            f"خط {line_number}: language باید fa باشد."
        )

    if metadata.get("subtopic") != "molecular_weight_averages":
        errors.append(
            f"خط {line_number}: subtopic نامعتبر است."
        )

    source = metadata.get("source")

    if not isinstance(source, dict):
        errors.append(
            f"خط {line_number}: source نامعتبر است."
        )
    else:
        missing_source = (
            REQUIRED_SOURCE_FIELDS - source.keys()
        )

        if missing_source:
            errors.append(
                f"خط {line_number}: فیلدهای source ناقص‌اند: "
                f"{sorted(missing_source)}"
            )

        pages = source.get("printed_pages")

        if not isinstance(pages, list) or not pages:
            errors.append(
                f"خط {line_number}: printed_pages باید "
                "یک لیست غیرخالی باشد."
            )
        elif not all(
            isinstance(page, int) and page > 0
            for page in pages
        ):
            errors.append(
                f"خط {line_number}: شماره صفحه نامعتبر است."
            )

    assistant_text = messages[2].get("content", "")

    for phrase in FORBIDDEN_ASSISTANT_PHRASES:
        if phrase.lower() in assistant_text.lower():
            errors.append(
                f"خط {line_number}: عبارت علمی ممنوع "
                f"پیدا شد: «{phrase}»"
            )

    required_items = REQUIRED_ANSWER_CONTENT.get(
        sample_id,
        [],
    )

    for required_item in required_items:
        if required_item not in assistant_text:
            errors.append(
                f"خط {line_number}: در پاسخ {sample_id} "
                f"عبارت لازم وجود ندارد: «{required_item}»"
            )

    return errors


def main() -> None:
    samples = load_samples(DATASET_PATH)

    seen_ids: set[str] = set()
    all_errors = []

    for sample in samples:
        all_errors.extend(
            validate_sample(
                sample=sample,
                seen_ids=seen_ids,
            )
        )

    actual_ids = {
        sample.get("metadata", {}).get("id")
        for sample in samples
    }

    if len(samples) != 10:
        all_errors.append(
            f"تعداد نمونه‌ها باید ۱۰ باشد، اما {len(samples)} است."
        )

    missing_ids = EXPECTED_IDS - actual_ids
    unexpected_ids = actual_ids - EXPECTED_IDS

    if missing_ids:
        all_errors.append(
            f"شناسه‌های مفقود: {sorted(missing_ids)}"
        )

    if unexpected_ids:
        all_errors.append(
            f"شناسه‌های غیرمنتظره: {sorted(unexpected_ids)}"
        )

    type_counts = Counter(
        sample["metadata"]["type"]
        for sample in samples
        if isinstance(sample.get("metadata"), dict)
        and "type" in sample["metadata"]
    )

    print("=" * 72)
    print("فایل:", DATASET_PATH)
    print("تعداد نمونه‌ها:", len(samples))

    print("\nتوزیع نوع نمونه‌ها:")

    for sample_type, count in sorted(type_counts.items()):
        print(f"- {sample_type}: {count}")

    print("\nنتیجه اعتبارسنجی:")

    if all_errors:
        print(f"تعداد خطاها: {len(all_errors)}")

        for error in all_errors:
            print("-", error)

        raise SystemExit(1)

    print("همه نمونه‌ها معتبر هستند.")
    print("فرمول‌های Mn، Mw، Đ و مارک–هووینک صحیح‌اند.")
    print("محاسبات عددی نمونه‌ها صحیح ثبت شده‌اند.")
    print("شناسه تکراری یا عبارت علمی ممنوع وجود ندارد.")


if __name__ == "__main__":
    main()
