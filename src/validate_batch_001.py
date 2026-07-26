import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(
    "data/training/main/batch_001/"
    "chain_transfer_verified_fa.jsonl"
)

REQUIRED_ROLES = ["system", "user", "assistant"]

FORBIDDEN_PHRASES = [
    "بازنویسی",
    "بازنویس",
    "Rp همان وزن مولکولی",
    "Rp وزن مولکولی",
    "ktr ثابت انتقال",
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


def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")

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
        return [f"خط {line_number}: messages باید یک لیست باشد."]

    if len(messages) != 3:
        errors.append(
            f"خط {line_number}: تعداد پیام‌ها باید دقیقاً ۳ باشد."
        )
    else:
        roles = [message.get("role") for message in messages]

        if roles != REQUIRED_ROLES:
            errors.append(
                f"خط {line_number}: ترتیب نقش‌ها نادرست است: {roles}"
            )

    for message_index, message in enumerate(messages, start=1):
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            errors.append(
                f"خط {line_number}: محتوای پیام {message_index} خالی است."
            )

    metadata = sample.get("metadata")

    if not isinstance(metadata, dict):
        errors.append(
            f"خط {line_number}: metadata وجود ندارد یا نامعتبر است."
        )
        return errors

    missing_metadata = REQUIRED_METADATA_FIELDS - metadata.keys()

    if missing_metadata:
        errors.append(
            f"خط {line_number}: فیلدهای metadata ناقص‌اند: "
            f"{sorted(missing_metadata)}"
        )

    sample_id = metadata.get("id")

    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append(f"خط {line_number}: شناسه نمونه نامعتبر است.")
    elif sample_id in seen_ids:
        errors.append(
            f"خط {line_number}: شناسه تکراری پیدا شد: {sample_id}"
        )
    else:
        seen_ids.add(sample_id)

    if metadata.get("verified") is not True:
        errors.append(
            f"خط {line_number}: مقدار verified باید True باشد."
        )

    if metadata.get("language") != "fa":
        errors.append(
            f"خط {line_number}: زبان نمونه باید fa باشد."
        )

    source = metadata.get("source")

    if not isinstance(source, dict):
        errors.append(
            f"خط {line_number}: اطلاعات source نامعتبر است."
        )
    else:
        missing_source = REQUIRED_SOURCE_FIELDS - source.keys()

        if missing_source:
            errors.append(
                f"خط {line_number}: فیلدهای source ناقص‌اند: "
                f"{sorted(missing_source)}"
            )

        pages = source.get("printed_pages")

        if not isinstance(pages, list) or not pages:
            errors.append(
                f"خط {line_number}: printed_pages باید یک لیست غیرخالی باشد."
            )
        elif not all(
            isinstance(page, int) and page > 0
            for page in pages
        ):
            errors.append(
                f"خط {line_number}: شماره صفحات نامعتبر است."
            )

    assistant_text = (
        messages[2].get("content", "")
        if len(messages) >= 3
        else ""
    )

    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in assistant_text.lower():
            errors.append(
                f"خط {line_number}: عبارت ممنوع پیدا شد: «{phrase}»"
            )

    if "Rp" in assistant_text and "سرعت پلیمریزاسیون" not in assistant_text:
        errors.append(
            f"خط {line_number}: Rp آمده ولی به‌عنوان سرعت "
            "پلیمریزاسیون روشن نشده است."
        )

    if "ktr" in assistant_text and "ثابت سرعت" not in assistant_text:
        errors.append(
            f"خط {line_number}: ktr آمده ولی عبارت «ثابت سرعت» "
            "در پاسخ وجود ندارد."
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

    print("=" * 72)
    print("فایل:", DATASET_PATH)
    print("تعداد نمونه‌ها:", len(samples))

    type_counts = Counter(
        sample["metadata"]["type"]
        for sample in samples
        if isinstance(sample.get("metadata"), dict)
        and "type" in sample["metadata"]
    )

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
    print("شناسه تکراری وجود ندارد.")
    print("ساختار messages و metadata صحیح است.")
    print("عبارت علمی ممنوع یا ترجمه اشتباه پیدا نشد.")


if __name__ == "__main__":
    main()
