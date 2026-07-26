import json
import random
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "data/training/main/polymer_tutor_verified_30.jsonl"
)

OUTPUT_DIR = Path(
    "data/training/main/splits"
)

TRAIN_PATH = OUTPUT_DIR / "train_verified_24.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "validation_verified_6.jsonl"


# دو نمونه از هر موضوع برای Validation نگه داشته می‌شوند.
VALIDATION_IDS = {
    "ct_002",
    "ct_006",
    "mw_004",
    "mw_007",
    "th_002",
    "th_009",
}

EXPECTED_ROLES = [
    "system",
    "user",
    "assistant",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد: {path}"
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

            samples.append(sample)

    return samples


def validate_samples(samples: list[dict]) -> None:
    if len(samples) != 30:
        raise ValueError(
            f"تعداد نمونه‌های ورودی باید ۳۰ باشد، "
            f"اما {len(samples)} نمونه پیدا شد."
        )

    seen_ids = set()

    for index, sample in enumerate(samples, start=1):
        metadata = sample.get("metadata", {})
        sample_id = metadata.get("id")
        messages = sample.get("messages")

        if not sample_id:
            raise ValueError(
                f"نمونه شماره {index} فاقد شناسه است."
            )

        if sample_id in seen_ids:
            raise ValueError(
                f"شناسه تکراری پیدا شد: {sample_id}"
            )

        seen_ids.add(sample_id)

        if not isinstance(messages, list):
            raise ValueError(
                f"messages در نمونه {sample_id} نامعتبر است."
            )

        roles = [
            message.get("role")
            for message in messages
        ]

        if roles != EXPECTED_ROLES:
            raise ValueError(
                f"ترتیب نقش‌ها در نمونه {sample_id} نادرست است: "
                f"{roles}"
            )

        if metadata.get("verified") is not True:
            raise ValueError(
                f"نمونه {sample_id} تأییدشده نیست."
            )

    missing_validation_ids = VALIDATION_IDS - seen_ids

    if missing_validation_ids:
        raise ValueError(
            "شناسه‌های Validation در دیتاست پیدا نشدند: "
            f"{sorted(missing_validation_ids)}"
        )


def write_jsonl(
    path: Path,
    samples: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_distribution(
    title: str,
    samples: list[dict],
) -> None:
    topic_counts = Counter(
        sample["metadata"]["subtopic"]
        for sample in samples
    )

    print(f"\n{title}:")

    for topic, count in sorted(topic_counts.items()):
        print(f"- {topic}: {count}")


def main() -> None:
    samples = load_jsonl(INPUT_PATH)
    validate_samples(samples)

    train_samples = []
    validation_samples = []

    for sample in samples:
        sample_id = sample["metadata"]["id"]

        if sample_id in VALIDATION_IDS:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    if len(train_samples) != 24:
        raise ValueError(
            f"تعداد Train باید ۲۴ باشد، "
            f"اما {len(train_samples)} است."
        )

    if len(validation_samples) != 6:
        raise ValueError(
            f"تعداد Validation باید ۶ باشد، "
            f"اما {len(validation_samples)} است."
        )

    train_ids = {
        sample["metadata"]["id"]
        for sample in train_samples
    }

    validation_ids = {
        sample["metadata"]["id"]
        for sample in validation_samples
    }

    overlap = train_ids & validation_ids

    if overlap:
        raise ValueError(
            f"هم‌پوشانی بین Train و Validation: {sorted(overlap)}"
        )

    random_generator = random.Random(42)
    random_generator.shuffle(train_samples)
    random_generator.shuffle(validation_samples)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        TRAIN_PATH,
        train_samples,
    )

    write_jsonl(
        VALIDATION_PATH,
        validation_samples,
    )

    print("=" * 72)
    print("تقسیم دیتاست با موفقیت انجام شد.")
    print("Train:", TRAIN_PATH)
    print("تعداد Train:", len(train_samples))
    print("Validation:", VALIDATION_PATH)
    print("تعداد Validation:", len(validation_samples))
    print("هم‌پوشانی Train و Validation:", len(overlap))

    print_distribution(
        "توزیع Train",
        train_samples,
    )

    print_distribution(
        "توزیع Validation",
        validation_samples,
    )


if __name__ == "__main__":
    main()
