import json
import random
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "data/training/verified_120/"
    "polymer_tutor_verified_120.jsonl"
)

HOLDOUT_PATH = Path(
    "data/training/verified_120/"
    "holdout_new_15.jsonl"
)

OUTPUT_DIR = Path(
    "data/training/verified_120/splits"
)

TRAIN_PATH = OUTPUT_DIR / "train_verified_108.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "validation_verified_12.jsonl"

VALIDATION_IDS = {
    "ct_003",
    "ct_008",
    "mw_003",
    "mw_009",
    "th_004",
    "th_008",
    "core_006",
    "core_024",
    "gen_014",
    "gen_025",
    "ground_010",
    "ground_026",
}

EXPECTED_ROLES = [
    "system",
    "user",
    "assistant",
]

SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل پیدا نشد: {path}"
        )

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
                    f"JSON نامعتبر در {path}، "
                    f"خط {line_number}: {error}"
                ) from error

    return samples


def validate_sample(sample: dict) -> None:
    metadata = sample.get("metadata", {})
    sample_id = metadata.get("id")
    messages = sample.get("messages")

    if not sample_id:
        raise ValueError("نمونه بدون شناسه پیدا شد.")

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
            f"ترتیب نقش‌ها در {sample_id} نادرست است: {roles}"
        )

    if metadata.get("verified") is not True:
        raise ValueError(
            f"نمونه {sample_id} تأییدشده نیست."
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
    prefix_counts = Counter(
        sample["metadata"]["id"].split("_")[0]
        for sample in samples
    )

    type_counts = Counter(
        sample["metadata"]["type"]
        for sample in samples
    )

    print(f"\n{title} — پیشوند شناسه:")

    for prefix, count in sorted(prefix_counts.items()):
        print(f"- {prefix}: {count}")

    print(f"\n{title} — نوع نمونه:")

    for sample_type, count in sorted(type_counts.items()):
        print(f"- {sample_type}: {count}")


def main() -> None:
    samples = load_jsonl(INPUT_PATH)
    holdout_samples = load_jsonl(HOLDOUT_PATH)

    if len(samples) != 120:
        raise ValueError(
            f"تعداد ورودی باید ۱۲۰ باشد، "
            f"اما {len(samples)} است."
        )

    if len(holdout_samples) != 15:
        raise ValueError(
            f"تعداد Holdout باید ۱۵ باشد، "
            f"اما {len(holdout_samples)} است."
        )

    all_ids = set()

    for sample in samples:
        validate_sample(sample)

        sample_id = sample["metadata"]["id"]

        if sample_id in all_ids:
            raise ValueError(
                f"شناسه تکراری پیدا شد: {sample_id}"
            )

        all_ids.add(sample_id)

    missing_validation_ids = (
        VALIDATION_IDS - all_ids
    )

    if missing_validation_ids:
        raise ValueError(
            "شناسه‌های Validation پیدا نشدند: "
            f"{sorted(missing_validation_ids)}"
        )

    train_samples = []
    validation_samples = []

    for sample in samples:
        sample_id = sample["metadata"]["id"]

        if sample_id in VALIDATION_IDS:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    if len(train_samples) != 108:
        raise ValueError(
            f"تعداد Train باید ۱۰۸ باشد، "
            f"اما {len(train_samples)} است."
        )

    if len(validation_samples) != 12:
        raise ValueError(
            f"تعداد Validation باید ۱۲ باشد، "
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

    holdout_ids = {
        sample["metadata"]["id"]
        for sample in holdout_samples
    }

    if train_ids & validation_ids:
        raise ValueError(
            "Train و Validation هم‌پوشانی دارند."
        )

    if (train_ids | validation_ids) & holdout_ids:
        raise ValueError(
            "Holdout با داده‌های آموزش هم‌پوشانی دارد."
        )

    random_generator = random.Random(SEED)
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
    print("Holdout دست‌نخورده:", len(holdout_samples))
    print("هم‌پوشانی با Holdout: 0")

    print_distribution(
        "Train",
        train_samples,
    )

    print_distribution(
        "Validation",
        validation_samples,
    )


if __name__ == "__main__":
    main()
