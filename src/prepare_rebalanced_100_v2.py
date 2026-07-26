import json
import random
from pathlib import Path


POSITIVE_PATHS = [
    Path(
        "data/training/main/batch_001/"
        "chain_transfer_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_002/"
        "molecular_weight_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_003/"
        "thermal_behavior_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_004/"
        "core_polymer_concepts_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_005/"
        "generalization_verified_fa.jsonl"
    ),
]

GROUNDING_PATH = Path(
    "data/training/main/batch_006/"
    "source_grounding_verified_fa.jsonl"
)

HOLDOUT_PATH = Path(
    "data/training/verified_120/"
    "holdout_new_15.jsonl"
)

OUTPUT_DIR = Path(
    "data/training/rebalanced_100_v2"
)

POOL_PATH = OUTPUT_DIR / "polymer_tutor_rebalanced_100.jsonl"
TRAIN_PATH = OUTPUT_DIR / "train_rebalanced_90.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "validation_rebalanced_10.jsonl"


UNIFIED_SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم پلیمر هستی. "
    "فقط براساس متن منبع ارائه‌شده پاسخ بده. "
    "اگر متن منبع پاسخ سؤال را پشتیبانی می‌کند، "
    "پاسخ علمی را مستقیم و کامل ارائه کن. "
    "اگر اطلاعات لازم واقعاً در متن وجود ندارد، "
    "صریحاً بگو منبع کافی نیست و حدس نزن. "
    "نمادها، روابط و واحدهای علمی را دقیق حفظ کن."
)


SELECTED_GROUNDING_IDS = {
    "ground_001",
    "ground_005",
    "ground_009",
    "ground_012",
    "ground_015",
    "ground_016",
    "ground_020",
    "ground_024",
    "ground_027",
    "ground_029",
}


VALIDATION_IDS = {
    # هشت نمونه مثبت
    "ct_003",
    "mw_003",
    "th_004",
    "core_011",
    "core_024",
    "gen_010",
    "gen_014",
    "gen_025",

    # دو نمونه ناکافی‌بودن منبع
    "ground_020",
    "ground_029",
}


SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")

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


def normalize_sample(
    sample: dict,
    behavior: str,
) -> dict:
    messages = sample.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(
            f"ساختار پیام نمونه "
            f"{sample.get('metadata', {}).get('id')} نامعتبر است."
        )

    roles = [message.get("role") for message in messages]

    if roles != ["system", "user", "assistant"]:
        raise ValueError(
            f"ترتیب نقش‌ها در نمونه "
            f"{sample.get('metadata', {}).get('id')} نادرست است."
        )

    normalized = json.loads(
        json.dumps(sample, ensure_ascii=False)
    )

    normalized["messages"][0]["content"] = (
        UNIFIED_SYSTEM_PROMPT
    )

    normalized["metadata"]["dataset_version"] = (
        "rebalanced_100_v2"
    )

    normalized["metadata"]["grounding_behavior"] = behavior

    return normalized


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


def main() -> None:
    positive_samples = []

    for path in POSITIVE_PATHS:
        loaded = load_jsonl(path)

        print(f"{path}: {len(loaded)} نمونه مثبت")

        positive_samples.extend(
            normalize_sample(
                sample,
                behavior="answer_from_context",
            )
            for sample in loaded
        )

    if len(positive_samples) != 90:
        raise ValueError(
            f"تعداد نمونه‌های مثبت باید ۹۰ باشد، "
            f"اما {len(positive_samples)} است."
        )

    all_grounding_samples = load_jsonl(
        GROUNDING_PATH
    )

    selected_grounding_samples = []

    for sample in all_grounding_samples:
        sample_id = sample["metadata"]["id"]

        if sample_id in SELECTED_GROUNDING_IDS:
            selected_grounding_samples.append(
                normalize_sample(
                    sample,
                    behavior="insufficient_context",
                )
            )

    selected_grounding_ids = {
        sample["metadata"]["id"]
        for sample in selected_grounding_samples
    }

    missing_grounding_ids = (
        SELECTED_GROUNDING_IDS
        - selected_grounding_ids
    )

    if missing_grounding_ids:
        raise ValueError(
            "نمونه‌های Grounding پیدا نشدند: "
            f"{sorted(missing_grounding_ids)}"
        )

    if len(selected_grounding_samples) != 10:
        raise ValueError(
            f"تعداد نمونه‌های Grounding باید ۱۰ باشد، "
            f"اما {len(selected_grounding_samples)} است."
        )

    pool_samples = (
        positive_samples
        + selected_grounding_samples
    )

    if len(pool_samples) != 100:
        raise ValueError(
            f"تعداد کل باید ۱۰۰ باشد، "
            f"اما {len(pool_samples)} است."
        )

    all_ids = [
        sample["metadata"]["id"]
        for sample in pool_samples
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "شناسه تکراری در دیتاست پیدا شد."
        )

    holdout_samples = load_jsonl(
        HOLDOUT_PATH
    )

    holdout_ids = {
        sample["metadata"]["id"]
        for sample in holdout_samples
    }

    if set(all_ids) & holdout_ids:
        raise ValueError(
            "Holdout با دیتاست آموزش هم‌پوشانی دارد."
        )

    missing_validation_ids = (
        VALIDATION_IDS - set(all_ids)
    )

    if missing_validation_ids:
        raise ValueError(
            "شناسه‌های Validation پیدا نشدند: "
            f"{sorted(missing_validation_ids)}"
        )

    train_samples = []
    validation_samples = []

    for sample in pool_samples:
        sample_id = sample["metadata"]["id"]

        if sample_id in VALIDATION_IDS:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    if len(train_samples) != 90:
        raise ValueError(
            f"تعداد Train باید ۹۰ باشد، "
            f"اما {len(train_samples)} است."
        )

    if len(validation_samples) != 10:
        raise ValueError(
            f"تعداد Validation باید ۱۰ باشد، "
            f"اما {len(validation_samples)} است."
        )

    train_negative_count = sum(
        sample["metadata"]["grounding_behavior"]
        == "insufficient_context"
        for sample in train_samples
    )

    validation_negative_count = sum(
        sample["metadata"]["grounding_behavior"]
        == "insufficient_context"
        for sample in validation_samples
    )

    random_generator = random.Random(SEED)
    random_generator.shuffle(train_samples)
    random_generator.shuffle(validation_samples)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        POOL_PATH,
        pool_samples,
    )

    write_jsonl(
        TRAIN_PATH,
        train_samples,
    )

    write_jsonl(
        VALIDATION_PATH,
        validation_samples,
    )

    print("\n" + "=" * 72)
    print("دیتاست اصلاح‌شده ساخته شد.")
    print("کل نمونه‌ها:", len(pool_samples))
    print("نمونه‌های مثبت:", len(positive_samples))
    print(
        "نمونه‌های منبع ناکافی:",
        len(selected_grounding_samples),
    )

    print("\nTrain:")
    print("- تعداد کل:", len(train_samples))
    print("- پاسخ علمی:", len(train_samples) - train_negative_count)
    print("- منبع ناکافی:", train_negative_count)

    print("\nValidation:")
    print("- تعداد کل:", len(validation_samples))
    print(
        "- پاسخ علمی:",
        len(validation_samples) - validation_negative_count,
    )
    print("- منبع ناکافی:", validation_negative_count)

    print("\nSystem Prompt همه نمونه‌ها یکسان شد.")
    print("Holdout پانزده‌تایی دست‌نخورده باقی ماند.")


if __name__ == "__main__":
    main()
