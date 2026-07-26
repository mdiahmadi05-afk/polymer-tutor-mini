import json
from pathlib import Path


INPUT_PATHS = [
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
]

OUTPUT_PATH = Path(
    "data/training/main/"
    "polymer_tutor_verified_30.jsonl"
)


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
                    f"JSON نامعتبر در فایل {path}، "
                    f"خط {line_number}: {error}"
                ) from error

    return samples


def main() -> None:
    all_samples = []
    seen_ids = set()

    for path in INPUT_PATHS:
        batch_samples = load_jsonl(path)

        print(
            f"{path}: "
            f"{len(batch_samples)} نمونه"
        )

        for sample in batch_samples:
            sample_id = sample.get(
                "metadata",
                {},
            ).get("id")

            if not sample_id:
                raise ValueError(
                    f"نمونه بدون شناسه در فایل {path}"
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"شناسه تکراری پیدا شد: {sample_id}"
                )

            seen_ids.add(sample_id)
            all_samples.append(sample)

    if len(all_samples) != 30:
        raise ValueError(
            f"تعداد کل باید ۳۰ باشد، "
            f"اما {len(all_samples)} نمونه پیدا شد."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for sample in all_samples:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 72)
    print("فایل نهایی ساخته شد:", OUTPUT_PATH)
    print("تعداد کل نمونه‌ها:", len(all_samples))
    print("تعداد شناسه‌های یکتا:", len(seen_ids))


if __name__ == "__main__":
    main()
