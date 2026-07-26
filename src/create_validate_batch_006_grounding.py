import json
import re
from collections import Counter
from pathlib import Path


INPUT_PATHS = [
    Path(
        "data/training/main/batch_004/"
        "core_polymer_concepts_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_005/"
        "generalization_verified_fa.jsonl"
    ),
]

OUTPUT_PATH = Path(
    "data/training/main/batch_006/"
    "source_grounding_verified_fa.jsonl"
)

SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم پلیمر هستی. "
    "فقط براساس متن منبع ارائه‌شده پاسخ بده. "
    "هنگامی که متن منبع اطلاعات لازم را ندارد، "
    "صریحاً بگو منبع برای پاسخ کافی نیست. "
    "در این حالت از حافظه، حدس یا دانش بیرونی استفاده نکن."
)

ANSWER_TEMPLATES = [
    (
        "متن منبع برای پاسخ دقیق به این سؤال کافی نیست؛ "
        "بنابراین نباید پاسخ علمی را حدس زد."
    ),
    (
        "براساس متن منبع نمی‌توان به این سؤال پاسخ داد. "
        "اطلاعات مرتبط در منبع ارائه نشده است."
    ),
    (
        "منبع ارائه‌شده برای پاسخ کافی نیست. "
        "برای پاسخ معتبر باید متن مرتبط دیگری بازیابی شود."
    ),
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
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در {path}، "
                    f"خط {line_number}: {error}"
                ) from error

    return samples


def split_user_content(
    user_content: str,
) -> tuple[str, str]:
    source_marker = "متن منبع:\n"
    question_marker = "\n\nسؤال:\n"

    if source_marker not in user_content:
        raise ValueError(
            "عبارت «متن منبع» در پیام کاربر پیدا نشد."
        )

    if question_marker not in user_content:
        raise ValueError(
            "عبارت «سؤال» در پیام کاربر پیدا نشد."
        )

    remaining = user_content.split(
        source_marker,
        1,
    )[1]

    source_text, question = remaining.split(
        question_marker,
        1,
    )

    return source_text.strip(), question.strip()


def normalize_words(text: str) -> set[str]:
    normalized = re.sub(
        r"[^\w\u0600-\u06FF]+",
        " ",
        text.lower(),
    )

    words = {
        word
        for word in normalized.split()
        if len(word) >= 3
    }

    stop_words = {
        "است",
        "هست",
        "شود",
        "شده",
        "برای",
        "این",
        "آن",
        "چه",
        "چگونه",
        "چرا",
        "دارد",
        "کنید",
        "کن",
        "را",
        "با",
        "در",
        "از",
        "به",
    }

    return words - stop_words


def overlap_score(
    question: str,
    context: str,
) -> float:
    question_words = normalize_words(question)
    context_words = normalize_words(context)

    if not question_words:
        return 0.0

    overlap = question_words & context_words

    return len(overlap) / len(question_words)


def extract_records(
    samples: list[dict],
) -> list[dict]:
    records = []

    for sample in samples:
        metadata = sample.get("metadata", {})
        messages = sample.get("messages", [])

        if len(messages) != 3:
            raise ValueError(
                f"ساختار پیام نمونه {metadata.get('id')} نامعتبر است."
            )

        context, question = split_user_content(
            messages[1]["content"]
        )

        records.append(
            {
                "id": metadata["id"],
                "question": question,
                "context": context,
                "subtopic": metadata["subtopic"],
                "source": metadata["source"],
            }
        )

    return records


def select_targets(
    records: list[dict],
) -> list[dict]:
    odian_records = [
        record
        for record in records
        if record["source"]["author"] == "George Odian"
    ]

    sperling_records = [
        record
        for record in records
        if record["source"]["author"] == "L. H. Sperling"
    ]

    if len(odian_records) < 15:
        raise ValueError(
            "حداقل ۱۵ نمونه Odian لازم است."
        )

    if len(sperling_records) < 15:
        raise ValueError(
            "حداقل ۱۵ نمونه Sperling لازم است."
        )

    return odian_records[:15] + sperling_records[:15]


def find_mismatched_context(
    target: dict,
    records: list[dict],
) -> tuple[dict, float]:
    target_author = target["source"]["author"]

    candidates = [
        record
        for record in records
        if record["source"]["author"] != target_author
        and record["subtopic"] != target["subtopic"]
    ]

    if not candidates:
        raise ValueError(
            f"منبع نامرتبط برای {target['id']} پیدا نشد."
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            overlap_score(
                target["question"],
                candidate["context"],
            ),
            candidate["id"],
        ),
    )

    best_candidate = ranked[0]

    score = overlap_score(
        target["question"],
        best_candidate["context"],
    )

    return best_candidate, score


def build_samples(
    records: list[dict],
) -> list[dict]:
    targets = select_targets(records)
    output_samples = []

    for index, target in enumerate(targets, start=1):
        context_record, score = find_mismatched_context(
            target=target,
            records=records,
        )

        output_samples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"متن منبع:\n"
                            f"{context_record['context']}\n\n"
                            f"سؤال:\n"
                            f"{target['question']}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": ANSWER_TEMPLATES[
                            (index - 1) % len(ANSWER_TEMPLATES)
                        ],
                    },
                ],
                "metadata": {
                    "id": f"ground_{index:03d}",
                    "topic": "polymer_science",
                    "subtopic": "source_grounding",
                    "type": "insufficient_context",
                    "difficulty": "intermediate",
                    "language": "fa",
                    "verified": True,
                    "behavior": "do_not_invent",
                    "question_origin_id": target["id"],
                    "context_origin_id": context_record["id"],
                    "question_subtopic": target["subtopic"],
                    "context_subtopic": context_record["subtopic"],
                    "lexical_overlap": round(score, 4),
                    "source": context_record["source"],
                },
            }
        )

    return output_samples


def validate(
    samples: list[dict],
) -> None:
    errors = []

    if len(samples) != 30:
        errors.append(
            f"تعداد نمونه‌ها باید ۳۰ باشد، "
            f"اما {len(samples)} است."
        )

    expected_ids = {
        f"ground_{number:03d}"
        for number in range(1, 31)
    }

    actual_ids = []
    question_origin_ids = []

    for sample in samples:
        metadata = sample.get("metadata", {})
        messages = sample.get("messages", [])
        sample_id = metadata.get("id")

        actual_ids.append(sample_id)
        question_origin_ids.append(
            metadata.get("question_origin_id")
        )

        if len(messages) != 3:
            errors.append(
                f"{sample_id}: تعداد پیام‌ها باید ۳ باشد."
            )
            continue

        roles = [
            message.get("role")
            for message in messages
        ]

        if roles != ["system", "user", "assistant"]:
            errors.append(
                f"{sample_id}: ترتیب نقش‌ها نامعتبر است."
            )

        assistant_text = messages[2].get(
            "content",
            "",
        )

        required_behavior_terms = [
            "منبع",
        ]

        if not all(
            term in assistant_text
            for term in required_behavior_terms
        ):
            errors.append(
                f"{sample_id}: پاسخ به منبع اشاره نکرده است."
            )

        if not any(
            phrase in assistant_text
            for phrase in [
                "کافی نیست",
                "نمی‌توان",
                "ارائه نشده",
            ]
        ):
            errors.append(
                f"{sample_id}: ناکافی‌بودن منبع روشن نشده است."
            )

        if metadata.get("verified") is not True:
            errors.append(
                f"{sample_id}: verified باید True باشد."
            )

        if metadata.get("language") != "fa":
            errors.append(
                f"{sample_id}: language باید fa باشد."
            )

        if metadata.get("behavior") != "do_not_invent":
            errors.append(
                f"{sample_id}: behavior نامعتبر است."
            )

        if (
            metadata.get("question_subtopic")
            == metadata.get("context_subtopic")
        ):
            errors.append(
                f"{sample_id}: سؤال و متن از یک زیرموضوع هستند."
            )

        overlap = metadata.get("lexical_overlap")

        if not isinstance(overlap, float):
            errors.append(
                f"{sample_id}: lexical_overlap نامعتبر است."
            )
        elif overlap > 0.35:
            errors.append(
                f"{sample_id}: هم‌پوشانی متن و سؤال زیاد است: "
                f"{overlap}"
            )

        source = metadata.get("source", {})
        pages = source.get("printed_pages")

        if (
            not isinstance(pages, list)
            or not pages
            or not all(
                isinstance(page, int) and page > 0
                for page in pages
            )
        ):
            errors.append(
                f"{sample_id}: اطلاعات صفحه نامعتبر است."
            )

    if set(actual_ids) != expected_ids:
        errors.append(
            "شناسه‌ها با ground_001 تا ground_030 "
            "مطابقت ندارند."
        )

    if len(actual_ids) != len(set(actual_ids)):
        errors.append(
            "شناسه تکراری وجود دارد."
        )

    if len(question_origin_ids) != len(
        set(question_origin_ids)
    ):
        errors.append(
            "یک سؤال اصلی بیش از یک‌بار استفاده شده است."
        )

    if errors:
        print("اعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def write_jsonl(
    samples: list[dict],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for sample in samples:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    all_samples = []

    for input_path in INPUT_PATHS:
        loaded = load_jsonl(input_path)

        print(
            f"{input_path}: "
            f"{len(loaded)} نمونه"
        )

        all_samples.extend(loaded)

    records = extract_records(all_samples)
    grounding_samples = build_samples(records)

    validate(grounding_samples)
    write_jsonl(grounding_samples)

    overlap_values = [
        sample["metadata"]["lexical_overlap"]
        for sample in grounding_samples
    ]

    author_counts = Counter(
        sample["metadata"]["source"]["author"]
        for sample in grounding_samples
    )

    print("\n" + "=" * 72)
    print("فایل ساخته شد:", OUTPUT_PATH)
    print("تعداد نمونه‌ها:", len(grounding_samples))
    print(
        "بیشترین هم‌پوشانی واژگانی:",
        max(overlap_values),
    )
    print(
        "میانگین هم‌پوشانی واژگانی:",
        round(
            sum(overlap_values) / len(overlap_values),
            4,
        ),
    )

    print("\nتوزیع منابع متن‌های نامرتبط:")

    for author, count in sorted(author_counts.items()):
        print(f"- {author}: {count}")

    print("\nنتیجه اعتبارسنجی:")
    print("همه ۳۰ نمونه معتبر هستند.")
    print("سؤال و متن منبع از زیرموضوع‌های متفاوت‌اند.")
    print("مدل باید ناکافی‌بودن منبع را اعلام کند.")
    print("پاسخ علمی حدسی در دیتاست وجود ندارد.")


if __name__ == "__main__":
    main()
