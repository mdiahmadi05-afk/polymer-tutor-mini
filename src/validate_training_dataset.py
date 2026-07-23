import json
from collections import Counter
from pathlib import Path

path = Path("data/training/pilot_train_fa.jsonl")
required_roles = ["system", "user", "assistant"]

samples = []
questions = []

with path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"خط {line_number}: JSON نامعتبر است: {error}") from error

        messages = sample.get("messages")
        metadata = sample.get("metadata")

        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"خط {line_number}: باید دقیقاً سه پیام داشته باشد.")

        roles = [message.get("role") for message in messages]
        if roles != required_roles:
            raise ValueError(
                f"خط {line_number}: ترتیب نقش‌ها باید system، user و assistant باشد."
            )

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"خط {line_number}: محتوای یکی از پیام‌ها خالی است.")

        if not isinstance(metadata, dict):
            raise ValueError(f"خط {line_number}: metadata وجود ندارد.")

        for field in ["topic", "type", "difficulty"]:
            if not metadata.get(field):
                raise ValueError(f"خط {line_number}: فیلد {field} ناقص است.")

        samples.append(sample)
        questions.append(messages[1]["content"].strip())

duplicates = [
    question for question, count in Counter(questions).items()
    if count > 1
]

if duplicates:
    raise ValueError(f"سؤال تکراری پیدا شد: {duplicates}")

topic_counts = Counter(
    sample["metadata"]["topic"] for sample in samples
)

type_counts = Counter(
    sample["metadata"]["type"] for sample in samples
)

print(f"تعداد کل نمونه‌ها: {len(samples)}")
print(f"تعداد سؤال‌های تکراری: {len(duplicates)}")

print("\nتعداد نمونه‌ها بر اساس موضوع:")
for topic, count in sorted(topic_counts.items()):
    print(f"  {topic}: {count}")

print("\nتعداد نمونه‌ها بر اساس نوع سؤال:")
for sample_type, count in sorted(type_counts.items()):
    print(f"  {sample_type}: {count}")

print("\nنتیجه: ساختار دیتاست سالم است.")
