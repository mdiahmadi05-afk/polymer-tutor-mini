import csv
import json
from pathlib import Path

QUESTIONS_PATH = Path("data/evaluation/polymer_baseline_fa.jsonl")
RESULTS_PATH = Path("results/baseline_30_gemma3_v2.jsonl")
REFERENCES_PATH = Path("data/evaluation/reference_answers_fa.jsonl")
OUTPUT_PATH = Path("results/baseline_30_gemma3_v2_review.csv")


def read_jsonl(path: Path) -> dict[str, dict]:
    items = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            item_id = item["id"]

            if item_id in items:
                raise ValueError(f"Duplicate ID in {path}: {item_id}")

            items[item_id] = item

    return items


questions = read_jsonl(QUESTIONS_PATH)
results = read_jsonl(RESULTS_PATH)
references = read_jsonl(REFERENCES_PATH)

question_ids = set(questions)
result_ids = set(results)
reference_ids = set(references)

if question_ids != result_ids or question_ids != reference_ids:
    print("Missing model results:", sorted(question_ids - result_ids))
    print("Missing references:", sorted(question_ids - reference_ids))
    print("Unknown result IDs:", sorted(result_ids - question_ids))
    print("Unknown reference IDs:", sorted(reference_ids - question_ids))
    raise RuntimeError("IDs do not match.")

fields = [
    "id",
    "level",
    "type",
    "category",
    "prompt",
    "model_answer",
    "reference_answer",
    "key_points",
    "scientific_accuracy_0_5",
    "completeness_0_5",
    "clarity_0_5",
    "terminology_0_5",
    "relevance_0_5",
    "total_score_0_25",
    "errors_and_notes",
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for item_id in sorted(question_ids):
        question = questions[item_id]
        result = results[item_id]
        reference = references[item_id]

        writer.writerow(
            {
                "id": item_id,
                "level": question["level"],
                "type": question["type"],
                "category": question["category"],
                "prompt": question["prompt"],
                "model_answer": result["answer"],
                "reference_answer": reference["reference_answer"],
                "key_points": " | ".join(reference["key_points"]),
                "scientific_accuracy_0_5": "",
                "completeness_0_5": "",
                "clarity_0_5": "",
                "terminology_0_5": "",
                "relevance_0_5": "",
                "total_score_0_25": "",
                "errors_and_notes": "",
            }
        )

print(f"Review sheet created: {OUTPUT_PATH}")
print(f"Questions included: {len(question_ids)}")
