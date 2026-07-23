import csv
import json
from pathlib import Path

INPUT_PATH = Path("results/baseline_30_gemma3_v2.jsonl")
OUTPUT_PATH = Path("results/baseline_30_gemma3_v2_scores.csv")

fields = [
    "id",
    "level",
    "type",
    "category",
    "prompt",
    "answer",
    "scientific_accuracy",
    "completeness",
    "clarity",
    "terminology",
    "relevance",
    "total_score",
    "errors_and_notes",
]

rows = []

with INPUT_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        item = json.loads(line)

        rows.append(
            {
                "id": item["id"],
                "level": item["level"],
                "type": item["type"],
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "scientific_accuracy": "",
                "completeness": "",
                "clarity": "",
                "terminology": "",
                "relevance": "",
                "total_score": "",
                "errors_and_notes": "",
            }
        )

with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Scoring sheet created: {OUTPUT_PATH}")
print(f"Questions included: {len(rows)}")
