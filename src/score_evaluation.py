import csv
from pathlib import Path

CSV_PATH = Path("results/baseline_30_gemma3_v2_review.csv")

SCORE_FIELDS = [
    "scientific_accuracy_0_5",
    "completeness_0_5",
    "clarity_0_5",
    "terminology_0_5",
    "relevance_0_5",
]


def read_score(label: str) -> int:
    while True:
        value = input(f"{label} [0 تا 5]: ").strip()

        if value.lower() == "q":
            raise KeyboardInterrupt

        try:
            score = int(value)
        except ValueError:
            print("فقط یک عدد از 0 تا 5 وارد کن.")
            continue

        if 0 <= score <= 5:
            return score

        print("نمره باید بین 0 و 5 باشد.")


with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

if not fieldnames:
    raise RuntimeError("ستون‌های فایل CSV پیدا نشدند.")

for index, row in enumerate(rows, start=1):
    if row["total_score_0_25"].strip():
        continue

    print("\n" + "=" * 90)
    print(f"سؤال {index} از {len(rows)} — {row['id']}")
    print("=" * 90)

    print("\nسؤال:")
    print(row["prompt"])

    print("\nپاسخ Gemma:")
    print(row["model_answer"])

    print("\nپاسخ مرجع:")
    print(row["reference_answer"])

    print("\nنکات کلیدی:")
    print(row["key_points"])

    print("\nنمره‌دهی — برای خروج و ادامه در آینده، q وارد کن.")

    try:
        scores = {
            "scientific_accuracy_0_5": read_score("دقت علمی"),
            "completeness_0_5": read_score("کامل‌بودن"),
            "clarity_0_5": read_score("وضوح"),
            "terminology_0_5": read_score("اصطلاحات علمی"),
            "relevance_0_5": read_score("تناسب با سؤال"),
        }
    except KeyboardInterrupt:
        print("\nنمره‌دهی متوقف شد؛ نمره‌های قبلی ذخیره مانده‌اند.")
        break

    row.update({key: str(value) for key, value in scores.items()})
    row["total_score_0_25"] = str(sum(scores.values()))
    row["errors_and_notes"] = input("خطاها و یادداشت‌ها: ").strip()

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"ذخیره شد — امتیاز کل: {row['total_score_0_25']}/25")

print(f"\nفایل نمره‌دهی: {CSV_PATH}")
