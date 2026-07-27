import csv
import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "data/knowledge/odian_ch3/source/"
    "section_3_3_source_chunks.jsonl"
)

OUTPUT_DIR = Path(
    "data/knowledge/odian_ch3/review"
)

CSV_PATH = OUTPUT_DIR / "knowledge_review_queue_3_3.csv"
JSONL_PATH = OUTPUT_DIR / "knowledge_review_queue_3_3.jsonl"


CSV_FIELDS = [
    "review_id",
    "chunk_id",
    "book",
    "chapter",
    "section",
    "subsection",
    "printed_page",
    "pdf_page_1_based",
    "equation_ids_found",
    "suggested_knowledge_types",
    "source_text",
    "statement_fa",
    "formula_latex",
    "variables_json",
    "assumptions_fa",
    "cause_fa",
    "effect_fa",
    "common_error_fa",
    "keywords_fa",
    "evidence_mode",
    "verification_status",
    "reviewer_notes",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: {error}"
                ) from error

    return records


def suggest_knowledge_types(chunk: dict) -> list[str]:
    text = chunk["source_text"].lower()
    equation_ids = chunk.get("equation_ids_found", [])

    suggestions = []

    if equation_ids:
        suggestions.append("formula")

    if any(
        term in text
        for term in [
            "steady-state",
            "steady state",
            "assumption",
            "assume",
        ]
    ):
        suggestions.append("assumption")

    if any(
        term in text
        for term in [
            "initiation",
            "propagation",
            "termination",
            "sequence of three steps",
        ]
    ):
        suggestions.append("mechanism")

    if any(
        term in text
        for term in [
            "rate",
            "kinetic",
            "concentration",
            "rate constant",
        ]
    ):
        suggestions.append("kinetics")

    if any(
        term in text
        for term in [
            "measure",
            "experimental",
            "weighing",
            "spectroscopic",
            "dilatometry",
            "isolation",
        ]
    ):
        suggestions.append("measurement_method")

    if any(
        term in text
        for term in [
            "increase",
            "decrease",
            "dependence",
            "doubling",
            "consequence",
        ]
    ):
        suggestions.append("effect_prediction")

    if not suggestions:
        suggestions.append("concept")

    return list(dict.fromkeys(suggestions))


def build_review_rows(
    chunks: list[dict],
) -> list[dict]:
    rows = []

    for index, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "review_id": f"odian_3_3_review_{index:03d}",
                "chunk_id": chunk["chunk_id"],
                "book": chunk["book"],
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "subsection": chunk["subsection"],
                "printed_page": chunk["printed_page"],
                "pdf_page_1_based": chunk["pdf_page_1_based"],
                "equation_ids_found": chunk.get(
                    "equation_ids_found",
                    [],
                ),
                "suggested_knowledge_types": (
                    suggest_knowledge_types(chunk)
                ),
                "source_text": chunk["source_text"],
                "statement_fa": "",
                "formula_latex": "",
                "variables_json": {},
                "assumptions_fa": "",
                "cause_fa": "",
                "effect_fa": "",
                "common_error_fa": "",
                "keywords_fa": "",
                "evidence_mode": "direct",
                "verification_status": (
                    "pending_scientific_review"
                ),
                "reviewer_notes": "",
            }
        )

    return rows


def validate(
    chunks: list[dict],
    rows: list[dict],
) -> None:
    errors = []

    if len(chunks) != 21:
        errors.append(
            f"تعداد Chunkها باید ۲۱ باشد، اما {len(chunks)} است."
        )

    if len(rows) != len(chunks):
        errors.append(
            "تعداد ردیف‌های Review با Chunkها برابر نیست."
        )

    review_ids = [
        row["review_id"]
        for row in rows
    ]

    chunk_ids = [
        row["chunk_id"]
        for row in rows
    ]

    if len(review_ids) != len(set(review_ids)):
        errors.append("شناسه Review تکراری است.")

    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("یک Chunk بیش از یک‌بار وارد صف شده است.")

    for row in rows:
        if not row["source_text"].strip():
            errors.append(
                f"{row['review_id']}: متن منبع خالی است."
            )

        if row["verification_status"] != (
            "pending_scientific_review"
        ):
            errors.append(
                f"{row['review_id']}: وضعیت بررسی نامعتبر است."
            )

        if row["statement_fa"]:
            errors.append(
                f"{row['review_id']}: statement_fa "
                "باید در این مرحله خالی باشد."
            )

        if not (
            204 <= row["printed_page"] <= 209
        ):
            errors.append(
                f"{row['review_id']}: صفحه چاپی نامعتبر است."
            )

    if errors:
        print("اعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def write_csv(rows: list[dict]) -> None:
    with CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            csv_row = dict(row)

            csv_row["equation_ids_found"] = ";".join(
                row["equation_ids_found"]
            )

            csv_row["suggested_knowledge_types"] = ";".join(
                row["suggested_knowledge_types"]
            )

            csv_row["variables_json"] = json.dumps(
                row["variables_json"],
                ensure_ascii=False,
            )

            writer.writerow(csv_row)


def write_jsonl(rows: list[dict]) -> None:
    with JSONL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_preview(rows: list[dict]) -> None:
    print("\nپیش‌نمایش صف بررسی:")

    for row in rows:
        equations = ",".join(
            row["equation_ids_found"]
        ) or "-"

        types = ",".join(
            row["suggested_knowledge_types"]
        )

        preview = row["source_text"][:110]

        if len(row["source_text"]) > 110:
            preview += "..."

        print(
            f"{row['review_id']} | "
            f"page={row['printed_page']} | "
            f"{row['subsection']} | "
            f"eq={equations} | "
            f"type={types}"
        )

        print(f"  {preview}")


def main() -> None:
    chunks = load_jsonl(INPUT_PATH)

    chunks = sorted(
        chunks,
        key=lambda item: (
            item["printed_page"],
            item["chunk_index_on_page"],
        ),
    )

    rows = build_review_rows(chunks)

    validate(
        chunks,
        rows,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(rows)
    write_jsonl(rows)
    print_preview(rows)

    subsection_counts = Counter(
        row["subsection"]
        for row in rows
    )

    print("\n" + "=" * 72)
    print("صف بررسی علمی ساخته شد.")
    print("CSV:", CSV_PATH)
    print("JSONL:", JSONL_PATH)
    print("تعداد ردیف‌ها:", len(rows))

    print("\nتوزیع زیربخش‌ها:")

    for subsection, count in sorted(
        subsection_counts.items()
    ):
        print(f"- {subsection}: {count}")

    print("\nوضعیت:")
    print("همه ردیف‌ها pending_scientific_review هستند.")
    print("هنوز هیچ رکوردی verified اعلام نشده است.")


if __name__ == "__main__":
    main()
