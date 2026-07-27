import csv
import hashlib
import json
import re
from pathlib import Path

import fitz


METADATA_PATH = Path(
    "data/knowledge/odian_ch3/source/"
    "section_3_3_source_metadata.json"
)

OUTPUT_DIR = Path(
    "data/knowledge/odian_ch3/source"
)

CSV_PATH = OUTPUT_DIR / "section_3_3_pages.csv"
JSONL_PATH = OUTPUT_DIR / "section_3_3_pages.jsonl"

START_PRINTED_PAGE = 204
END_PRINTED_PAGE = 209

START_PATTERN = re.compile(
    r"3\s*[-–—]\s*3\s+RATE\s+OF\s+RADICAL\s+"
    r"CHAIN\s+POLYMERIZATION",
    flags=re.IGNORECASE,
)

END_PATTERN = re.compile(
    r"3\s*[-–—]\s*4\s+INITIATION",
    flags=re.IGNORECASE,
)

SUBSECTION_PATTERNS = {
    "3-3a": re.compile(
        r"3\s*[-–—]\s*3a",
        flags=re.IGNORECASE,
    ),
    "3-3b": re.compile(
        r"3\s*[-–—]\s*3b",
        flags=re.IGNORECASE,
    ),
    "3-3c": re.compile(
        r"3\s*[-–—]\s*3c(?:\s*[-–—]\s*\d+)?",
        flags=re.IGNORECASE,
    ),
}

REQUIRED_MARKERS = [
    "3-13",
    "3-19",
    "3-20",
    "3-21",
    "3-22",
    "3-23",
    "3-24",
    "3-25",
]

CSV_FIELDS = [
    "page_id",
    "book_id",
    "book",
    "author",
    "edition",
    "chapter",
    "section",
    "page_sequence",
    "printed_page",
    "pdf_page_1_based",
    "subsections_found",
    "equation_ids_found",
    "character_count",
    "word_count",
    "text_sha256",
    "source_pdf_sha256",
    "verification_status",
    "clean_text",
]


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"فایل متادیتا پیدا نشد: {METADATA_PATH}"
        )

    return json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )


def normalize_characters(text: str) -> str:
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "\u00ad": "",
        "\u00a0": " ",
        "−": "-",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_known_headers(text: str) -> str:
    cleaned_lines = []

    header_patterns = [
        re.compile(
            r"^\s*\d+\s+RADICAL\s+CHAIN\s+"
            r"POLYMERIZATION\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*RATE\s+OF\s+RADICAL\s+CHAIN\s+"
            r"POLYMERIZATION\s+\d+\s*$",
            flags=re.IGNORECASE,
        ),
    ]

    for line in text.splitlines():
        normalized_line = normalize_characters(line).strip()

        if any(
            pattern.match(normalized_line)
            for pattern in header_patterns
        ):
            continue

        cleaned_lines.append(normalized_line)

    return "\n".join(cleaned_lines)


def make_single_line(text: str) -> str:
    return " ".join(text.split())


def find_subsections(text: str) -> list[str]:
    found = []

    for subsection, pattern in SUBSECTION_PATTERNS.items():
        if pattern.search(text):
            found.append(subsection)

    return found


def find_equation_ids(text: str) -> list[str]:
    found = []

    for equation_number in range(13, 26):
        equation_id = f"3-{equation_number}"

        if equation_id in text:
            found.append(equation_id)

    return found


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def extract_pages(
    metadata: dict,
) -> list[dict]:
    pdf_path = Path(metadata["source_pdf"])

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"فایل PDF پیدا نشد: {pdf_path}"
        )

    start_pdf_page = metadata["pdf_page_start_1_based"]
    end_pdf_page = metadata["pdf_page_end_1_based"]

    expected_page_count = (
        END_PRINTED_PAGE
        - START_PRINTED_PAGE
        + 1
    )

    actual_page_count = (
        end_pdf_page
        - start_pdf_page
        + 1
    )

    if actual_page_count != expected_page_count:
        raise ValueError(
            "تعداد صفحات PDF و صفحات چاپی برابر نیست: "
            f"{actual_page_count} در برابر "
            f"{expected_page_count}"
        )

    document = fitz.open(pdf_path)
    records = []

    try:
        for offset, pdf_page_1_based in enumerate(
            range(
                start_pdf_page,
                end_pdf_page + 1,
            )
        ):
            printed_page = START_PRINTED_PAGE + offset

            raw_text = document[
                pdf_page_1_based - 1
            ].get_text(
                "text",
                sort=True,
            )

            if offset == 0:
                start_match = START_PATTERN.search(raw_text)

                if start_match is None:
                    raise RuntimeError(
                        "عنوان شروع بخش 3-3 در صفحه اول "
                        "پیدا نشد."
                    )

                raw_text = raw_text[start_match.start():]

            if offset == actual_page_count - 1:
                end_match = END_PATTERN.search(raw_text)

                if end_match is None:
                    raise RuntimeError(
                        "عنوان بخش 3-4 در صفحه آخر "
                        "پیدا نشد."
                    )

                raw_text = raw_text[:end_match.start()]

            normalized_text = normalize_characters(
                raw_text
            )

            clean_multiline_text = remove_known_headers(
                normalized_text
            ).strip()

            clean_text = make_single_line(
                clean_multiline_text
            )

            subsections = find_subsections(
                clean_multiline_text
            )

            equation_ids = find_equation_ids(
                clean_multiline_text
            )

            record = {
                "page_id": (
                    f"odian_3_3_page_{printed_page}"
                ),
                "book_id": "odian_4e",
                "book": metadata["book"],
                "author": metadata["author"],
                "edition": metadata["edition"],
                "chapter": str(metadata["chapter"]),
                "section": metadata["section"],
                "page_sequence": offset + 1,
                "printed_page": printed_page,
                "pdf_page_1_based": pdf_page_1_based,
                "subsections_found": subsections,
                "equation_ids_found": equation_ids,
                "character_count": len(
                    clean_multiline_text
                ),
                "word_count": len(
                    clean_multiline_text.split()
                ),
                "text_sha256": sha256_text(
                    clean_multiline_text
                ),
                "source_pdf_sha256": metadata[
                    "pdf_sha256"
                ],
                "verification_status": (
                    "source_extracted_not_scientifically_reviewed"
                ),
                "clean_text": clean_text,
                "raw_text": raw_text,
                "clean_multiline_text": (
                    clean_multiline_text
                ),
            }

            records.append(record)

    finally:
        document.close()

    return records


def validate(records: list[dict]) -> None:
    errors = []

    if len(records) != 6:
        errors.append(
            f"تعداد صفحات باید ۶ باشد، "
            f"اما {len(records)} است."
        )

    expected_printed_pages = list(
        range(
            START_PRINTED_PAGE,
            END_PRINTED_PAGE + 1,
        )
    )

    actual_printed_pages = [
        record["printed_page"]
        for record in records
    ]

    if actual_printed_pages != expected_printed_pages:
        errors.append(
            "ترتیب صفحات چاپی نادرست است: "
            f"{actual_printed_pages}"
        )

    page_ids = [
        record["page_id"]
        for record in records
    ]

    if len(page_ids) != len(set(page_ids)):
        errors.append(
            "شناسه صفحه تکراری وجود دارد."
        )

    for record in records:
        if not record["clean_text"]:
            errors.append(
                f"{record['page_id']}: متن صفحه خالی است."
            )

        if record["character_count"] < 200:
            errors.append(
                f"{record['page_id']}: متن صفحه "
                "بیش از حد کوتاه است."
            )

        if "\n" in record["clean_text"]:
            errors.append(
                f"{record['page_id']}: clean_text "
                "باید تک‌خطی باشد."
            )

    joined_text = " ".join(
        record["clean_text"]
        for record in records
    )

    for subsection in [
        "3-3a",
        "3-3b",
        "3-3c",
    ]:
        pattern = SUBSECTION_PATTERNS[subsection]

        if pattern.search(joined_text) is None:
            errors.append(
                f"زیربخش پیدا نشد: {subsection}"
            )

    for marker in REQUIRED_MARKERS:
        if marker not in joined_text:
            errors.append(
                f"شماره رابطه پیدا نشد: {marker}"
            )

    if errors:
        print("اعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def write_csv(records: list[dict]) -> None:
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

        for record in records:
            csv_record = {
                field: record[field]
                for field in CSV_FIELDS
            }

            csv_record["subsections_found"] = ";".join(
                record["subsections_found"]
            )

            csv_record["equation_ids_found"] = ";".join(
                record["equation_ids_found"]
            )

            writer.writerow(csv_record)


def write_jsonl(records: list[dict]) -> None:
    with JSONL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_preview(records: list[dict]) -> None:
    print("\nجدول صفحات:")

    print(
        f"{'page_id':<24}"
        f"{'printed':<10}"
        f"{'pdf':<8}"
        f"{'words':<8}"
        f"{'subsections':<20}"
        f"equations"
    )

    print("-" * 100)

    for record in records:
        subsections = ",".join(
            record["subsections_found"]
        )

        equations = ",".join(
            record["equation_ids_found"]
        )

        print(
            f"{record['page_id']:<24}"
            f"{record['printed_page']:<10}"
            f"{record['pdf_page_1_based']:<8}"
            f"{record['word_count']:<8}"
            f"{subsections:<20}"
            f"{equations}"
        )


def main() -> None:
    metadata = load_metadata()
    records = extract_pages(metadata)

    validate(records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(records)
    write_jsonl(records)
    print_preview(records)

    print("\n" + "=" * 72)
    print("Source Table صفحه‌به‌صفحه ساخته شد.")
    print("CSV:", CSV_PATH)
    print("JSONL:", JSONL_PATH)
    print("تعداد صفحات:", len(records))
    print(
        "وضعیت علمی:",
        "هنوز نیازمند تبدیل به Knowledge Table",
    )


if __name__ == "__main__":
    main()
