import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "data/knowledge/odian_ch3/source/"
    "section_3_3_pages.jsonl"
)

OUTPUT_DIR = Path(
    "data/knowledge/odian_ch3/source"
)

CSV_PATH = OUTPUT_DIR / "section_3_3_source_chunks.csv"
JSONL_PATH = OUTPUT_DIR / "section_3_3_source_chunks.jsonl"

MIN_TARGET_CHARS = 280
MAX_CHARS = 950


FIELDNAMES = [
    "chunk_id",
    "book_id",
    "book",
    "author",
    "edition",
    "chapter",
    "section",
    "subsection",
    "printed_page",
    "pdf_page_1_based",
    "page_id",
    "chunk_index_on_page",
    "paragraph_indices",
    "equation_ids_found",
    "character_count",
    "word_count",
    "text_sha256",
    "source_page_sha256",
    "source_pdf_sha256",
    "review_status",
    "source_text",
]


SUBSECTION_RULES = [
    (
        "3-3c-1",
        re.compile(
            r"\b3\s*-\s*3c\s*-\s*1\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "3-3c-2",
        re.compile(
            r"\b3\s*-\s*3c\s*-\s*2\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "3-3c-3",
        re.compile(
            r"\b3\s*-\s*3c\s*-\s*3\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "3-3a",
        re.compile(
            r"\b3\s*-\s*3a\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "3-3b",
        re.compile(
            r"\b3\s*-\s*3b\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "3-3c",
        re.compile(
            r"\b3\s*-\s*3c\b",
            flags=re.IGNORECASE,
        ),
    ),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: "
                    f"{error}"
                ) from error

    return records


def normalize_text(text: str) -> str:
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

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def detect_subsection(
    line: str,
) -> str | None:
    normalized = normalize_text(line)

    for subsection, pattern in SUBSECTION_RULES:
        if pattern.search(normalized):
            return subsection

    return None


def is_subsection_heading(line: str) -> bool:
    subsection = detect_subsection(line)

    if subsection is None:
        return False

    normalized = normalize_text(line)

    return len(normalized) <= 160


def split_page_into_paragraphs(
    page: dict,
    starting_subsection: str,
) -> tuple[list[dict], str]:
    text = page.get("clean_multiline_text", "")

    if not text.strip():
        raise ValueError(
            f"متن چندخطی صفحه وجود ندارد: "
            f"{page.get('page_id')}"
        )

    paragraphs = []
    buffer = []
    current_subsection = starting_subsection
    paragraph_index = 0

    def flush_buffer() -> None:
        nonlocal buffer
        nonlocal paragraph_index

        if not buffer:
            return

        paragraph_text = normalize_text(
            " ".join(buffer)
        )

        buffer = []

        if not paragraph_text:
            return

        paragraph_index += 1

        paragraphs.append(
            {
                "subsection": current_subsection,
                "paragraph_index": paragraph_index,
                "text": paragraph_text,
            }
        )

    for original_line in text.splitlines():
        line = normalize_text(original_line)

        if not line:
            flush_buffer()
            continue

        detected = detect_subsection(line)

        if (
            detected is not None
            and is_subsection_heading(line)
        ):
            flush_buffer()
            current_subsection = detected
            continue

        buffer.append(line)

    flush_buffer()

    return paragraphs, current_subsection


def split_oversized_text(
    text: str,
) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        proposed = (
            sentence
            if not current
            else f"{current} {sentence}"
        )

        if len(proposed) <= MAX_CHARS:
            current = proposed
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(sentence) <= MAX_CHARS:
            current = sentence
            continue

        words = sentence.split()
        word_buffer = ""

        for word in words:
            proposed_word_buffer = (
                word
                if not word_buffer
                else f"{word_buffer} {word}"
            )

            if (
                len(proposed_word_buffer)
                <= MAX_CHARS
            ):
                word_buffer = proposed_word_buffer
            else:
                if word_buffer:
                    pieces.append(word_buffer)

                word_buffer = word

        current = word_buffer

    if current:
        pieces.append(current)

    return pieces


def find_equation_ids(text: str) -> list[str]:
    matches = re.findall(
        r"(?<!\d)3\s*[-–—]\s*(1[3-9]|2[0-5])(?!\d)",
        text,
        flags=re.IGNORECASE,
    )

    return sorted(
        {
            f"3-{number}"
            for number in matches
        },
        key=lambda value: int(
            value.split("-")[1]
        ),
    )

def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def group_paragraphs(
    paragraphs: list[dict],
) -> list[dict]:
    expanded = []

    for paragraph in paragraphs:
        pieces = split_oversized_text(
            paragraph["text"]
        )

        for piece in pieces:
            expanded.append(
                {
                    "subsection": (
                        paragraph["subsection"]
                    ),
                    "paragraph_indices": [
                        paragraph["paragraph_index"]
                    ],
                    "text": piece,
                }
            )

    chunks = []
    current = None

    for item in expanded:
        if current is None:
            current = dict(item)
            continue

        same_subsection = (
            current["subsection"]
            == item["subsection"]
        )

        proposed_text = (
            f"{current['text']} {item['text']}"
        )

        should_merge = (
            same_subsection
            and (
                len(current["text"])
                < MIN_TARGET_CHARS
                or len(proposed_text) <= MAX_CHARS
            )
            and len(proposed_text) <= MAX_CHARS
        )

        if should_merge:
            current["text"] = proposed_text
            current["paragraph_indices"].extend(
                item["paragraph_indices"]
            )
        else:
            chunks.append(current)
            current = dict(item)

    if current is not None:
        chunks.append(current)

    if (
        len(chunks) >= 2
        and len(chunks[-1]["text"]) < 120
        and chunks[-1]["subsection"]
        == chunks[-2]["subsection"]
    ):
        proposed = (
            f"{chunks[-2]['text']} "
            f"{chunks[-1]['text']}"
        )

        if len(proposed) <= MAX_CHARS:
            chunks[-2]["text"] = proposed
            chunks[-2][
                "paragraph_indices"
            ].extend(
                chunks[-1][
                    "paragraph_indices"
                ]
            )

            chunks.pop()

    return chunks


def build_chunks(
    pages: list[dict],
) -> list[dict]:
    pages = sorted(
        pages,
        key=lambda item: item["printed_page"],
    )

    output = []
    current_subsection = "3-3a"

    for page in pages:
        paragraphs, current_subsection = (
            split_page_into_paragraphs(
                page,
                current_subsection,
            )
        )

        page_chunks = group_paragraphs(
            paragraphs
        )

        for chunk_index, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            source_text = normalize_text(
                chunk["text"]
            )

            equation_ids = find_equation_ids(
                source_text
            )

            chunk_id = (
                f"odian_3_3_p"
                f"{page['printed_page']}_"
                f"c{chunk_index:02d}"
            )

            output.append(
                {
                    "chunk_id": chunk_id,
                    "book_id": page["book_id"],
                    "book": page["book"],
                    "author": page["author"],
                    "edition": page["edition"],
                    "chapter": page["chapter"],
                    "section": page["section"],
                    "subsection": (
                        chunk["subsection"]
                    ),
                    "printed_page": (
                        page["printed_page"]
                    ),
                    "pdf_page_1_based": (
                        page["pdf_page_1_based"]
                    ),
                    "page_id": page["page_id"],
                    "chunk_index_on_page": (
                        chunk_index
                    ),
                    "paragraph_indices": (
                        chunk[
                            "paragraph_indices"
                        ]
                    ),
                    "equation_ids_found": (
                        equation_ids
                    ),
                    "character_count": len(
                        source_text
                    ),
                    "word_count": len(
                        source_text.split()
                    ),
                    "text_sha256": sha256_text(
                        source_text
                    ),
                    "source_page_sha256": (
                        page["text_sha256"]
                    ),
                    "source_pdf_sha256": (
                        page["source_pdf_sha256"]
                    ),
                    "review_status": (
                        "source_extracted_"
                        "needs_scientific_review"
                    ),
                    "source_text": source_text,
                }
            )

    return output


def validate(
    pages: list[dict],
    chunks: list[dict],
) -> None:
    errors = []

    if len(pages) != 6:
        errors.append(
            f"تعداد صفحات ورودی باید ۶ باشد، "
            f"اما {len(pages)} است."
        )

    if not 15 <= len(chunks) <= 45:
        errors.append(
            f"تعداد Chunkها غیرعادی است: "
            f"{len(chunks)}"
        )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "شناسه Chunk تکراری وجود دارد."
        )

    for chunk in chunks:
        if not chunk["source_text"].strip():
            errors.append(
                f"{chunk['chunk_id']}: متن خالی است."
            )

        if chunk["character_count"] > MAX_CHARS:
            errors.append(
                f"{chunk['chunk_id']}: طول بیش از "
                f"{MAX_CHARS} کاراکتر است."
            )

        if not (
            204
            <= chunk["printed_page"]
            <= 209
        ):
            errors.append(
                f"{chunk['chunk_id']}: "
                "صفحه چاپی نامعتبر است."
            )

    found_subsections = {
        chunk["subsection"]
        for chunk in chunks
    }

    required_subsections = {
        "3-3a",
        "3-3b",
        "3-3c",
        "3-3c-1",
        "3-3c-2",
        "3-3c-3",
    }

    missing_subsections = (
        required_subsections
        - found_subsections
    )

    if missing_subsections:
        errors.append(
            "زیربخش‌های پیدا‌نشده: "
            f"{sorted(missing_subsections)}"
        )

    found_equations = {
        equation_id
        for chunk in chunks
        for equation_id in (
            chunk["equation_ids_found"]
        )
    }

    required_equations = {
        f"3-{number}"
        for number in range(13, 26)
    }

    missing_equations = (
        required_equations
        - found_equations
    )

    if missing_equations:
        errors.append(
            "روابط پیدا‌نشده در Chunkها: "
            f"{sorted(missing_equations)}"
        )

    if errors:
        print("اعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def write_csv(
    chunks: list[dict],
) -> None:
    with CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for chunk in chunks:
            row = dict(chunk)

            row["paragraph_indices"] = ";".join(
                str(value)
                for value in chunk[
                    "paragraph_indices"
                ]
            )

            row["equation_ids_found"] = ";".join(
                chunk["equation_ids_found"]
            )

            writer.writerow(row)


def write_jsonl(
    chunks: list[dict],
) -> None:
    with JSONL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    pages = load_jsonl(INPUT_PATH)
    chunks = build_chunks(pages)

    validate(
        pages,
        chunks,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(chunks)
    write_jsonl(chunks)

    subsection_counts = Counter(
        chunk["subsection"]
        for chunk in chunks
    )

    page_counts = Counter(
        chunk["printed_page"]
        for chunk in chunks
    )

    print("=" * 72)
    print("Source Chunk Table ساخته شد.")
    print("CSV:", CSV_PATH)
    print("JSONL:", JSONL_PATH)
    print("تعداد Chunkها:", len(chunks))

    print("\nتوزیع بر اساس صفحه:")

    for page, count in sorted(
        page_counts.items()
    ):
        print(f"- صفحه {page}: {count}")

    print("\nتوزیع بر اساس زیربخش:")

    for subsection, count in sorted(
        subsection_counts.items()
    ):
        print(f"- {subsection}: {count}")

    print("\nوضعیت:")
    print(
        "Chunkها به متن و صفحه واقعی متصل‌اند، "
        "اما هنوز نیازمند بررسی علمی هستند."
    )


if __name__ == "__main__":
    main()
