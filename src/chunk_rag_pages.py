import json
import re
from pathlib import Path


PROCESSED_DIR = Path("data/rag/processed")
OUTPUT_DIR = Path("data/rag/processed/chunks")

CHUNK_SIZE_WORDS = 280
CHUNK_OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 40


BOOKS = [
    {
        "input": PROCESSED_DIR
        / "odian_principles_polymerization_4e_pages.jsonl",
        "output": OUTPUT_DIR
        / "odian_principles_polymerization_4e_chunks.jsonl",
        "pdf_start_page": 28,
        "pdf_end_page": 815,
        "printed_page_offset": 27,
    },
    {
        "input": PROCESSED_DIR
        / "sperling_physical_polymer_science_4e_pages.jsonl",
        "output": OUTPUT_DIR
        / "sperling_physical_polymer_science_4e_chunks.jsonl",
        "pdf_start_page": 34,
        "pdf_end_page": 859,
        "printed_page_offset": 33,
    },
]


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pages(path: Path) -> list[dict]:
    pages = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                pages.append(json.loads(line))

    return pages


def create_page_chunks(
    page_record: dict,
    printed_page_offset: int,
) -> list[dict]:
    text = normalize_text(page_record["text"])
    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return []

    chunks = []
    start = 0
    chunk_number = 1

    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) < MIN_CHUNK_WORDS:
            break

        pdf_page = page_record["pdf_page"]
        printed_page = pdf_page - printed_page_offset

        chunk = {
            "id": (
                f'{page_record["book_id"]}'
                f'_pdf_{pdf_page:04d}'
                f'_chunk_{chunk_number:02d}'
            ),
            "book_id": page_record["book_id"],
            "title": page_record["title"],
            "author": page_record["author"],
            "edition": page_record["edition"],
            "source_file": page_record["source_file"],
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "chunk_number_on_page": chunk_number,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        }

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP_WORDS
        chunk_number += 1

    return chunks


def process_book(book: dict) -> None:
    input_path = book["input"]
    output_path = book["output"]

    if not input_path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {input_path}")

    pages = load_pages(input_path)

    selected_pages = [
        page
        for page in pages
        if book["pdf_start_page"]
        <= page["pdf_page"]
        <= book["pdf_end_page"]
    ]

    all_chunks = []

    for page in selected_pages:
        page_chunks = create_page_chunks(
            page_record=page,
            printed_page_offset=book["printed_page_offset"],
        )

        all_chunks.extend(page_chunks)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 72)
    print("کتاب:", selected_pages[0]["title"])
    print("صفحات پردازش‌شده:", len(selected_pages))
    print("تعداد قطعه‌ها:", len(all_chunks))
    print("فایل خروجی:", output_path)

    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]

        print("\nنمونه قطعه:")
        print("شناسه:", sample["id"])
        print("صفحه PDF:", sample["pdf_page"])
        print("صفحه چاپی کتاب:", sample["printed_page"])
        print("تعداد کلمات:", sample["word_count"])
        print("متن:")
        print(sample["text"][:700])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for book in BOOKS:
        process_book(book)

    print("\nتقسیم‌بندی متن کتاب‌ها با موفقیت تمام شد.")


if __name__ == "__main__":
    main()
