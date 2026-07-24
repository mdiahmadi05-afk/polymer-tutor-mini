import json
import re
from pathlib import Path

import pymupdf


SOURCE_DIR = Path("data/rag/source_pdfs")
OUTPUT_DIR = Path("data/rag/processed")

BOOKS = [
    {
        "path": SOURCE_DIR / "Principles of Polymerization.pdf",
        "book_id": "odian_principles_polymerization_4e",
        "title": "Principles of Polymerization",
        "author": "George Odian",
        "edition": "Fourth Edition",
    },
    {
        "path": SOURCE_DIR / "Introduction to Physical Polymer Science.pdf",
        "book_id": "sperling_physical_polymer_science_4e",
        "title": "Introduction to Physical Polymer Science",
        "author": "L. H. Sperling",
        "edition": "Fourth Edition",
    },
]


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\x00", " ")

    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_book(book: dict) -> None:
    pdf_path = book["path"]

    if not pdf_path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {pdf_path}")

    output_path = OUTPUT_DIR / f'{book["book_id"]}_pages.jsonl'

    document = pymupdf.open(pdf_path)

    extracted_pages = 0
    empty_pages = 0
    total_characters = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for pdf_page_index, page in enumerate(document):
            text = clean_text(page.get_text("text"))

            if len(text) < 30:
                empty_pages += 1
                continue

            record = {
                "id": f'{book["book_id"]}_pdf_{pdf_page_index + 1:04d}',
                "book_id": book["book_id"],
                "title": book["title"],
                "author": book["author"],
                "edition": book["edition"],
                "source_file": pdf_path.name,
                "pdf_page": pdf_page_index + 1,
                "text": text,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            extracted_pages += 1
            total_characters += len(text)

    document.close()

    print("\n" + "=" * 70)
    print("کتاب:", book["title"])
    print("فایل خروجی:", output_path)
    print("صفحات دارای متن:", extracted_pages)
    print("صفحات خالی یا تصویری:", empty_pages)
    print("مجموع کاراکترها:", total_characters)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for book in BOOKS:
        extract_book(book)

    print("\nاستخراج صفحه‌به‌صفحه دو کتاب با موفقیت تمام شد.")


if __name__ == "__main__":
    main()
