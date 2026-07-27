import hashlib
import json
import re
from pathlib import Path

import fitz


OUTPUT_DIR = Path(
    "data/knowledge/odian_ch3/source"
)

TEXT_OUTPUT = OUTPUT_DIR / "section_3_3_source.txt"
METADATA_OUTPUT = OUTPUT_DIR / "section_3_3_source_metadata.json"

START_MARKER = "3-3 RATE OF RADICAL CHAIN POLYMERIZATION"
END_MARKER = "3-4 INITIATION"

REQUIRED_MARKERS = [
    "3-3a",
    "Sequence of Events",
    "3-3b",
    "Rate Expression",
    "3-3c",
    "Experimental Determination of Rp",
    "Physical Separation and Isolation",
    "Chemical and Spectroscopic Analysis",
    "Other Techniques",
    "Dilatometry",
]

REQUIRED_EQUATION_IDS = [
    "3-13",
    "3-19",
    "3-20",
    "3-21",
    "3-22",
    "3-23",
    "3-24",
    "3-25",
]


def find_pdf() -> Path:
    preferred_paths = [
        Path("books/Principles of Polymerization.pdf"),
        Path("data/books/Principles of Polymerization.pdf"),
        Path("data/rag/books/Principles of Polymerization.pdf"),
        Path("rag/books/Principles of Polymerization.pdf"),
        Path("Principles of Polymerization.pdf"),
    ]

    for path in preferred_paths:
        if path.exists():
            return path.resolve()

    matches = sorted(
        path
        for path in Path(".").rglob("*.pdf")
        if (
            "principles" in path.name.lower()
            and "polymerization" in path.name.lower()
        )
    )

    if not matches:
        raise FileNotFoundError(
            "فایل Principles of Polymerization.pdf "
            "در پروژه پیدا نشد."
        )

    return matches[0].resolve()


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def extract_section(
    pdf_path: Path,
) -> tuple[str, int, int]:
    document = fitz.open(pdf_path)

    start_page = None
    end_page = None
    page_texts = []

    for page_index in range(document.page_count):
        text = document[page_index].get_text(
            "text",
            sort=True,
        )

        page_texts.append(text)

        normalized = " ".join(text.split())

        if (
            start_page is None
            and START_MARKER in normalized
        ):
            start_page = page_index

        if (
            start_page is not None
            and page_index >= start_page
            and END_MARKER in normalized
        ):
            end_page = page_index
            break

    if start_page is None:
        raise RuntimeError(
            "ابتدای بخش 3-3 در PDF پیدا نشد."
        )

    if end_page is None:
        raise RuntimeError(
            "انتهای بخش 3-3 یعنی عنوان 3-4 پیدا نشد."
        )

    extracted_pages = []

    for page_index in range(
        start_page,
        end_page + 1,
    ):
        extracted_pages.append(
            page_texts[page_index]
        )

    section_text = "\n".join(extracted_pages)

    start_position = section_text.find(
        START_MARKER
    )

    if start_position == -1:
        raise RuntimeError(
            "نشانگر شروع در متن استخراج‌شده پیدا نشد."
        )

    end_match = re.search(
        r"3\s*[-–—]\s*4\s+INITIATION",
        section_text[start_position:],
        flags=re.IGNORECASE,
    )

    if end_match is None:
        raise RuntimeError(
            "عنوان بخش 3-4 حتی با جست‌وجوی منعطف پیدا نشد."
        )

    end_position = (
        start_position
        + end_match.start()
    )

    section_text = section_text[
        start_position:end_position
    ].strip()

    document.close()

    return (
        section_text,
        start_page,
        end_page,
    )


def validate_section(text: str) -> None:
    errors = []

    normalized = " ".join(text.split())

    for marker in REQUIRED_MARKERS:
        if marker not in normalized:
            errors.append(
                f"عنوان یا عبارت ضروری پیدا نشد: {marker}"
            )

    for equation_id in REQUIRED_EQUATION_IDS:
        if equation_id not in normalized:
            errors.append(
                f"شماره رابطه پیدا نشد: {equation_id}"
            )

    if len(text) < 10000:
        errors.append(
            "متن استخراج‌شده بیش از حد کوتاه است."
        )

    if errors:
        print("اعتبارسنجی متن ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def main() -> None:
    pdf_path = find_pdf()

    print("فایل PDF:")
    print(pdf_path)

    section_text, start_page, end_page = (
        extract_section(pdf_path)
    )

    validate_section(section_text)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEXT_OUTPUT.write_text(
        section_text + "\n",
        encoding="utf-8",
    )

    metadata = {
        "book": "Principles of Polymerization",
        "author": "George Odian",
        "edition": "Fourth Edition",
        "chapter": 3,
        "section": "3-3",
        "section_title": (
            "Rate of Radical Chain Polymerization"
        ),
        "printed_page_start": 204,
        "printed_page_end": 209,
        "pdf_page_start_1_based": start_page + 1,
        "pdf_page_end_1_based": end_page + 1,
        "character_count": len(section_text),
        "line_count": len(section_text.splitlines()),
        "pdf_sha256": calculate_sha256(pdf_path),
        "source_pdf": str(pdf_path),
        "subsections": [
            "3-3a Sequence of Events",
            "3-3b Rate Expression",
            "3-3c Experimental Determination of Rp",
            (
                "3-3c-1 Physical Separation and "
                "Isolation of Reaction Product"
            ),
            (
                "3-3c-2 Chemical and "
                "Spectroscopic Analysis"
            ),
            "3-3c-3 Other Techniques",
        ],
        "validation": {
            "required_markers_found": True,
            "required_equations_found": True,
        },
    }

    METADATA_OUTPUT.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("استخراج بخش 3-3 موفق بود.")
    print("صفحه PDF شروع:", start_page + 1)
    print("صفحه PDF پایان:", end_page + 1)
    print("صفحات چاپی: 204 تا 209")
    print("تعداد کاراکتر:", len(section_text))
    print("تعداد خطوط:", len(section_text.splitlines()))
    print("\nمتن منبع:")
    print(TEXT_OUTPUT)
    print("\nمتادیتا:")
    print(METADATA_OUTPUT)


if __name__ == "__main__":
    main()
