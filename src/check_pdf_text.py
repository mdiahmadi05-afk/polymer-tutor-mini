from pathlib import Path
import pymupdf


PDF_DIR = Path("data/rag/source_pdfs")

pdf_files = [
    PDF_DIR / "Principles of Polymerization.pdf",
    PDF_DIR / "Introduction to Physical Polymer Science.pdf",
    PDF_DIR / "Polymers in Nature.pdf",
]


for pdf_path in pdf_files:
    if not pdf_path.exists():
        print(f"\nفایل پیدا نشد: {pdf_path}")
        continue

    document = pymupdf.open(pdf_path)

    print("\n" + "=" * 70)
    print("نام فایل:", pdf_path.name)
    print("تعداد صفحات:", len(document))

    total_characters = 0
    sample_page = None
    sample_text = ""

    for page_number in range(min(40, len(document))):
        text = document[page_number].get_text("text").strip()
        total_characters += len(text)

        if sample_page is None and len(text) > 300:
            sample_page = page_number + 1
            sample_text = " ".join(text.split())[:500]

    print("تعداد کاراکتر استخراج‌شده از ۴۰ صفحه اول:", total_characters)

    if sample_page is not None:
        print("اولین صفحه دارای متن مناسب:", sample_page)
        print("نمونه متن:")
        print(sample_text)
    else:
        print("متن قابل‌استفاده پیدا نشد؛ احتمالاً PDF اسکن تصویری است.")

    document.close()
