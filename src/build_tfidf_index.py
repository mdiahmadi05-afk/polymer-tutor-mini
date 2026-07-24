import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


INDEX_DIR = Path("data/rag/index")

METADATA_PATH = INDEX_DIR / "polymer_chunks_metadata.jsonl"
TFIDF_MATRIX_PATH = INDEX_DIR / "polymer_chunks_tfidf.npz"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.joblib"
MANIFEST_PATH = INDEX_DIR / "tfidf_manifest.json"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")

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
                    f"خطای JSON در خط {line_number}"
                ) from error

    return records


def main() -> None:
    print("در حال خواندن قطعه‌های کتاب‌ها...")

    records = load_chunks(METADATA_PATH)
    documents = [record["text"] for record in records]

    print("تعداد قطعه‌ها:", len(documents))
    print("در حال ساخت نمایه TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    if tfidf_matrix.shape[0] != len(records):
        raise RuntimeError(
            "تعداد سطرهای ماتریس با تعداد قطعه‌ها برابر نیست."
        )

    sparse.save_npz(
        TFIDF_MATRIX_PATH,
        tfidf_matrix,
        compressed=True,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
        compress=3,
    )

    manifest = {
        "number_of_chunks": int(tfidf_matrix.shape[0]),
        "vocabulary_size": int(tfidf_matrix.shape[1]),
        "ngram_range": [1, 2],
        "normalization": "l2",
        "matrix_file": TFIDF_MATRIX_PATH.name,
        "vectorizer_file": VECTORIZER_PATH.name,
        "metadata_file": METADATA_PATH.name,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    matrix_size_mb = (
        TFIDF_MATRIX_PATH.stat().st_size / (1024 * 1024)
    )

    print("\n" + "=" * 72)
    print("نمایه TF-IDF با موفقیت ساخته شد.")
    print("شکل ماتریس:", tfidf_matrix.shape)
    print("تعداد واژه‌ها و عبارت‌ها:", len(vectorizer.vocabulary_))
    print(f"حجم فایل ماتریس: {matrix_size_mb:.2f} MB")
    print("ماتریس:", TFIDF_MATRIX_PATH)
    print("Vectorizer:", VECTORIZER_PATH)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
