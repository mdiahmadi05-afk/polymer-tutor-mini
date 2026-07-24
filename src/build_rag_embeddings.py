import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

CHUNK_FILES = [
    Path(
        "data/rag/processed/chunks/"
        "odian_principles_polymerization_4e_chunks.jsonl"
    ),
    Path(
        "data/rag/processed/chunks/"
        "sperling_physical_polymer_science_4e_chunks.jsonl"
    ),
]

INDEX_DIR = Path("data/rag/index")
EMBEDDINGS_PATH = INDEX_DIR / "polymer_chunks_embeddings.npy"
METADATA_PATH = INDEX_DIR / "polymer_chunks_metadata.jsonl"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"


def load_chunks(paths: list[Path]) -> list[dict]:
    chunks: list[dict] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"فایل پیدا نشد: {path}")

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"خطای JSON در {path}، خط {line_number}"
                    ) from error

    return chunks


def validate_chunks(chunks: list[dict]) -> None:
    required_fields = {
        "id",
        "book_id",
        "title",
        "author",
        "pdf_page",
        "printed_page",
        "text",
    }

    ids: set[str] = set()

    for chunk in chunks:
        missing = required_fields - chunk.keys()

        if missing:
            raise ValueError(
                f"فیلدهای {sorted(missing)} در قطعه‌ای وجود ندارند."
            )

        if chunk["id"] in ids:
            raise ValueError(f"شناسه تکراری پیدا شد: {chunk['id']}")

        ids.add(chunk["id"])


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("در حال خواندن قطعه‌های کتاب‌ها...")
    chunks = load_chunks(CHUNK_FILES)
    validate_chunks(chunks)

    print("تعداد کل قطعه‌ها:", len(chunks))

    passages = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    print("در حال بارگذاری مدل Embedding روی GPU...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    print("Device:", model.device)
    print("Embedding dimension:", model.get_embedding_dimension())
    print("Maximum sequence length:", model.max_seq_length)
    print("در حال ساخت Embeddingها...")

    embeddings = model.encode(
        passages,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            "تعداد Embeddingها با تعداد قطعه‌ها برابر نیست."
        )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
        allow_pickle=False,
    )

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    manifest = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_chunks": int(embeddings.shape[0]),
        "normalized_embeddings": True,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "embeddings_file": EMBEDDINGS_PATH.name,
        "metadata_file": METADATA_PATH.name,
        "source_chunk_files": [
            str(path)
            for path in CHUNK_FILES
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 72)
    print("ساخت نمایه برداری با موفقیت تمام شد.")
    print("شکل ماتریس:", embeddings.shape)
    print(f"حجم فایل Embedding: {size_mb:.2f} MB")
    print("Embeddingها:", EMBEDDINGS_PATH)
    print("اطلاعات قطعه‌ها:", METADATA_PATH)
    print("مشخصات نمایه:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
