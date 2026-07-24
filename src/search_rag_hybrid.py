import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("data/rag/index")

EMBEDDINGS_PATH = INDEX_DIR / "polymer_chunks_embeddings.npy"
METADATA_PATH = INDEX_DIR / "polymer_chunks_metadata.jsonl"
DENSE_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

TFIDF_MATRIX_PATH = INDEX_DIR / "polymer_chunks_tfidf.npz"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.joblib"


def load_metadata(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def looks_like_reference_chunk(text: str) -> bool:
    upper_text = text.upper()

    if "REFERENCES" in upper_text:
        return True

    years = re.findall(r"\b(?:18|19|20)\d{2}\b", text)

    reference_terms = [
        "J. Polym. Sci.",
        "Macromolecules",
        "Makromol. Chem.",
        "Academic Press",
        "Springer",
        "Wiley-Interscience",
        "Vol.",
        "pp.",
        "Ed.,",
    ]

    hits = sum(
        term.lower() in text.lower()
        for term in reference_terms
    )

    if len(years) >= 4 and hits >= 2:
        return True

    if text.count(";") >= 6 and len(years) >= 3:
        return True

    return False


def reciprocal_rank_fusion(
    dense_ranking: np.ndarray,
    lexical_ranking: np.ndarray,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.2,
    rrf_k: int = 60,
) -> dict[int, float]:
    scores: dict[int, float] = {}

    for rank, index in enumerate(dense_ranking, start=1):
        index = int(index)
        scores[index] = scores.get(index, 0.0) + (
            dense_weight / (rrf_k + rank)
        )

    for rank, index in enumerate(lexical_ranking, start=1):
        index = int(index)
        scores[index] = scores.get(index, 0.0) + (
            lexical_weight / (rrf_k + rank)
        )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="جست‌وجوی ترکیبی در کتاب‌های پلیمر"
    )

    parser.add_argument(
        "question",
        type=str,
        help="سؤال اصلی، فارسی یا انگلیسی",
    )

    parser.add_argument(
        "--english-query",
        type=str,
        default="",
        help="نسخه فنی انگلیسی سؤال برای جست‌وجوی واژه‌ای",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    metadata = load_metadata(METADATA_PATH)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    with DENSE_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if embeddings.shape[0] != len(metadata):
        raise RuntimeError(
            "تعداد Embeddingها با متادیتا برابر نیست."
        )

    if tfidf_matrix.shape[0] != len(metadata):
        raise RuntimeError(
            "تعداد سطرهای TF-IDF با متادیتا برابر نیست."
        )

    technical_query = args.english_query.strip()

    if technical_query:
        dense_input = (
            args.question.strip()
            + " "
            + technical_query
        )
        lexical_input = technical_query
    else:
        dense_input = args.question.strip()
        lexical_input = args.question.strip()

    print("در حال بارگذاری مدل Embedding...")

    model = SentenceTransformer(
        manifest["embedding_model"],
        device="cuda",
    )

    dense_query = (
        manifest.get("query_prefix", "query: ")
        + dense_input
    )

    query_embedding = model.encode(
        [dense_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    dense_scores = embeddings @ query_embedding

    lexical_vector = vectorizer.transform(
        [lexical_input]
    )

    lexical_scores = (
        tfidf_matrix @ lexical_vector.T
    ).toarray().ravel()

    candidate_k = min(
        args.candidate_k,
        len(metadata),
    )

    dense_ranking = np.argsort(
        dense_scores
    )[::-1][:candidate_k]

    lexical_ranking = np.argsort(
        lexical_scores
    )[::-1][:candidate_k]

    fused_scores = reciprocal_rank_fusion(
        dense_ranking=dense_ranking,
        lexical_ranking=lexical_ranking,
    )

    ranked_fused = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected = []
    skipped_references = 0

    for index, hybrid_score in ranked_fused:
        record = metadata[index]

        if looks_like_reference_chunk(record["text"]):
            skipped_references += 1
            continue

        selected.append(
            (
                index,
                hybrid_score,
                float(dense_scores[index]),
                float(lexical_scores[index]),
                record,
            )
        )

        if len(selected) >= args.top_k:
            break

    print("\n" + "=" * 80)
    print("سؤال فارسی:", args.question)

    if technical_query:
        print("عبارت فنی انگلیسی:", technical_query)

    print("قطعه‌های منابع حذف‌شده:", skipped_references)
    print("=" * 80)

    for rank, item in enumerate(selected, start=1):
        (
            index,
            hybrid_score,
            dense_score,
            lexical_score,
            record,
        ) = item

        text = " ".join(record["text"].split())

        print(f"\nنتیجه {rank}")
        print(f"امتیاز Hybrid: {hybrid_score:.6f}")
        print(f"امتیاز Dense: {dense_score:.4f}")
        print(f"امتیاز TF-IDF: {lexical_score:.4f}")
        print(f"کتاب: {record['title']}")
        print(f"صفحه PDF: {record['pdf_page']}")
        print(f"صفحه چاپی: {record['printed_page']}")
        print(f"شناسه: {record['id']}")
        print("متن:")
        print(text[:1400])
        print("-" * 80)


if __name__ == "__main__":
    main()
