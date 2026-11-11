
import json
from pathlib import Path

from src.embeddings import generate_embeddings
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    search_vectors
)

DATABASE_DIR = Path("database")
INDEX_PATH = DATABASE_DIR / "reference.index"
METADATA_PATH = DATABASE_DIR / "reference_chunks.json"


def build_reference_database(reference_chunks):
    if not reference_chunks:
        raise ValueError("No reference chunks available")

    reference_texts = [item["text"] for item in reference_chunks]

    embeddings = generate_embeddings(reference_texts)
    index = create_vector_store(embeddings)

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    save_vector_store(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(reference_chunks, file, ensure_ascii=False, indent=4)

    return len(reference_chunks)


def reference_database_exists():
    return INDEX_PATH.exists() and METADATA_PATH.exists()


def retrieve_sources(query_chunks, top_k=5):

    if not query_chunks:
        return []

    if not reference_database_exists():
        raise FileNotFoundError(
            "Reference database not found. Build it first."
        )

    index = load_vector_store(INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        reference_chunks = json.load(file)

    query_embeddings = generate_embeddings(query_chunks)

    scores, indices = search_vectors(index, query_embeddings, top_k)

    results = []

    for query_index, row in enumerate(indices):

        for rank, ref_index in enumerate(row):

            if ref_index < 0:
                continue

            score = float(scores[query_index][rank])

            results.append({
                "query_chunk": query_chunks[query_index],
                "source_chunk": reference_chunks[ref_index]["text"],
                "source_name": reference_chunks[ref_index]["source"],
                "similarity": round(score * 100, 2)
            })

    return results
