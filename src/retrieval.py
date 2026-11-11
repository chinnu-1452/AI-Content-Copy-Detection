
from src.embeddings import generate_embeddings
from src.vector_store import create_vector_store, search_vectors


def retrieve_sources(query_chunks, reference_chunks, top_k=5):

    if not query_chunks or not reference_chunks:
        return []

    reference_texts = [
        item["text"] for item in reference_chunks
    ]

    reference_embeddings = generate_embeddings(reference_texts)
    query_embeddings = generate_embeddings(query_chunks)

    index = create_vector_store(reference_embeddings)

    scores, indices = search_vectors(
        index,
        query_embeddings,
        top_k
    )

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
