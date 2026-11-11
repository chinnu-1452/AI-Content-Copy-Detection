
import faiss
import numpy as np


def create_vector_store(embeddings):

    if len(embeddings) == 0:
        raise ValueError("No embeddings available")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return index


def search_vectors(index, query_embeddings, top_k=5):

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        np.asarray(query_embeddings, dtype="float32"),
        top_k
    )

    return scores, indices
