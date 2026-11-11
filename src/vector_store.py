
import faiss
import numpy as np
from pathlib import Path


def create_vector_store(embeddings):
    if len(embeddings) == 0:
        raise ValueError("No embeddings available")

    embeddings = np.asarray(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_vector_store(index, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(file_path))


def load_vector_store(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError("FAISS database not found. Build it first.")

    return faiss.read_index(str(file_path))


def search_vectors(index, query_embeddings, top_k=5):
    if index.ntotal == 0:
        return np.array([]), np.array([])

    top_k = min(top_k, index.ntotal)
    query_embeddings = np.asarray(query_embeddings, dtype="float32")

    scores, indices = index.search(query_embeddings, top_k)

    return scores, indices

