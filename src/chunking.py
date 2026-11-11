
def split_into_chunks(text, chunk_size=120, overlap=30):

    words = text.split()

    chunks = []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = words[start:start + chunk_size]

        if len(chunk) < 15:
            continue

        chunks.append(" ".join(chunk))

    return chunks
