def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


if __name__ == "__main__":
    text = load_document("docs/sample.txt")
    chunks = split_into_chunks(text)

    print(f"Document length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("\n--- First chunk ---")
    print(chunks[0])
    print("\n--- Second chunk ---")
    print(chunks[1])