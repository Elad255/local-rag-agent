import chromadb
from chunk import load_document, split_into_chunks

# 1. Create an in-memory Chroma database + a collection to hold our chunks
client = chromadb.Client()
collection = client.create_collection(name="espresso")

# 2. Load and chunk the document (reusing Session 1's functions)
text = load_document("docs/sample.txt")
chunks = split_into_chunks(text)

# 3. Store the chunks — Chroma embeds each one automatically
collection.add(
    documents=chunks,
    ids=[f"chunk-{i}" for i in range(len(chunks))],
)
print(f"Stored {len(chunks)} chunks in Chroma.\n")

# 4. Ask a question — Chroma finds the most relevant chunks
question = "How many grams for a double shot?"
results = collection.query(query_texts=[question], n_results=2)

print(f"Question: {question}\n")
print("--- Most relevant chunks ---")
for doc in results["documents"][0]:
    print(doc)
    print("---")