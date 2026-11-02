import chromadb
import ollama
from chunk import load_document, split_into_chunks

# --- Indexing (same as Session 3) ---
client = chromadb.Client()
collection = client.create_collection(name="espresso")

text = load_document("docs/sample.txt")
chunks = split_into_chunks(text)
collection.add(
    documents=chunks,
    ids=[f"chunk-{i}" for i in range(len(chunks))],
)

# --- Query ---
question = "What is the capital of France?"

# 1. Retrieve the most relevant chunks
results = collection.query(query_texts=[question], n_results=2)
retrieved_chunks = results["documents"][0]
context = "\n\n".join(retrieved_chunks)

# 2. Build a prompt that grounds the LLM in the retrieved context
prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

# 3. Ask the local LLM
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
)

print(f"Question: {question}\n")
print("Answer:")
print(response["message"]["content"])