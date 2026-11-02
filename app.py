from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import ollama
from chunk import load_document, split_into_chunks

app = FastAPI()

# --- Build the index ONCE, when the app starts ---
client = chromadb.Client()
collection = client.create_collection(name="espresso")
text = load_document("docs/sample.txt")
chunks = split_into_chunks(text)
collection.add(
    documents=chunks,
    ids=[f"chunk-{i}" for i in range(len(chunks))],
)


# --- The shape of the incoming request ---
class Question(BaseModel):
    question: str


def answer_question(question: str):
    results = collection.query(query_texts=[question], n_results=2)
    sources = results["documents"][0]
    context = "\n\n".join(sources)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], sources


@app.post("/ask")
def ask(payload: Question):
    answer, sources = answer_question(payload.question)
    return {"question": payload.question, "answer": answer, "sources": sources}