from chromadb.utils import embedding_functions
import numpy as np

# The embedder turns text into vectors (Chroma's built-in small model)
embedder = embedding_functions.DefaultEmbeddingFunction()

texts = [
    "I love espresso",
    "Coffee is my favorite drink",
    "The weather is cold today",
]

vectors = embedder(texts)

print(f"Number of texts: {len(texts)}")
print(f"Each vector has {len(vectors[0])} numbers (dimensions)")
print("\nFirst 10 numbers of the first vector:")
print(vectors[0][:10])


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n--- How close are these in meaning? (1.0 = identical) ---")
print("espresso vs coffee: ", cosine_similarity(vectors[0], vectors[1]))
print("espresso vs weather:", cosine_similarity(vectors[0], vectors[2]))