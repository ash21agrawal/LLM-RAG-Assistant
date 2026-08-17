from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The Transformer uses self-attention.",
    "The Transformer is based on attention mechanisms.",
    "The weather is very pleasant today."
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])

from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(embeddings)

print("\nCosine Similarity Matrix:")
print(similarity)