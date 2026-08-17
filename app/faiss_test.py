import faiss
import numpy as np


# 3 vectors, each having 4 dimensions
vectors = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype="float32")


# Normalize vectors
faiss.normalize_L2(vectors)


# Create Inner Product index
index = faiss.IndexFlatIP(4)

# Add normalized vectors
index.add(vectors)

print("Number of vectors:", index.ntotal)


# Query vector
query = np.array([
    [1, 0.1, 0, 0]
], dtype="float32")


# Normalize query
faiss.normalize_L2(query)


# Search top 2
similarities, indices = index.search(query, 2)


print("Similarities:")
print(similarities)

print("Indices:")
print(indices)