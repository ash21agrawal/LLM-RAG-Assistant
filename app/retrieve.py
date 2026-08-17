import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/knowledge_base.pkl"


# --------------------------------
# Load existing knowledge base
# --------------------------------

def load_knowledge_base():

    print("Loading existing knowledge base...")

    with open(DATA_PATH, "rb") as file:
        knowledge_base = pickle.load(file)

    chunks = knowledge_base["chunks"]
    embeddings = knowledge_base["embeddings"]

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")

    return chunks, embeddings


# --------------------------------
# Load chunks + embeddings
# --------------------------------

chunks, chunk_embeddings = load_knowledge_base()


# --------------------------------
# Prepare embeddings for FAISS
# --------------------------------

chunk_embeddings = np.asarray(
    chunk_embeddings,
    dtype="float32"
)

# Normalize embeddings
faiss.normalize_L2(chunk_embeddings)


# --------------------------------
# Create / Load FAISS index
# --------------------------------

FAISS_INDEX_PATH = "data/faiss_index.bin"

if os.path.exists(FAISS_INDEX_PATH):

    print("Loading existing FAISS index...")

    index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"FAISS index loaded.")
    print(f"Vectors in index: {index.ntotal}")

else:

    print("Creating FAISS index...")

    embedding_dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)

    # Add normalized embeddings
    index.add(chunk_embeddings)

    # Save FAISS index to disk
    faiss.write_index(
        index,
        FAISS_INDEX_PATH
    )

    print("FAISS index created.")
    print(f"Vectors in index: {index.ntotal}")
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")


# --------------------------------
# Load embedding model
# --------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# Retrieval function
# --------------------------------

def retrieve_chunks(query, top_k=3):

    # Convert query into embedding
    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in range(top_k):

        chunk_index = indices[0][i]
        similarity = similarities[0][i]

        results.append(
            (
                chunks[chunk_index],
                similarity
            )
        )

    return results


# --------------------------------
# Test retrieval
# --------------------------------

if __name__ == "__main__":

    query = input("\nEnter your query: ")

    results = retrieve_chunks(query)

    print("\n===== TOP RESULTS =====")

    for rank, (chunk, score) in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Rank {rank} ---")
        print(f"Similarity: {score:.4f}")
        print(chunk)