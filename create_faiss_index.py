import numpy as np
import faiss

EMB_PATH = "card_embeddings.npy"
embeddings = np.load(EMB_PATH).astype("float32")

# dimension of embedding
dim = embeddings.shape[1]

# HNSW index (fast + very accurate)
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 40

print("Building FAISS index...")
index.add(embeddings)

faiss.write_index(index, "cards.faiss")

print("FAISS index saved.")