import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
CHUNKS_PATH = "text_chunks.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# --- Load Chunks ---
try:
    with open(CHUNKS_PATH, 'rb') as f:
        text_chunks = pickle.load(f)
    print(f"✅ Loaded {len(text_chunks)} chunks from {CHUNKS_PATH}\n")
except FileNotFoundError:
    print(f"❌ Error: {CHUNKS_PATH} not found. Please run the RAG system first.")
    exit()
except Exception as e:
    print(f"❌ Error loading chunks: {e}")
    exit()

# --- Input a Query ---
query = input("\n  অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?: ").strip()

# --- Load Embedding Model ---
print("\n🔄 Encoding chunks and query...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Encode and normalize chunk embeddings
chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True).cpu().numpy()
chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

# Encode and normalize query
query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

# --- Compute Cosine Similarities ---
similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
top_indices = np.argsort(similarities)[::-1][:5]

# --- Show Top Matches ---
print("\n📚 Top 5 relevant chunks:")
for rank, idx in enumerate(top_indices, start=1):
    print(f"\n--- Rank {rank} | Similarity: {similarities[idx]:.4f} ---")
    print(text_chunks[idx])
    print("-" * 40)
