import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

def create_and_save_embeddings(chunks, model_name, index_path, chunks_path):
    """Generates embeddings for text chunks and saves them along with the chunks."""
    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings for text chunks...")
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    
    # Ensure embeddings are on CPU and are numpy arrays for FAISS
    embeddings_np = embeddings.cpu().numpy()
    
    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    # Save the index and chunks
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving text chunks to {chunks_path}")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
        
    print("Data processing complete.")

def load_embedding_data(index_path, chunks_path):
    """Loads the FAISS index and text chunks."""
    print("Loading FAISS index and text chunks...")
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def load_embedding_model(model_name):
    """Loads the embedding model."""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(model_name)
    return embedding_model
