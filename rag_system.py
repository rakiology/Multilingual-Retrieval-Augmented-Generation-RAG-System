import sys
import os
import pickle

# Import functions from refactored modules
from text_extractor import extract_text_from_pdf, chunk_text_by_paragraphs
from embed_index import create_and_save_embeddings, load_embedding_data, load_embedding_model
from generator import configure_gemini_api, load_generative_model, answer_question

# Ensure Unicode support for Bangla text in terminal output
sys.stdout.reconfigure(encoding='utf-8')

# --- 1. Configuration ---
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "text_chunks.pkl"
RAW_TEXT_PATH = "raw_text.txt"  # File to save the extracted text
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL = 'gemini-2.5-flash' # Using Gemini 2.5 Flash
# IMPORTANT: You must configure your Google API key below
# You can get one here: https://aistudio.google.com/app/apikey
API_KEY = "YOUR_GEMINI_API_KEY" # Replace with your actual API key

# --- Main Execution ---
if __name__ == "__main__":
    # Check if data processing is needed
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH) or not os.path.exists(RAW_TEXT_PATH):
        print("Index, chunks, or raw text not found. Starting data processing...")
        if os.path.exists(RAW_TEXT_PATH):
            print(f"Loading raw text from {RAW_TEXT_PATH}")
            with open(RAW_TEXT_PATH, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            full_text = extract_text_from_pdf(PDF_PATH)
            # Save the raw extracted text to a file
            with open(RAW_TEXT_PATH, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Raw text extracted from PDF saved to {RAW_TEXT_PATH}")
        
        text_chunks = chunk_text_by_paragraphs(full_text)
        create_and_save_embeddings(text_chunks, EMBEDDING_MODEL, INDEX_PATH, CHUNKS_PATH)
    
    # Load everything
    index, chunks = load_embedding_data(INDEX_PATH, CHUNKS_PATH)
    embedding_model = load_embedding_model(EMBEDDING_MODEL)
    configure_gemini_api(API_KEY)
    generator = load_generative_model(GENERATIVE_MODEL)
    
    # Interactive Q&A loop
    print("\n--- Bangla/English RAG System Ready ---")
    print("Ask a question in Bangla or English. Type 'exit' to quit.")
    
    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() == 'exit':
            break
        
        # Call answer_question with the modified logic
        # The modification to include the first chunk will be handled inside answer_question in generator.py
        answer = answer_question(user_question, index, chunks, embedding_model, generator)
        print(f"\nAnswer: {answer}")
