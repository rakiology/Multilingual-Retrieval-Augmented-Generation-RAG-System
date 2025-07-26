from flask import Flask, request, jsonify
import os
import sys

# Ensure Unicode support for Bangla text in terminal output
sys.stdout.reconfigure(encoding='utf-8')

# Import functions from refactored modules
from text_extractor import extract_text_from_pdf, chunk_text_by_paragraphs
from embed_index import create_and_save_embeddings, load_embedding_data, load_embedding_model
from generator import configure_gemini_api, load_generative_model, answer_question

app = Flask(__name__)

# --- 1. Configuration ---
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "text_chunks.pkl"
RAW_TEXT_PATH = "raw_text.txt"  # File to save the extracted text
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL = 'gemini-2.5-flash' # Using Gemini 1.5 Flash
# IMPORTANT: You must configure your Google API key below
# You can get one here: https://aistudio.google.com/app/apikey
API_KEY = "AIzaSyDynY3P-ZbVoih47AF0bgGOdB8tOKh8fI4" # Replace with your actual API key

# Global variables to store loaded models and data
index = None
chunks = None
embedding_model = None
generator = None

def initialize_rag_system():
    global index, chunks, embedding_model, generator

    if index is None or chunks is None or embedding_model is None or generator is None:
        print("Initializing RAG system...")
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
        print("RAG system initialized.")
    else:
        print("RAG system already initialized.")

@app.route('/ask', methods=['GET'])
def ask_question_api():
    question = request.args.get('question')
    if not question:
        return jsonify({"error": "Please provide a 'question' parameter in the URL."}), 400

    initialize_rag_system() # Ensure RAG system is initialized

    try:
        answer = answer_question(question, index, chunks, embedding_model, generator)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the RAG system when the app starts
    initialize_rag_system()
    app.run(debug=True, port=5000)
