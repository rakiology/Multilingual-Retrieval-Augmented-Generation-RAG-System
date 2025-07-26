import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

# Configure Tesseract (adjust path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using OCR for both Bangla and English."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Specify both Bangla and English languages
        text += pytesseract.image_to_string(img, lang="ben+eng")
    doc.close()
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def chunk_text_by_paragraphs(text, chunk_size=150, overlap=30):
    """
    Chunks text by sentences into groups of a certain token size.
    This is more robust than splitting by paragraphs.
    """
    from transformers import AutoTokenizer

    # Use a tokenizer to count tokens accurately
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Simple sentence splitting
    sentences = text.split('।') # Using 'দাঁড়ি' as a sentence separator for Bangla
    
    chunks = []
    current_chunk_tokens = []
    current_chunk_text = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = tokenizer.tokenize(sentence)
        
        # If adding the new sentence exceeds the chunk size, save the current chunk
        if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size and current_chunk_tokens:
            chunks.append(" ".join(current_chunk_text))
            
            # Start a new chunk with overlap
            overlap_tokens = int(overlap / 2)
            current_chunk_tokens = current_chunk_tokens[-overlap_tokens:]
            current_chunk_text = tokenizer.convert_tokens_to_string(current_chunk_tokens).split()

        current_chunk_tokens.extend(sentence_tokens)
        current_chunk_text.append(sentence)

    # Add the last chunk if it's not empty
    if current_chunk_text:
        chunks.append(" ".join(current_chunk_text))

    return chunks
