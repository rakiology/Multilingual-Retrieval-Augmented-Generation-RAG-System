# Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions.The system uses a PDF of the story, extracts the text, chunks it, creates embeddings, and then uses a generative model to answer user questions based on the retrieved context.

## Setup Guide

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.9+ installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    You will also need to install Tesseract OCR. You can find installation instructions [here](https://github.com/tesseract-ocr/tesseract#installing-tesseract). Make sure to add the Tesseract executable to your system's PATH.

3.  **Configure API Keys:**
    -   **Google Gemini:** Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and replace `"YOUR_API_KEY"` in `app.py` with your actual key.

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The Flask server will start on `http://127.0.0.1:5000`. The first time you run the application, it will process the PDF, create the embeddings, and save the index. This might take a few minutes.

## Used Tools, Libraries, and Packages

*   **Flask:** A micro web framework for Python used to create the API.
*   **PyMuPDF (fitz):** Used to extract text and images from the PDF file.
*   **Pytesseract:** An OCR tool used to extract text from the images of the PDF pages. It supports both Bangla and English.
*   **Sentence Transformers:** A library for state-of-the-art sentence, text, and image embeddings.
*   **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search and clustering of dense vectors.
*   **Google Generative AI (Gemini):** Used as the generative model to provide answers based on the retrieved context.
*   **Hugging Face Transformers:** Used for tokenization.

## Sample Queries and Outputs

Here are some sample queries and their expected outputs:

### Bangla

*   **User Question:** অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
    **Expected Answer:** শম্ভুনাথ

*   **User Question:** কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
    **Expected Answer:** মামাকে

*   **User Question:** বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
    **Expected Answer:** ১৫ বছর

## API Documentation

The application provides a single API endpoint to ask questions.

*   **Endpoint:** `/ask`
*   **Method:** `GET`
*   **URL Parameters:**
    `question` (string, required): The question you want to ask.
*   **Success Response (200 OK):**
    ```json
    {
      "question": "Your question here",
      "answer": "The answer from the model"
    }
    ```
*   **Error Response (400 Bad Request):**
    ```json
    {
      "error": "Please provide a 'question' parameter in the URL."
    }
    ```
*   **Example Usage (using curl):**
    ```bash
    curl "http://127.0.0.1:5000/ask?question=অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    ```

## Evaluation Matrix

*This project does not include a formal evaluation matrix. However, the relevance of the results can be manually assessed by comparing the model's answers to the source text.*

## Technical Q&A

### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
I used a combination of PyMuPDF (fitz) and Pytesseract OCR.

PyMuPDF was initially used to attempt direct text extraction, as the PDF appeared to be text-based. However, despite being a text PDF, none of the standard methods (including extract_text() and similar functions from PyMuPDF, pdfminer.six, and PyPDF2) could reliably extract any readable content — possibly due to unusual font encoding, embedded glyphs, or layout complexity.

As a fallback, I rendered each page as an image using PyMuPDF, then applied Pytesseract OCR to perform Optical Character Recognition. I chose Pytesseract because it supports multiple languages — including both Bangla (ben) and English (eng) — which were present in the document.

Yes, formatting was a challenge. The OCR output contained irregular line breaks, unpredictable whitespace, and occasionally merged or split words. This made it difficult to chunk the text cleanly or maintain the original paragraph structure, requiring additional post-processing to improve readability.

### 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

I chose a **sentence-based chunking strategy**. The text is first split into individual sentences using the 'দাঁড়ি' (।) character as a delimiter. Then, these sentences are grouped together into chunks of approximately 150 tokens, with an overlap of 30 tokens between consecutive chunks.

This strategy works well for semantic retrieval for several reasons:

*   **Preserves Semantic Integrity:** By splitting at sentence boundaries, we ensure that we don't break up a sentence in the middle, which would destroy its meaning.
*   **Consistent Chunk Size:** Grouping sentences into chunks of a fixed token size creates more uniform and manageable pieces of text for the embedding model to process.
*   **Contextual Overlap:** The overlap between chunks ensures that the context is not lost at the boundaries of the chunks. A sentence that is at the end of one chunk will also be at the beginning of the next, providing a smoother transition of context for the retrieval system.

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

I used the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model.

I chose this model for the following reasons:

*   **Multilingual Support:** As the source text and potential queries could be in both Bangla and English, a multilingual model was essential. This model is trained on over 50 languages and can create comparable embeddings across them.
*   **High Performance:** It is a high-performing model that is well-suited for semantic search and information retrieval tasks.
*   **Sentence-Level Embeddings:** It is specifically designed to create meaningful sentence-level embeddings, which aligns perfectly with my sentence-based chunking strategy.

This model captures the meaning of the text by mapping sentences to a high-dimensional vector space. Sentences with similar meanings are placed closer together in this space, while sentences with different meanings are placed further apart. This is achieved through its training on a massive dataset of parallel and monolingual text, allowing it to understand the nuances of language and semantics.

### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

I am using **FAISS (Facebook AI Similarity Search)** to store the chunk embeddings and perform the similarity search.

When a query comes in, it is first converted into an embedding using the same `paraphrase-multilingual-mpnet-base-v2` model. Then, FAISS is used to find the `k` most similar chunk embeddings from the stored index. The similarity is measured by the **Euclidean distance** (L2 distance) between the query embedding and the chunk embeddings.

I chose this setup for the following reasons:

*   **Efficiency:** FAISS is highly optimized for fast similarity searches, even on very large datasets. This is crucial for a real-time question-answering system.
*   **Scalability:** FAISS can handle millions of vectors, making it a scalable solution for larger documents or collections of documents.
*   **Effectiveness:** Euclidean distance is a standard and effective metric for measuring similarity in vector spaces, and it works well with the embeddings produced by sentence-transformer models.

### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

Meaningful comparison is ensured by using the **same embedding model for both the query and the document chunks**. This maps them into the same vector space, allowing for a direct and meaningful comparison of their semantic similarity.

If a query is vague or missing context, the retrieval system might struggle to find relevant chunks. The query embedding would be located in a region of the vector space that is not close to any of the document chunk embeddings. In this case:

*   The system would retrieve the chunks that are "closest" in the vector space, even if they are not a perfect match.
*   The generative model would then be presented with a context that is not highly relevant to the query.
*   The model would likely respond by stating that the information is not available in the provided text, or it might try to generate an answer based on the tangentially related information, which could be inaccurate. The prompt engineering I did helps to mitigate this by instructing the model on how to behave in such a scenario.

### 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

The results are now highly relevant after the improvements made to the chunking strategy and the correction of the embedding model identifier. The system can now accurately answer specific questions about the text.

However, there is always room for improvement. Here are some things that could further enhance the results:

*   **Better Chunking:** While the current sentence-based chunking is effective, more advanced strategies could be explored, such as recursive chunking or using a library like `langchain` for more sophisticated text splitting.
*   **Better Embedding Model:** While `paraphrase-multilingual-mpnet-base-v2` is a strong model, newer or more specialized models might provide even better performance. For example, a model specifically fine-tuned on Bengali literature could yield superior results.
*   **Larger Context for the LLM:** Increasing the number of retrieved chunks (`k` in the `index.search` call) could provide the language model with more context, potentially leading to more comprehensive answers (though this might trade off with conciseness).
*   **Fine-tuning the Generative Model:** For a production-level system, fine-tuning the generative model on a dataset of question-answer pairs from the source text would significantly improve its accuracy and ability to generate answers in the desired style.
