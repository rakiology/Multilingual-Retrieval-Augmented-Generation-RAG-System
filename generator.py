import google.generativeai as genai

def configure_gemini_api(api_key):
    """Configures the Gemini API with the provided API key."""
    print("Configuring Gemini API...")
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError("Please replace 'YOUR_API_KEY' with your actual Google API key in the script.")
    genai.configure(api_key=api_key)

def load_generative_model(model_name):
    """Loads the generative model."""
    print("Loading generative model...")
    generator = genai.GenerativeModel(model_name)
    return generator

def answer_question(question, index, chunks, embedding_model, generator):
    """Answers a question based on the indexed PDF content."""
    # Embed the question
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()
    
    # Search for the most relevant chunks
    k = 50  # Increased number of chunks to retrieve for more context
    distances, indices = index.search(question_embedding, k)
    
    # Retrieve the relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(relevant_chunks)
    
    # Prepare the prompt for the generative model
    prompt = f"""
    Based on the following context, provide a direct, one or two-word answer to the question. Do not add any extra information or sentences. Use your creative knowledge to answer, not just what you found.

    Context: {context}

    Question: {question}

    Answer:
    """
    
    # Generate the answer with a lower temperature for more focused output
    response = generator.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
    
    return response.text
