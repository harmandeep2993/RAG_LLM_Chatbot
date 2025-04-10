import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import _0_config

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

# Cache the FAISS index in memory
faiss_index_cache = None

def load_faiss_index(index_path=_0_config.VECTOR_STORE_PATH):
    """
    Loads the FAISS index from the specified file path. Caches the index in memory.
    """
    global faiss_index_cache
    if faiss_index_cache is None:
        faiss_index_cache = faiss.read_index(index_path)
    return faiss_index_cache

def load_combined_question_answer_texts(chunk_dir=_0_config.CHUNK_DATA_PATH):
    """
    Loads the combined question-answer texts from the specified directory.

    Returns:
        list: A list of combined question-answer texts.
    """
    combined_texts = []
    try:
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")]
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            
            with open(chunk_path, "r", encoding="utf-8") as f:
                combined_text = f.read().strip()
                combined_texts.append(combined_text)
    except Exception as e:
        raise Exception(f"Error loading combined question-answer texts: {e}")
    
    return combined_texts

def retrieve_top_k_chunks(query, index, combined_data, k=5, distance_threshold=0.8):  # Use a reasonable threshold
    """
    Retrieves the top-k most relevant combined question-answer chunks from the FAISS index based on the query.
    Filters based on a distance threshold to ensure only relevant chunks are returned.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    top_chunks = []
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:  # Filter based on distance threshold
            top_chunks.append(combined_data[indices[0][i]])

    return top_chunks

def filter_relevant_chunks(query, chunks):
    """
    Filters the chunks to keep only those that are most relevant to the query.
    """
    query_keywords = set(query.lower().split())
    filtered_chunks = []

    for chunk in chunks:
        chunk_keywords = set(chunk.lower().split())
        match_count = len(query_keywords.intersection(chunk_keywords))

        if match_count > 1:  # Only add chunks with a reasonable number of matches
            filtered_chunks.append(chunk)

    return filtered_chunks if filtered_chunks else []

def generate_response(query, context_chunks):
    """
    Generates a response to the user's query using Mistral via Ollama with provided context (retrieved chunks).
    """
    if not context_chunks:
        return "Sorry, I don't have information about that. Please ask another question."

    # Use the top relevant chunks for context (you can use more chunks if needed)
    context = "\n".join(context_chunks[:2])  # Taking top 2 chunks for context

    # Create a prompt combining the retrieved chunks (context) and the user's query
    prompt = f"""
    You are Helpbee, an assistant designed to respond to customer questions. Based on the information below, answer the user's question clearly and concisely.

    Context: {context}

    Question: {query}
    """

    # Use Ollama to generate the response
    response = ollama.generate(model=_0_config.MODEL_NAME, prompt=prompt)
    
    return response['response']

def get_helpbee_response(query, k=3, distance_threshold=35, confidence_threshold=45):
    """
    Combines all query handling functions:
    - Retrieves top chunks from FAISS.
    - Filters the relevant chunks.
    - Generates a response based on the chunks.
    """
    index = load_faiss_index()
    combined_chunks = load_combined_question_answer_texts()

    # Retrieve top chunks based on distance
    top_chunks = retrieve_top_k_chunks(query, index, combined_chunks, k, distance_threshold)

    # Filter relevant chunks
    relevant_chunks = filter_relevant_chunks(query, top_chunks)

    # Generate response
    if relevant_chunks:
        return generate_response(query, relevant_chunks)
    else:
        return "Sorry, I couldn't find enough relevant information for your question."

if __name__ == "__main__":
    query = input("Enter your question: ")

    # Step 3: Retrieve the top-3 relevant combined question-answer chunks from FAISS
    response = get_helpbee_response(query)

    print("\nResponse:\n", response)


'''import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import _0_config

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

# Cache the FAISS index in memory
faiss_index_cache = None

def load_faiss_index(index_path=_0_config.VECTOR_STORE_PATH):
    """
    Loads the FAISS index from the specified file path. Caches the index in memory.
    """
    global faiss_index_cache
    if faiss_index_cache is None:
        faiss_index_cache = faiss.read_index(index_path)
    return faiss_index_cache

def load_combined_question_answer_texts(chunk_dir=_0_config.CHUNK_DATA_PATH):
    """
    Loads the combined question-answer texts from the specified directory.

    Returns:
        list: A list of combined question-answer texts.
    """
    combined_texts = []
    
    try:
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")]
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            
            with open(chunk_path, "r", encoding="utf-8") as f:
                combined_text = f.read().strip()
                combined_texts.append(combined_text)
    except Exception as e:
        raise Exception(f"Error loading combined question-answer texts: {e}")
    
    return combined_texts

def retrieve_top_k_chunks(query, index, combined_data, k=5, distance_threshold=30):  # Changed threshold for debugging
    """
    Retrieves the top-k most relevant combined question-answer chunks from the FAISS index based on the query.
    Filters based on a distance threshold to ensure only relevant chunks are returned.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    top_chunks = []
    for i, distance in enumerate(distances[0]):
        # Added detailed logging for each distance score and comparison
        print(f"Chunk {i+1}: Distance = {distance}")
        if distance < distance_threshold:  # Filter based on distance threshold
            top_chunks.append(combined_data[indices[0][i]])

    # Display retrieved chunks
    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Chunk {i}:\n{chunk}\n{'-'*80}")

    return top_chunks

def filter_relevant_chunks(query, chunks):
    """
    Filters the chunks to keep only those that are most relevant to the query.
    """
    query_keywords = set(query.lower().split())
    filtered_chunks = []

    for chunk in chunks:
        chunk_keywords = set(chunk.lower().split())
        match_count = len(query_keywords.intersection(chunk_keywords))

        # Only add chunks that have a reasonable number of keyword matches
        if match_count > 1:
            filtered_chunks.append(chunk)

    return filtered_chunks if filtered_chunks else []

def generate_response(query, context_chunks):
    """
    Generates a response to the user's query using Mistral via Ollama with provided context (retrieved chunks).
    """
    if not context_chunks:
        return "Sorry, I don't have information about that. Please ask another question."

    # Use the top relevant chunks for context (you can use more chunks if needed)
    context = "\n".join(context_chunks[:2])  # Taking top 2 chunks for context

    # Create a prompt combining the retrieved chunks (context) and the user's query
    prompt = f""" 
    You are Helpbee, designed to respond to customer questions. You are a helpful assistant and polite. Based on the context information, provide concise response in a paragraph.

    Context: {context}

    Question: {query}
    """

    # Use Ollama to generate the response
    response = ollama.generate(model=_0_config.MODEL_NAME, prompt=prompt)
    
    # Return the generated response
    return response['response']

if __name__ == "__main__":
    # Step 1: Load the FAISS index and combined question-answer texts
    index = load_faiss_index()
    combined_data = load_combined_question_answer_texts()

    # Step 2: Accept a user query
    query = input("Enter your question: ")

    # Step 3: Retrieve the top-3 relevant combined question-answer chunks from FAISS
    top_chunks = retrieve_top_k_chunks(query, index, combined_data, k=3)

    # Step 4: Generate a response based on the query and relevant chunks
    if top_chunks:
        response = generate_response(query, top_chunks)
    else:
        response = "Sorry, this question is out of context. I do not have any match to your query,Please ask another question."

    # Output the generated response
    print("\nResponse:\n", response)'''