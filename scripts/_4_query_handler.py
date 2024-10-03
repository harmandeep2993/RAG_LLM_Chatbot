import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import VECTOR_STORE_PATH, EMBEDDING_MODEL, LANGUAGE_MODEL, CHUNK_DATA_PATH
import os
import re
import nltk
nltk.download('punkt')

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Load the FLAN-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LANGUAGE_MODEL)

# Cache the FAISS index in memory
faiss_index_cache = None

def load_faiss_index(index_path=VECTOR_STORE_PATH):
    """
    Loads the FAISS index from the specified file path. Caches the index in memory.
    """
    global faiss_index_cache
    if faiss_index_cache is None:
        faiss_index_cache = faiss.read_index(index_path)
    return faiss_index_cache

def load_chunks(chunk_dir=CHUNK_DATA_PATH):
    """
    Loads all text chunks from the specified directory and tokenizes them into sentences.
    """
    chunks = []
    for filename in os.listdir(chunk_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(chunk_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                sentences = nltk.sent_tokenize(text)  # Tokenize text into sentences
                chunks.extend(sentences)  # Add sentences as chunks
    return chunks

def retrieve_top_k_chunks(query, index, chunks, k=5, distance_threshold=40.0):
    """
    Retrieves the top-k most relevant chunks from the FAISS index based on the user query.
    """
    # Step 1: Embed the user query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    # Step 2: Search the FAISS index for the top-k closest embeddings
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Retrieve the corresponding chunks, filtered by distance
    top_chunks = []
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:
            top_chunks.append(chunks[indices[0][i]])  # Only keep chunks within threshold
    
    return top_chunks

def filter_relevant_chunks(query, chunks):
    """
    Filters chunks to keep only those that are most relevant to the query based on keyword matching.
    """
    query_keywords = set(query.lower().split())  # Split the query into keywords
    filtered_chunks = [chunk for chunk in chunks if query_keywords.intersection(set(chunk.lower().split()))]
    
    # If no chunks match, return the original FAISS results
    return filtered_chunks if filtered_chunks else chunks

def generate_response(query, context_chunks):
    """
    Generates a response using the FLAN-T5 model based on the provided context.
    """
    if not context_chunks:
        return "Sorry, I don't have information about that. Please ask another question."
    
    # Use the top 2 relevant chunks for more context
    context = " ".join(context_chunks[:2])
    
    # Prepare the input prompt for the language model
    input_text = f"Here is some context: {context} Now answer the query: {query}"

    # Tokenize the input and generate a response using the FLAN-T5 model
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    
    # Decode and return the generated response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Step 1: Load the FAISS index and text chunks
    index = load_faiss_index()
    text_chunks = load_chunks()
    
    # Step 2: Accept a user query
    query = input("Enter your question: ")
    
    # Step 3: Retrieve the top-5 relevant chunks from FAISS
    top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=5)
    
    # Step 4: Filter chunks to keep only the most relevant ones
    relevant_chunks = filter_relevant_chunks(query, top_chunks)
    
    # Step 5: Generate a response based on the query and relevant chunks
    response = generate_response(query, relevant_chunks)
    
    # Output the generated response
    print("\nResponse:\n", response)