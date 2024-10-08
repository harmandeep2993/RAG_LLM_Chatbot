import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import _0_config
import os
import re
import nltk
nltk.download('punkt')

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

# Load the FLAN-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(_0_config.LANGUAGE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(_0_config.LANGUAGE_MODEL)

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
def remove_common_labels(text):
    """
    Removes common labels like 'Answer:', 'Response:', etc. from the text.
    Add new labels to the 'labels_to_remove' list as needed.
    """
    labels_to_remove = ["Answer:", "Response:", "Solution:", "Explanation:", "Resolution:"]  # Add more as needed

    # Dynamically remove any labels present in the text
    for label in labels_to_remove:
        if label in text:
            text = text.replace(label, "").strip()  # Remove label and extra spaces

    return text

def load_chunks_and_metadata(chunk_dir=_0_config.CHUNK_DATA_PATH):
    """
    Loads both text chunks (answers) and their corresponding metadata (questions) from the specified directory.
    Dynamically removes common labels like 'Answer:', 'Response:', etc.
    Returns combined question-answer pairs for use in retrieval.
    """
    combined_data = []
    
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")]
    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        metadata_file = f"metadata_{chunk_file.split('_')[1]}"  # Match metadata file based on chunk number
        metadata_path = os.path.join(chunk_dir, metadata_file)

        with open(chunk_path, "r", encoding="utf-8") as chunk_f, open(metadata_path, "r", encoding="utf-8") as metadata_f:
            chunk_text = remove_common_labels(chunk_f.read().strip())  # Remove common labels dynamically
            metadata_text = metadata_f.read().strip()
            combined_text = f"Question: {metadata_text}\nAnswer: {chunk_text}"
            combined_data.append(combined_text)
    
    return combined_data

def retrieve_top_k_chunks(query, index, combined_data, k=5, distance_threshold=30.0):
    """
    Retrieves the top-k most relevant combined question-answer chunks from the FAISS index based on the query.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    top_chunks = []
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:
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

        # Only add chunks that have a reasonable number of keyword matches
        if match_count > 1:  # Adjust this threshold based on testing
            filtered_chunks.append(chunk)

    # If no chunks pass the filter, return the top chunks
    return filtered_chunks if filtered_chunks else chunks

def generate_response(query, context_chunks):
    """
    Generates a response to the user's query using FLAN-T5 with provided context.
    Dynamically removes common labels like 'Answer:', 'Response:', etc.
    Adds conversational structure to improve human-like response generation.
    """
    if not context_chunks:
        return "Sorry, I don't have information about that. Please ask another question."

    # Dynamically remove any labels from the chunks
    context_chunks = [remove_common_labels(chunk) for chunk in context_chunks]

    # Use up to the top 3 relevant chunks for more context
    context = " ".join(context_chunks[:3])

    # Improve the prompt to make the LLM generate a more human-like response
    input_text = f"""You are a helpful assistant. Based on the following information:

{context}

Please provide a detailed and polite answer to the user's question: {query}"""

    # Pass the improved input text to the model
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Allow the model to generate longer responses by increasing max_length
    outputs = model.generate(**inputs, max_length=300)

    # Decode the output and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Step 1: Load the FAISS index and combined question-answer chunks
    index = load_faiss_index()
    combined_data = load_chunks_and_metadata()

    # Step 2: Accept a user query
    query = input("Enter your question: ")

    # Step 3: Retrieve the top-5 relevant combined question-answer chunks from FAISS
    top_chunks = retrieve_top_k_chunks(query, index, combined_data, k=3)

    # Step 4: Filter chunks to keep only the most relevant ones
    relevant_chunks = filter_relevant_chunks(query, top_chunks)

    # Step 5: Generate a response based on the query and relevant chunks
    response = generate_response(query, relevant_chunks)

    # Output the generated response
    print("\nResponse:\n", response)

'''#_4_query_handler.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import _0_config
import os
import re
import nltk
nltk.download('punkt')

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

# Load the FLAN-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(_0_config.LANGUAGE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(_0_config.LANGUAGE_MODEL)

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

def load_chunks(chunk_dir=_0_config.CHUNK_DATA_PATH):
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

def retrieve_top_k_chunks(query, index, chunks, k=5, distance_threshold=30.0):  # Adjusted threshold
    """
    Retrieves the top-k most relevant chunks from the FAISS index based on the query.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    top_chunks = []
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:
            top_chunks.append(chunks[indices[0][i]])

    return top_chunks


# def filter_relevant_chunks(query, chunks):
#     """
#     Filters the chunks to keep only those that are most relevant to the query.
#     """
#     query_keywords = set(query.lower().split())
#     ranked_chunks = []

#     for chunk in chunks:
#         chunk_keywords = set(chunk.lower().split())
#         match_count = len(query_keywords.intersection(chunk_keywords))
#         ranked_chunks.append((chunk, match_count))

#     # Sort by the number of keyword matches (highest first)
#     ranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
#     # Return only chunks with highest matches, or all if no matches
#     if ranked_chunks and ranked_chunks[0][1] > 0:
#         return [chunk[0] for chunk in ranked_chunks if chunk[1] > 0]
    
#     return chunks  # Default to returning all chunks if no match found
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
        if match_count > 1:  # Adjust this threshold based on testing
            filtered_chunks.append(chunk)

    # If no chunks pass the filter, return the top chunks
    return filtered_chunks if filtered_chunks else chunks


def generate_response(query, context_chunks):
    """
    Generates a response to the user's query using FLAN-T5 with provided context.
    """
    if not context_chunks:
        return "Sorry, I don't have information about that. Please ask another question."

    # Use more chunks to provide better context for the response generation
    context = " ".join(context_chunks[:3])  # Use up to the top 3 relevant chunks for more context

    input_text = f"Here is some context: {context} Now answer the query: {query}"

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=200)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Step 1: Load the FAISS index and text chunks
    index = load_faiss_index()
    text_chunks = load_chunks()
    
    # Step 2: Accept a user query
    query = input("Enter your question: ")
    
    # Step 3: Retrieve the top-5 relevant chunks from FAISS
    top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=3)
    
    # Step 4: Filter chunks to keep only the most relevant ones
    relevant_chunks = filter_relevant_chunks(query, top_chunks)
    
    # Step 5: Generate a response based on the query and relevant chunks
    response = generate_response(query, relevant_chunks)
    
    # Output the generated response
    print("\nResponse:\n", response)'''