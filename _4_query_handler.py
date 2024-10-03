import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import VECTOR_STORE_PATH, EMBEDDING_MODEL, LANGUAGE_MODEL, CHUNK_DATA_PATH
import os

# Load the embedding model (multi-qa-mpnet-base-dot-v1) to embed the user query
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Load the FLAN-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LANGUAGE_MODEL)

faiss_index_cache = None

def load_faiss_index(index_path=VECTOR_STORE_PATH):
    """
    Loads the FAISS index from the specified file path. Caches the index in memory.
    
    Args:
        index_path (str): The file path of the FAISS index.
        
    Returns:
        faiss.IndexFlatL2: The FAISS index.
    """
    global faiss_index_cache
    if faiss_index_cache is None:
        faiss_index_cache = faiss.read_index(index_path)
    return faiss_index_cache

def load_chunks(chunk_dir=CHUNK_DATA_PATH):
    """
    Loads all text chunks from the specified directory.
    
    Args:
        chunk_dir (str): Directory where the chunked text files are stored.
        
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    for filename in os.listdir(chunk_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(chunk_dir, filename), "r", encoding="utf-8") as f:
                chunks.append(f.read())
    return chunks

def retrieve_top_k_chunks(query, index, chunks, k=5, return_distances=False, distance_threshold=40):
    """
    Retrieves the top-k most relevant chunks from the FAISS index based on the query.
    Args:
        query (str): The user query.
        index (faiss.Index): The FAISS index containing the chunk embeddings.
        chunks (list): List of text chunks corresponding to the embeddings in the index.
        k (int): The number of top relevant chunks to retrieve.
        return_distances (bool): Whether to return distances along with the chunks.
        distance_threshold (float): A threshold to filter out irrelevant chunks based on distance.
    Returns:
        list: A list of the top-k most relevant chunks.
        list: A list of distances if return_distances is True.
    """
    # Step 1: Embed the user query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    # Step 2: Search the FAISS index for the top-k closest embeddings
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Retrieve the corresponding chunks, filter out based on distance
    top_chunks = []
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:  # Only take chunks within the distance threshold
            top_chunks.append(chunks[indices[0][i]])
    
    # If return_distances is True, return both chunks and distances
    if return_distances:
        return top_chunks, distances[0]
    
    print(f"Distances: {distances[0]}")
    print(f"Retrieved chunks: {top_chunks}")
    
    return top_chunks

# def retrieve_top_k_chunks(query, index, chunks, k=3):
#     """
#     Retrieves the top-k most relevant chunks from the FAISS index based on the query.
    
#     Args:
#         query (str): The user query.
#         index (faiss.Index): The FAISS index containing the chunk embeddings.
#         chunks (list): List of text chunks corresponding to the embeddings in the index.
#         k (int): The number of top relevant chunks to retrieve.
        
#     Returns:
#         list: A list of the top-k most relevant chunks.
#     """
#     # Step 1: Embed the user query
#     query_embedding = embedder.encode([query], convert_to_numpy=True)
    
#     # Step 2: Search the FAISS index for the top-k closest embeddings
#     distances, indices = index.search(query_embedding, k)
    
#     # Print the distances to check similarity
#     print(f"Distances: {distances[0]}")
    
#     # Step 3: Retrieve the corresponding chunks
#     top_chunks = [chunks[i] for i in indices[0]]
    
#     return top_chunks

def generate_response(query, context_chunks):
    """
    Generates a response to the user's query using FLAN-T5 with provided context.
    
    Args:
        query (str): The user's query.
        context_chunks (list): A list of relevant chunks (text) to use as context.
        
    Returns:
        str: The generated response or a fallback message.
    """
    if not context_chunks:  # If no chunks were provided
        return "Sorry, I don't have information about that. Please ask another question."

    # Use only the most relevant chunk (or top 1-2 chunks)
    context = " ".join(context_chunks[:1])  # Use just the first relevant chunk
    
    # Prepare the prompt strictly using the context
    input_text = f"Here is some context: {context} Now answer the query: {query}"

    # Print the input for debugging purposes
    print(f"\nFLAN-T5 Input (Improved Structure): {input_text}\n")

    # Tokenize the input for the model
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the output using the model
    outputs = model.generate(**inputs, max_length=200)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def filter_relevant_chunks(query, chunks):
    """
    Filters the chunks to keep only those that are most relevant to the query.
    This can be based on simple keyword matching or more advanced logic.
    
    Args:
        query (str): The user's query.
        chunks (list): List of chunks retrieved from FAISS.
        
    Returns:
        list: A filtered list of chunks relevant to the query.
    """
    query_keywords = query.lower().split()  # Break the query into keywords
    filtered_chunks = [chunk for chunk in chunks if any(keyword in chunk.lower() for keyword in query_keywords)]
    
    # If no chunks match, return the original chunks
    return filtered_chunks if filtered_chunks else chunks

if __name__ == "__main__":
    # Step 1: Load the FAISS index and text chunks
    index = load_faiss_index()
    text_chunks = load_chunks()
    
    # Step 2: Accept a user query
    query = input("Enter your question: ")
    
    # Step 3: Retrieve the top-5 relevant chunks
    top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=5)
    
    # Step 4: Filter chunks to keep only the most relevant
    relevant_chunks = filter_relevant_chunks(query, top_chunks)

   
    # Step 5: Generate a response based on the query and relevant chunks
    response = generate_response(query, relevant_chunks)
    
    # Output the response
    print("\nResponse:\n", response)