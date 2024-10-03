import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import VECTOR_STORE_PATH, EMBEDDING_MODEL, CHUNK_DATA_PATH

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

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

def create_embeddings(text_chunks):
    """
    Creates embeddings for a list of text chunks using the specified embedding model.
    
    Args:
        text_chunks (list): List of text chunks to embed.
        
    Returns:
        np.array: An array of embeddings.
    """
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from the given embeddings.
    
    Args:
        embeddings (np.array): Array of embeddings.
        
    Returns:
        faiss.IndexFlatL2: The FAISS index.
    """
    # Initialize FAISS index with L2 distance metric (Euclidean distance)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Add embeddings to the index
    index.add(embeddings)
    
    return index

def save_faiss_index(index, index_path=VECTOR_STORE_PATH):
    """
    Saves the FAISS index to the specified file path.
    
    Args:
        index (faiss.Index): The FAISS index to save.
        index_path (str): File path to save the index.
    """
    # Ensure the directory exists
    directory = os.path.dirname(index_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the FAISS index
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    # Step 1: Load the text chunks
    text_chunks = load_chunks()

    # Step 2: Create embeddings for the chunks
    embeddings = create_embeddings(text_chunks)
    
    # Step 3: Create a FAISS index from the embeddings
    index = create_faiss_index(embeddings)
    
    # Step 4: Save the FAISS index to the vector_store directory
    save_faiss_index(index)
    
    print(f"FAISS index created and saved at {VECTOR_STORE_PATH}")