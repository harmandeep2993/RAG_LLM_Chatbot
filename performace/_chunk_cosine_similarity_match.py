import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scripts._4_query_handler import load_faiss_index, retrieve_top_k_chunks
from scripts._3_vector_store import load_chunks, create_embeddings
from config import VECTOR_STORE_PATH, CHUNK_DATA_PATH
# Append the correct path to the vector_store module
sys.path.append("C:/Users/harma/helpbee_chatbot")  # Adjust this path to the parent directory of _3_vector_store

from scripts._3_vector_store import create_embeddings, load_chunks  # Import functions for creating embeddings and loading chunks
from config import CHUNK_DATA_PATH


def check_cosine_similarity(embeddings):
    """
    Check cosine similarity between the chunks' embeddings.

    Cosine similarity measures how similar two vectors (embeddings) are.
    The value ranges from 0 to 1:
        - 1 means the vectors (chunks) are identical or very similar in content.
        - 0 means the vectors (chunks) are completely dissimilar or unrelated.
    
    Higher Cosine Similarity (closer to 1):
        Indicates that two chunks are very similar in meaning or content, meaning their embeddings are close together 
        in the vector space.
    
    Lower Cosine Similarity (closer to 0):
        Indicates that two chunks are quite different in meaning or content, meaning their embeddings are far apart 
        in the vector space.
    
    Example:
        - If two chunks describe how to "create an account," their cosine similarity will likely be high (close to 1), 
          meaning they contain similar information.
        - If one chunk talks about "tracking an order" and another talks about "payment methods," the cosine similarity 
          between these chunks will likely be low (closer to 0), indicating they are not related.
    
    Args:
        embeddings (np.array): Array of embeddings for the chunks.
        
    Returns:
        np.array: Cosine similarity matrix, where each value represents the similarity between two chunks.
    """
    # Compute the cosine similarity matrix between all embeddings
    cosine_sim_matrix = cosine_similarity(embeddings)
    
    return cosine_sim_matrix

if __name__ == "__main__":
    # Load chunks from the chunk directory
    text_chunks = load_chunks(CHUNK_DATA_PATH)
    
    # Create embeddings for the chunks
    embeddings = create_embeddings(text_chunks)
    
    # Compute cosine similarity between the embeddings
    cosine_sim_matrix = check_cosine_similarity(embeddings)
    
    # Print a portion of the cosine similarity matrix for the first 5 chunks to observe closeness
    # The values represent how semantically similar each chunk is to every other chunk
    print("Cosine Similarity Matrix (First 5x5):")
    print(cosine_sim_matrix[:5, :5])  # Displaying the first 5x5 part of the matrix for simplicity