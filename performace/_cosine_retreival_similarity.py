import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Append the correct path to the folder containing _4_query_handler.py
sys.path.append("C:/Users/harma/helpbee_chatbot")  # Adjust this path to the root directory

# Now import the necessary functions
from _4_query_handler import load_faiss_index, retrieve_top_k_chunks
from scripts._3_vector_store import load_chunks, create_embeddings
from scripts._0_config import CHUNK_DATA_PATH, VECTOR_STORE_PATH

# Initialize the embedding model
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

def cosine_similarity_match(retrieved_chunk, ground_truth_chunk, threshold=0.8):
    """
    Checks if the cosine similarity between the retrieved chunk and the ground truth chunk is above a certain threshold.
    """
    # Encode both the retrieved chunk and the ground truth chunk
    retrieved_embedding = embedder.encode([retrieved_chunk], convert_to_numpy=True)
    ground_truth_embedding = embedder.encode([ground_truth_chunk], convert_to_numpy=True)
    
    # Compute cosine similarity
    similarity = cosine_similarity(retrieved_embedding, ground_truth_embedding)[0][0]
    
    return similarity >= threshold

def evaluate_retrieval(query, index, chunks, ground_truth_chunks, k=3, similarity_threshold=0.9):
    """
    Evaluate the retrieval performance using embedding-based matching (cosine similarity).
    """
    # Embed the user query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    # Search the FAISS index for the top-k closest embeddings
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the corresponding chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # Calculate the number of matches using cosine similarity
    y_true = [1 if any(cosine_similarity_match(chunk, truth_chunk, threshold=similarity_threshold) 
                       for truth_chunk in ground_truth_chunks) else 0 for chunk in chunks]
    
    y_pred = [1 if any(cosine_similarity_match(chunk, truth_chunk, threshold=similarity_threshold) 
                       for truth_chunk in retrieved_chunks) else 0 for chunk in chunks]

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    # Load the FAISS index and text chunks
    index = load_faiss_index()
    text_chunks = load_chunks()

    # Define a sample query and ground truth
    query = "How do I create an account?"
    ground_truth_chunks = ["click on the 'Sign Up' button on the top right corner of our website..."]

    # Evaluate retrieval performance
    # results = evaluate_retrieval(query, index, text_chunks, ground_truth_chunks, k=5)
    results = evaluate_retrieval(query, index, text_chunks, ground_truth_chunks, k=3)

    # Print precision, recall, and F1-score
    print(f"Precision: {results['precision']}")
    print(f"Recall: {results['recall']}")
    print(f"F1-Score: {results['f1']}")