import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Append the correct path to the folder containing _4_query_handler.py
sys.path.append("C:/Users/harma/helpbee_chatbot")  # Adjust this path to the root directory

from scripts._4_query_handler import load_faiss_index, retrieve_top_k_chunks  # Now this import should work
from scripts._0_config import VECTOR_STORE_PATH, CHUNK_DATA_PATH
from scripts._3_vector_store import load_chunks, create_embeddings  # Import from vector store module
from sentence_transformers import SentenceTransformer

# Assuming the embedding model is the same as before
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# More relaxed ground truth matching using substring or keywords
def is_chunk_in_ground_truth(chunk, ground_truth_chunks):
    for truth_chunk in ground_truth_chunks:
        # Check if the ground truth chunk is a substring of the retrieved chunk (case-insensitive)
        if truth_chunk.lower() in chunk.lower():
            return True
    return False

# Modify the evaluate_retrieval function to use the relaxed matching
def evaluate_retrieval(query, index, chunks, ground_truth_chunks, k=5):
    """
    Evaluate the retrieval performance based on precision, recall, and F1 score using relaxed matching.
    """
    # Step 1: Embed the user query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    # Step 2: Search the FAISS index for the top-k closest embeddings
    distances, indices = index.search(query_embedding, k)
    
    # Print distances for debugging
    print(f"Distances: {distances}")
    
    # Step 3: Retrieve the corresponding chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # Print retrieved chunks for debugging
    print("\nRetrieved Chunks:")
    for chunk in retrieved_chunks:
        print(chunk)

    print("\nGround Truth Chunks:")
    for chunk in ground_truth_chunks:
        print(chunk)

    # Step 4: Use relaxed matching for evaluation
    y_true = [1 if is_chunk_in_ground_truth(chunk, ground_truth_chunks) else 0 for chunk in chunks]
    y_pred = [1 if is_chunk_in_ground_truth(chunk, retrieved_chunks) else 0 for chunk in chunks]

    # Step 5: Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def mean_reciprocal_rank(relevant_chunks, retrieved_chunks):
    """
    Compute Mean Reciprocal Rank (MRR) for the query.

    MRR measures the effectiveness of ranking in the retrieval system. It evaluates how far the first relevant
    chunk appears in the list of retrieved chunks. A higher MRR means the system ranks relevant chunks higher.
    
    MRR = 1 / Rank of the first correct chunk
    
    Args:
        relevant_chunks (list): The correct chunks for the query.
        retrieved_chunks (list): The chunks retrieved by FAISS for the query.
    
    Returns:
        float: The MRR score for this query.
    """
    for i, chunk in enumerate(retrieved_chunks):
        if chunk in relevant_chunks:
            return 1 / (i + 1)  # Rank is 1-based
    return 0.0


if __name__ == "__main__":
    # Step 1: Load chunks and the FAISS index
    text_chunks = load_chunks(CHUNK_DATA_PATH)  # Load chunks of text from the chunk directory
    index = load_faiss_index(VECTOR_STORE_PATH)  # Load FAISS index from the stored path
    
    # Step 2: Define a query for evaluation
    query = "How do I create an account?"

    # Step 3: Define the correct (ground truth) chunks for this query
    # These are the chunks that should ideally be retrieved by the system for the given query
    ground_truth_chunks = ["click on the 'Sign Up' button on the top right corner of our website..."]
    
    # Step 4: Evaluate the retrieval system
    results = evaluate_retrieval(query, index, text_chunks, ground_truth_chunks, k=5)
    
    # Step 5: Print Precision, Recall, and F1 scores
    print(f"Precision: {results['precision']}")
    print(f"Recall: {results['recall']}")
    print(f"F1-Score: {results['f1']}")
    
    # Step 6: Compute and print Mean Reciprocal Rank (MRR)
    top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=5)
    mrr_score = mean_reciprocal_rank(ground_truth_chunks, top_chunks)
    print(f"Mean Reciprocal Rank: {mrr_score}")