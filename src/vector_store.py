
import os
import faiss
import numpy as np
import config
from sentence_transformers import SentenceTransformer

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(config.EMBEDDING_MODEL)

def load_combined_question_answer_texts(chunk_dir=config.CHUNK_DATA_PATH):
    """
    Loads the text chunks (combined question-answer pairs) from the specified directory.
    
    Returns:
        list: A list of combined question-answer texts.
    """
    combined_texts = []
    
    try:
        # Get all files that start with 'chunk_' and end with '.txt'
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")]
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            
            with open(chunk_path, "r", encoding="utf-8") as chunk_f:
                combined_text = chunk_f.read()  # Read both question and answer (combined)
                if combined_text.strip():  # Only add if the text is not empty
                    combined_texts.append(combined_text)
    
    except Exception as e:
        raise Exception(f"Error loading combined question-answer texts: {e}")
    
    if not combined_texts:
        raise Exception("No combined question-answer texts were loaded. Check the input directory.")
    
    return combined_texts

def create_embeddings(text_chunks):
    """
    Creates embeddings for a list of combined question-answer texts using the specified embedding model.
    
    Args:
        text_chunks (list): A list of combined question-answer strings.
    
    Returns:
        np.ndarray: A numpy array of embeddings.
    """
    if not text_chunks:
        raise ValueError("No text chunks provided for embedding.")
    
    try:
        embeddings = embedder.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        raise Exception(f"Error creating embeddings: {e}")
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from the given embeddings.
    
    Args:
        embeddings (np.ndarray): A numpy array of text embeddings.
    
    Returns:
        faiss.IndexFlatL2: A FAISS index of the embeddings.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot create a FAISS index with no embeddings.")
    
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    except Exception as e:
        raise Exception(f"Error creating FAISS index: {e}")
    return index

def save_faiss_index(index, index_path=config.VECTOR_STORE_PATH):
    """
    Saves the FAISS index to the specified file path.
    
    Args:
        index (faiss.IndexFlatL2): The FAISS index to be saved.
        index_path (str): Path where the FAISS index should be saved.
    """
    try:
        directory = os.path.dirname(index_path)
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(index, index_path)
    except Exception as e:
        raise Exception(f"Error saving FAISS index to {index_path}: {e}")

if __name__ == "__main__":
    try:
        # Step 1: Load the combined question-answer texts
        combined_texts = load_combined_question_answer_texts()
        print(f"Loaded {len(combined_texts)} combined question-answer pairs.")
        
        if not combined_texts:
            print("No text data found, aborting process.")
            exit(1)
    except Exception as e:
        print(f"Error loading combined question-answer texts: {e}")
        exit(1)

    # Step 2: Create embeddings for the combined question-answer texts
    try:
        embeddings = create_embeddings(combined_texts)
        print(f"Created embeddings of shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        exit(1)

    # Step 3: Create a FAISS index from the embeddings
    try:
        index = create_faiss_index(embeddings)
        save_faiss_index(index)
        print(f"FAISS index created and saved at {config.VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Error creating or saving FAISS index: {e}")
        exit(1)