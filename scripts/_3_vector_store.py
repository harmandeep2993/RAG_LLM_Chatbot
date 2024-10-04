import os
import _0_config
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

def load_chunks(chunk_dir=_0_config.CHUNK_DATA_PATH):
    """
    Loads all text chunks from the specified directory.
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
    """
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from the given embeddings.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_faiss_index(index, index_path=_0_config.VECTOR_STORE_PATH):
    """
    Saves the FAISS index to the specified file path.
    """
    directory = os.path.dirname(index_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    # Step 1: Load the text chunks
    try:
        text_chunks = load_chunks()
        print(f"Loaded {len(text_chunks)} chunks.")
    except Exception as e:
        print(f"Error loading chunks: {e}")
        exit()

    # Step 2: Create embeddings for the chunks
    try:
        embeddings = create_embeddings(text_chunks)
        print(f"Created embeddings of shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        exit()

    # Step 3: Create a FAISS index from the embeddings
    try:
        index = create_faiss_index(embeddings)
        save_faiss_index(index)
        print(f"FAISS index created and saved at {_0_config.VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Error creating or saving FAISS index: {e}")