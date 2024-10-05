import os
import faiss
import numpy as np
import _0_config
from sentence_transformers import SentenceTransformer

# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

def load_chunks_and_metadata(chunk_dir=_0_config.CHUNK_DATA_PATH):
    """
    Loads both the text chunks (answers) and their corresponding metadata (questions) from the specified directory.
    Combines the question and answer for each chunk into a single unit.
    
    Returns:
        list: A list of combined question-answer texts.
    """
    combined_texts = []
    
    try:
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")]
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            metadata_file = f"metadata_{chunk_file.split('_')[1]}"  # Match metadata file based on chunk number
            metadata_path = os.path.join(chunk_dir, metadata_file)
            
            with open(chunk_path, "r", encoding="utf-8") as chunk_f, open(metadata_path, "r", encoding="utf-8") as metadata_f:
                chunk_text = chunk_f.read()
                metadata_text = metadata_f.read()
                combined_text = f"{metadata_text}\n{chunk_text}"  # Combine question (metadata) and answer (chunk)
                combined_texts.append(combined_text)
    except Exception as e:
        raise Exception(f"Error loading chunks and metadata: {e}")
    
    return combined_texts

def create_embeddings(text_chunks):
    """
    Creates embeddings for a list of combined question-answer texts using the specified embedding model.
    """
    try:
        embeddings = embedder.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        raise Exception(f"Error creating embeddings: {e}")
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from the given embeddings.
    """
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    except Exception as e:
        raise Exception(f"Error creating FAISS index: {e}")
    return index

def save_faiss_index(index, index_path=_0_config.VECTOR_STORE_PATH):
    """
    Saves the FAISS index to the specified file path.
    """
    try:
        directory = os.path.dirname(index_path)
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(index, index_path)
    except Exception as e:
        raise Exception(f"Error saving FAISS index to {index_path}: {e}")

if __name__ == "__main__":
    # Step 1: Load the combined question-answer texts
    try:
        combined_texts = load_chunks_and_metadata()
        print(f"Loaded {len(combined_texts)} combined question-answer pairs.")
    except Exception as e:
        print(f"Error loading chunks and metadata: {e}")
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
        print(f"FAISS index created and saved at {_0_config.VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Error creating or saving FAISS index: {e}")
        exit(1)


'''#_3_vector_store.py 

 This script converts chunks into embeddings and creates a FAISS index for them.

Step 1: Load the chunk files generated using the _2_chunker.py script and create embeddings for these chunks 
        using the embedding model defined in the _0_config.py file.

Step 2: After generating the embeddings, a FAISS vector index is created to store these embeddings in a vector 
        store. Since we are not utilizing a dedicated vector database, the embeddings will be saved locally in 
        a file with a .index extension, which will be stored in the vector_store directory.

import os
import faiss
import numpy as np
import _0_config
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
        print(f"Error creating or saving FAISS index: {e}")'''