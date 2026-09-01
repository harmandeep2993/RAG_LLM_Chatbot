# config.py

# Path to the PDF containing FAQ data
FAQ_PDF_PATH = "data/raw/Miscellaneous Frequent Question and Answers (F&Q).pdf"

# Path to store the extract data conda
EXTRACT_TEXT_DATA_PATH="data/extracted_text"

# Path to chunk data
CHUNK_DATA_PATH = "data/chunk_data/"

# Paths to vector store
VECTOR_STORE_PATH = "vector_store/faq_store.index"

# Embedding and language model names
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# LANGUAGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Ollama Model for text generation
LANGUAGE_MODEL = "ollama"  # Ollama as the platform
MODEL_NAME = "mistral"  # Using the Mistral model