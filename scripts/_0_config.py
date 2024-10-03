# config.py

# Path to the PDF containing FAQ data
FAQ_PDF_PATH = "data/Miscellaneous Frequent Question and Answers (F&Q).pdf"

# Path to stroe the extract data
EXTRACT_TEXT_DATA_PATH="data/extracted_text"

# Path to chunk data
CHUNK_DATA_PATH = "data/chunk_data/"

# Paths to vector store
VECTOR_STORE_PATH = "vector_store/faq_store.index"

# Embedding and language model names
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
LANGUAGE_MODEL = "google/flan-t5-large"