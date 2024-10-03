import os
import re
import _0_config

# Ensure chunk_data directory exists
CHUNK_DIR = "data/chunk_data"
if not os.path.exists(CHUNK_DIR):
    os.makedirs(CHUNK_DIR)

def clean_text(text):
    """
    Cleans the extracted text, removing unnecessary characters but preserving paragraphs.
    """
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces/newlines with a single space
    return text.strip()

def chunk_text_by_answer(text):
    """
    Splits the text into chunks by isolating the answers and keeping them complete.
    
    Args:
        text (str): The full extracted text (containing questions and answers).
        
    Returns:
        list: A list of answer chunks.
    """
    # Use a more flexible regex to handle variations in question formatting
    chunks = re.findall(r'Answer:\s*(.*?)(?=Answer:|\Z)', text, re.DOTALL)
    return chunks

def save_chunks(chunks, output_dir=CHUNK_DIR):
    """
    Saves the chunks as individual text files in the specified directory.
    
    Args:
        chunks (list): A list of text chunks.
        output_dir (str): Directory to save the chunked text files.
    """
    for idx, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_dir, f"chunk_{idx+1}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as f:
            f.write(chunk)

if __name__ == "__main__":
    # Import the PDF extraction function from 1_pdf_processor
    from _1_pdf_processor import extract_text_from_pdf
    
    # Step 1: Extract the text from the PDF
    pdf_text = extract_text_from_pdf(_0_config.FAQ_PDF_PATH)
    
    # Step 2: Clean the text
    cleaned_text = clean_text(pdf_text)
    
    # Step 3: Chunk the text by answer
    text_chunks = chunk_text_by_answer(cleaned_text)
    
    # Step 4: Save the chunks to the chunk_data directory
    save_chunks(text_chunks)
    
    print(f"Chunking completed! {len(text_chunks)} chunks created and saved in {CHUNK_DIR}.")
    
    # Print the first few chunks to verify their contents
    print(f"First few chunks:\n")
    for i, chunk in enumerate(text_chunks[:3]):
        print(f"Chunk {i+1}:\n{chunk}\n")