import re
import os
from _1_pdf_processor import extract_text_from_pdf
import _0_config

def clean_text(text):
    """
    Cleans the extracted text, removing unnecessary characters but preserving paragraphs.
    """
    text = re.sub(r'\n+', '\n', text)  # Use raw string for regex
    text = re.sub(r'\s+', ' ', text)   # Use raw string for regex
    return text.strip()
    

def chunk_text_by_question_answer(text):
    """
    Splits the text into chunks, each containing a question and its corresponding answer.
    
    Args:
        text (str): The full extracted text (containing questions and answers).
        
    Returns:
        list: A list of question-answer chunks and the list of questions.
    """
    # Use regex to capture both "Question" and "Answer" pairs
    chunks = re.findall(r'Question\s*\d+:\s*(.*?)\s*Answer:\s*(.*?)(?=Question|\Z)', text, re.DOTALL)
    
    questions = [q for q, _ in chunks]  # Extract questions
    answers = [a for _, a in chunks]    # Extract answers
    
    return answers, questions

def save_chunks_with_metadata(chunks, questions, output_dir=_0_config.CHUNK_DATA_PATH):
    """
    Saves the chunks and metadata as individual text files in the specified directory.
    
    Args:
        chunks (list): A list of answer chunks.
        questions (list): A list of questions corresponding to the chunks.
        output_dir (str): Directory to save the chunked text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, (chunk, question) in enumerate(zip(chunks, questions)):
        chunk_filename = os.path.join(output_dir, f"chunk_{idx+1}.txt")
        metadata_filename = os.path.join(output_dir, f"metadata_{idx+1}.txt")
        
        # Save chunk (the answer)
        with open(chunk_filename, "w", encoding="utf-8") as f:
            f.write(chunk)
        
        # Save metadata (the question)
        with open(metadata_filename, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}")

if __name__ == "__main__":
    # Step 1: Extract the text from the PDF
    pdf_text = extract_text_from_pdf(_0_config.FAQ_PDF_PATH)
   
    # Step 1: Clean the text
    cleaned_text = clean_text(pdf_text)
    
    # Step 2: Chunk the text by question and answer
    text_chunks, questions = chunk_text_by_question_answer(cleaned_text)
    
    # Step 3: Save the chunks with metadata (questions)
    save_chunks_with_metadata(text_chunks, questions)

    print(f"{len(text_chunks)} chunks created and saved with corresponding questions.")


'''''# _2_chunker.py file
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

def save_chunks_with_metadata(chunks, questions, output_dir=CHUNK_DIR):
    """
    Saves the chunks and metadata as individual text files in the specified directory.
    
    Args:
        chunks (list): A list of answer chunks.
        questions (list): A list of questions corresponding to the chunks.
        output_dir (str): Directory to save the chunked text files.
    """
    for idx, (chunk, question) in enumerate(zip(chunks, questions)):
        chunk_filename = os.path.join(output_dir, f"chunk_{idx+1}.txt")
        metadata_filename = os.path.join(output_dir, f"metadata_{idx+1}.txt")
        
        # Save chunk (the answer)
        with open(chunk_filename, "w", encoding="utf-8") as f:
            f.write(chunk)
        
        # Save metadata (the question)
        with open(metadata_filename, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}")


This function do spliting in question and answer pair
def chunk_text_by_answer(text):
    """
    Splits the text into chunks, each containing a question and its corresponding answer.
    
    Args:
        text (str): The full extracted text (containing questions and answers).
        
    Returns:
        list: A list of question-answer chunks.
    """
    # Use regex to capture both "Question" and "Answer" pairs
    chunks = re.findall(r'Question\s*\d+:\s*(.*?)\s*Answer:\s*(.*?)(?=Question|\Z)', text, re.DOTALL)
    
    # Combine each question and answer into a chunk
    return [f"Question: {q}\nAnswer: {a}" for q, a in chunks]


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
    text_chunks = save_chunks_with_metadata(cleaned_text)
    
    # Step 4: Save the chunks to the chunk_data directory
    save_chunks(text_chunks)
    
    # Print the first few chunks to verify their contents
    print(f"First few sample chunks:\n")
    for i, chunk in enumerate(text_chunks[:2]):
        print(f"Chunk {i+1}:\n{chunk}\n")

    print(f"{len(text_chunks)} chunks created and saved in {CHUNK_DIR}.\n")'''''