#_2_chunker.py
import re
import os
from _1_pdf_processor import extract_text_from_pdf
import _0_config

def clean_text(text):
    """
    Cleans the extracted text by removing excess newlines and multiple whitespaces,
    and converts the text to lowercase.
    
    Args:
        text (str): The raw text extracted from the PDF.
    
    Returns:
        str: Cleaned and lowercase text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\n+', '\n', text)  # Remove excess newlines
    text = re.sub(r'\s+', ' ', text)   # Eliminate multiple whitespaces
    return text.strip()

def chunk_text_by_question_answer(text):
    """
    Splits the cleaned text into chunks based on the format of numbered questions and answers.
    
    Args:
        text (str): The cleaned and lowercase text containing the FAQs.
    
    Returns:
        list: A list of answers.
        list: A list of corresponding questions.
    """
    # Use regex to capture both "Question" and "Answer" pairs, case-insensitive
    chunks = re.findall(r'\d+\.\s*(.*?)\s*answer:\s*(.*?)(?=\d+\.|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if not chunks:
        print("Warning: No question-answer pairs found.")
    
    # Extract and strip whitespace from both questions and answers
    questions = [q.strip() for q, _ in chunks if q.strip()]
    answers = [a.strip() for _, a in chunks if a.strip()]
    
    # Ensure both lists have the same length (if not, something went wrong in extraction)
    if len(questions) != len(answers):
        print(f"Warning: Mismatch in number of questions ({len(questions)}) and answers ({len(answers)}).")
    
    return answers, questions

def save_chunks_with_metadata(chunks, questions, output_dir=_0_config.CHUNK_DATA_PATH):
    """
    Saves the question-answer chunks into individual text files in the specified directory.
    
    Args:
        chunks (list): A list of answers.
        questions (list): A list of corresponding questions.
        output_dir (str): The directory to save the chunked files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, (chunk, question) in enumerate(zip(chunks, questions)):
        if not chunk or not question:
            print(f"Skipping empty chunk or question at index {idx}.")
            continue
        
        # Create a filename for each chunk
        chunk_filename = os.path.join(output_dir, f"chunk_{idx+1}.txt")
        
        try:
            # Save both the question and the answer in the same file
            with open(chunk_filename, "w", encoding="utf-8") as f:
                f.write(f"Question: {question}\nAnswer: {chunk}")
            print(f"Saved chunk {idx+1} with question and answer.")
        except Exception as e:
            print(f"Error saving chunk {idx+1}: {e}")
    
    print(f"Total chunks saved: {len(chunks)}")

if __name__ == "__main__":
    try:
        # Step 1: Extract the text from the PDF
        pdf_text = extract_text_from_pdf(_0_config.FAQ_PDF_PATH)

        if not pdf_text:
            print("Error: No text extracted from PDF.")
            exit(1)
        
        # Step 2: Clean the text (including converting to lowercase)
        cleaned_text = clean_text(pdf_text)
        
        # Step 3: Chunk the text by question and answer
        text_chunks, questions = chunk_text_by_question_answer(cleaned_text)

        if not text_chunks or not questions:
            print("Error: No valid chunks or questions found.")
            exit(1)

        # Step 4: Save the chunks with metadata (questions)
        save_chunks_with_metadata(text_chunks, questions)

        print(f"{len(text_chunks)} chunks created and saved with corresponding questions.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        