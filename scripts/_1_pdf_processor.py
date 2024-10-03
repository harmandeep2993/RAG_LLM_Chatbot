import pdfplumber
import os
import config

def extract_text_from_pdf(pdf_path):
    """
    Extracts and returns text from the PDF.
    
    Args:
        pdf_path (str): The file path of the PDF.
        
    Returns:
        str: The extracted text from the PDF.
    """
    all_text = ""
    
    # Open the PDF file using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through each page and extract text
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    
    return all_text

def save_extracted_text(text, output_file):
    """
    Saves the extracted text into a file.
    
    Args:
        text (str): The extracted text from the PDF.
        output_file (str): The file path to save the extracted text.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(config.FAQ_PDF_PATH)

    # Step 2: Save the extracted text to a file in data/extracted_text
    output_dir = "data/extracted_text"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extracted_faq.txt")
    save_extracted_text(pdf_text, output_file)

    print(f"Text extracted and saved at {output_file}.")

'''# 1_pdf_processor.py

import pdfplumber
from config import FAQ_PDF_PATH

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from the given PDF file.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

def split_text_into_chunks(text, chunk_size=500):
    """
    Splits the text into chunks of the given size (default 500 characters).
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

if __name__ == "__main__":
    # Example usage
    faq_text = extract_text_from_pdf(FAQ_PDF_PATH)
    chunks = split_text_into_chunks(faq_text)

    # Print the first chunk as a sample output
    print(f"First chunk:\n{chunks[0]}")
'''