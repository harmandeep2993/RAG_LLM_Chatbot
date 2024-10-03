
# 1_pdf_processor.py
import pdfplumber
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

if __name__ == "__main__":
    # Extract text from the PDF specified in config.py
    pdf_text = extract_text_from_pdf(config.FAQ_PDF_PATH)

     # Print a sample of the extracted text
    print("Sample extracted text:\n")
    print(pdf_text[:1000])  # Displaying only the first 1000 characters


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