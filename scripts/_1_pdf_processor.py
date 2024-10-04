import os
import pdfplumber
import _0_config

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
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(_0_config.FAQ_PDF_PATH)

    # Step 2: Save the extracted text to a file in data/extracted_text
    output_dir = _0_config.EXTRACT_TEXT_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extracted_faq.txt")
    save_extracted_text(pdf_text, output_file)

    print(f"Text extracted and saved at {output_file}.")