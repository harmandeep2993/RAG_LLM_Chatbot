import os
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
    
    # Check if the file exists before proceeding
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    try:
        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Iterate through each page and extract text
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:  # Only add text if something was extracted
                    all_text += text + "\n"
                else:
                    print(f"Warning: No text extracted from page {page_num + 1}.")
    except Exception as e:
        print(f"Error while processing the PDF: {e}")
    
    return all_text.strip()

def save_extracted_text(text, output_file):
    """
    Saves the extracted text to a file.
    
    Args:
        text (str): The text to be saved.
        output_file (str): The output file path.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text successfully saved at {output_file}.")
    except Exception as e:
        print(f"Error while saving the text to file: {e}")

if __name__ == "__main__":
    try:
        # Step 1: Extract text from the PDF
        pdf_text = extract_text_from_pdf(config.FAQ_PDF_PATH)
        
        if pdf_text:
            # Step 2: Save the extracted text to a file in data/extracted_text
            output_dir = config.EXTRACT_TEXT_DATA_PATH
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "extracted_faq.txt")
            save_extracted_text(pdf_text, output_file)
        else:
            print("No text was extracted from the PDF.")
    
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")