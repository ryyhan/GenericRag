import fitz

def extract_using_pymupdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted plain text from the PDF.
    """
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
        
        # Initialize an empty string to store the extracted text
        text = ""
        
        # Loop through all pages and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load the page
            text += page.get_text("text") + "\n"  # Add a newline between pages
        
        # Strip leading/trailing whitespace from the final text
        return text.strip()
    
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        return None