from chunker.chunkingTechniques import chunk_recursively, chunk_by_characters, chunk_by_tokens
from extractor.extractingTechniques import extract_using_pymupdf

def process_pdf_and_chunk_to_txt(pdf_file_path, output_txt_file_path, chunk_size, chunk_overlap, chunking_method='recursive'):
    """
    Extracts text from a PDF, chunks it using specified method, and writes chunks to a text file.

    Args:
        pdf_file_path (str): Path to the input PDF file.
        output_txt_file_path (str): Path to the output text file to write chunks.
        chunk_size (int): Size of each chunk (characters or tokens based on method).
        chunk_overlap (int): Overlap between chunks (characters or tokens based on method).
        chunking_method (str, optional): Chunking method to use.
            Options: 'recursive', 'characters', 'tokens'. Defaults to 'recursive'.
    """
    try:
        # 1. Extract text from PDF
        extracted_text = extract_using_pymupdf(pdf_file_path)
        if not extracted_text:
            print("Text extraction failed. Please check the PDF file or error messages.")
            return

        # 2. Chunk the extracted text based on the chosen method
        if chunking_method == 'recursive':
            chunks = chunk_recursively(extracted_text, chunk_size, chunk_overlap)
        elif chunking_method == 'characters':
            chunks = chunk_by_characters(extracted_text, chunk_size, chunk_overlap)
        elif chunking_method == 'tokens':
            chunks = chunk_by_tokens(extracted_text, chunk_size, chunk_overlap)
        else:
            print(f"Invalid chunking method: {chunking_method}. Using 'recursive' as default.")
            chunks = chunk_recursively(extracted_text, chunk_size, chunk_overlap)

        # 3. Write chunks to a text file
        with open(output_txt_file_path, 'w', encoding='utf-8') as txt_file:
            for chunk in chunks:
                txt_file.write(chunk + "\n")  # Write each chunk to a new line

        print(f"PDF processed and chunks written to: {output_txt_file_path}")

    except Exception as e:
        print(f"An error occurred during the process: {e}")


if __name__ == "__main__":
    # Example usage:
    pdf_input_path = 'example.pdf'  
    txt_output_path = 'output_chunks.txt' 
    chunk_size_val = 150  
    chunk_overlap_val = 30 
    method = 'recursive' 

    try:
        import os
        if not os.path.exists(pdf_input_path):
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(pdf_input_path)
            textobject = c.beginText()
            textobject.setTextOrigin(10, 720)
            textobject.setFont("Helvetica", 12)
            dummy_text = "This is a sample PDF file for testing the script.\nIt contains multiple lines and paragraphs to test text extraction and chunking.\nThis is line 3.\nLine 4 is here. And this is the fifth line."
            for line in dummy_text.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            c.save()
            print(f"Dummy PDF file '{pdf_input_path}' created for testing.")
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")
        pdf_input_path = None # Indicate no pdf is available for test


    if pdf_input_path and os.path.exists(pdf_input_path):
        process_pdf_and_chunk_to_txt(pdf_input_path, txt_output_path, chunk_size_val, chunk_overlap_val, method)
    else:
        print("Please provide a valid PDF file path or ensure example.pdf is in the same directory.")