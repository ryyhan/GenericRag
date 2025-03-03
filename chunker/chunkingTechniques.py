from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

# Chunk by characters
def chunk_by_characters(text: str, chunk_size: int, chunk_overlap: int = 0) -> list:
    """
    Chunk the text by a fixed number of characters.
    
    :param text: The input text to be chunked.
    :param chunk_size: The number of characters per chunk.
    :param chunk_overlap: The number of overlapping characters between chunks.
    :return: A list of chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Chunk by tokens
def chunk_by_tokens(text: str, chunk_size: int, chunk_overlap: int = 0) -> list:
    """
    Chunk the text by a fixed number of tokens.
    
    :param text: The input text to be chunked.
    :param chunk_size: The number of tokens per chunk.
    :param chunk_overlap: The number of overlapping tokens between chunks.
    :return: A list of chunks.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Chunk recursively by characters
def chunk_recursively(text: str, chunk_size: int, chunk_overlap: int = 0) -> list:
    """
    Chunk the text recursively by characters.
    
    :param text: The input text to be chunked.
    :param chunk_size: The number of characters per chunk.
    :param chunk_overlap: The number of overlapping characters between chunks.
    :return: A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

"""
Example usage
if __name__ == "__main__":
    print("Chunk by characters:")
    print(chunk_by_characters(text, 50, 10))

    print("\nChunk by tokens:")
    print(chunk_by_tokens(text, 20, 5))

    print("\nChunk recursively by characters:")
    print(chunk_recursively(text, 100, 20))
"""