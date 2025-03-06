import os
from extractor.extractingTechniques import extract_using_pymupdf
from chunker.chunkingTechniques import chunk_by_characters, chunk_by_tokens, chunk_recursively
from embedder.embeddingTechniques import convert_chunks_to_embeddings, convert_chunks_to_openai_embeddings
from pinecone import Pinecone, ServerlessSpec
import numpy as np

def store_embeddings_in_pinecone(
    pdf_path: str,
    pinecone_api_key: str,
    index_name: str,
    embedding_dimension: int,
    chunking_method: str = 'recursive',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_method: str = 'sentence_transformer',
    embedding_model: str = 'all-MiniLM-L6-v2',
    openai_api_key: str = None,
    batch_size: int = 100
):
    """
    Extracts text from a PDF, chunks it, generates embeddings, and stores them in a Pinecone index.

    Args:
        pdf_path (str): Path to the PDF file.
        pinecone_api_key (str): API key for Pinecone.
        index_name (str): Name of the Pinecone index.
        embedding_dimension (int): Dimension of the embeddings (e.g., 384 for 'all-MiniLM-L6-v2', 1536 for 'text-embedding-3-small').
        chunking_method (str, optional): Method for chunking text ('characters', 'tokens', 'recursive'). Defaults to 'recursive'.
        chunk_size (int, optional): Size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
        embedding_method (str, optional): Method for generating embeddings ('sentence_transformer', 'openai'). Defaults to 'sentence_transformer'.
        embedding_model (str, optional): Model name for embeddings. Defaults to 'all-MiniLM-L6-v2'.
        openai_api_key (str, optional): API key for OpenAI if using OpenAI embeddings. Defaults to None.
        batch_size (int, optional): Batch size for upserting to Pinecone. Defaults to 100.

    Returns:
        None: Prints status messages and stores embeddings in Pinecone.
    """
    # Step 1: Extract text from PDF
    text = extract_using_pymupdf(pdf_path)
    if text is None:
        print("Failed to extract text from PDF.")
        return

    # Step 2: Chunk the text
    if chunking_method == 'characters':
        chunks = chunk_by_characters(text, chunk_size, chunk_overlap)
    elif chunking_method == 'tokens':
        chunks = chunk_by_tokens(text, chunk_size, chunk_overlap)
    elif chunking_method == 'recursive':
        chunks = chunk_recursively(text, chunk_size, chunk_overlap)
    else:
        print(f"Invalid chunking method: {chunking_method}. Choose from 'characters', 'tokens', or 'recursive'.")
        return

    if not chunks:
        print("No chunks were created.")
        return

    # Step 3: Generate embeddings
    if embedding_method == 'sentence_transformer':
        embeddings = convert_chunks_to_embeddings(chunks, model_name=embedding_model)
    elif embedding_method == 'openai':
        if openai_api_key is None:
            print("OpenAI API key is required for OpenAI embeddings.")
            return
        embeddings = convert_chunks_to_openai_embeddings(chunks, model_name=embedding_model, api_key=openai_api_key)
    else:
        print(f"Invalid embedding method: {embedding_method}. Choose from 'sentence_transformer' or 'openai'.")
        return

    # Convert embeddings to lists if they are numpy arrays (Pinecone expects lists)
    embeddings = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]

    # Step 4: Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists, create if not
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    index = pc.Index(index_name)

    # Prepare data for upsert
    filename = os.path.basename(pdf_path)
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{filename}_{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })

    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Successfully stored {len(chunks)} embeddings in Pinecone index '{index_name}'.")

# Example usage
if __name__ == "__main__":
    # Example parameters
    pdf_path = "sample.pdf"
    pinecone_api_key = "your-pinecone-api-key"
    index_name = "pdf-embeddings"
    embedding_dimension = 384  # For 'all-MiniLM-L6-v2'; use 1536 for 'text-embedding-3-small'

    # Store embeddings with default settings (SentenceTransformer, recursive chunking)
    store_embeddings_in_pinecone(pdf_path, pinecone_api_key, index_name, embedding_dimension)

    # Example with OpenAI embeddings
    # store_embeddings_in_pinecone(
    #     pdf_path=pdf_path,
    #     pinecone_api_key=pinecone_api_key,
    #     index_name=index_name,
    #     embedding_dimension=1536,
    #     embedding_method='openai',
    #     embedding_model='text-embedding-3-small',
    #     openai_api_key='your-openai-api-key'
    # )