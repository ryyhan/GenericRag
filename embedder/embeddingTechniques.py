from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

def convert_chunks_to_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Convert a list of text chunks into embeddings using a specified SentenceTransformer model.
    
    Args:
        chunks (list of str): The text chunks to convert.
        model_name (str, optional): Name of the pre-trained model to use.
            Defaults to 'all-MiniLM-L6-v2'.
    
    Returns:
        list of numpy.array: The embeddings for each chunk.
    """
    # Cache models to avoid reloading the same model multiple times
    if not hasattr(convert_chunks_to_embeddings, '_models'):
        convert_chunks_to_embeddings._models = {}
    
    # Load model if not already cached
    if model_name not in convert_chunks_to_embeddings._models:
        convert_chunks_to_embeddings._models[model_name] = SentenceTransformer(model_name)
    
    model = convert_chunks_to_embeddings._models[model_name]
    return model.encode(chunks)

def convert_chunks_to_openai_embeddings(chunks, model_name="text-embedding-3-small", api_key=None):
    """
    Convert a list of text chunks into embeddings using OpenAI's API.
    
    Args:
        chunks (list of str): The text chunks to convert.
        model_name (str, optional): Name of the OpenAI embedding model to use.
            Defaults to 'text-embedding-3-small'.
        api_key (str, optional): OpenAI API key. Required for OpenAI API access.
    
    Returns:
        list of numpy.array: The embeddings for each chunk.
    
    Raises:
        ValueError: If API key is not provided.
    """
    if api_key is None:
        raise ValueError("OpenAI API key is required for generating embeddings")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Get embeddings from OpenAI API
    response = client.embeddings.create(
        model=model_name,
        input=chunks
    )
    
    # Extract embeddings and convert to numpy arrays
    embeddings = [np.array(embedding.embedding) for embedding in response.data]
    return embeddings

"""
if __name__ == "__main__":
    chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language.",
        "Machine learning models can process text data efficiently."
    ]
    
    # Using default SentenceTransformer model
    embeddings_default = convert_chunks_to_embeddings(chunks)
    print(f"\nSentenceTransformer embeddings (dimension: {len(embeddings_default[0])}):")
    print(embeddings_default[0][:5], "...")  # Show first 5 elements
"""