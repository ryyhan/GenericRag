from sentence_transformers import SentenceTransformer

def convert_chunks_to_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Convert a list of text chunks into embeddings using a specified model.
    
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

"""

 Example usage with default model
if __name__ == "__main__":
    chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language.",
        "Machine learning models can process text data efficiently."
    ]
    
    # Using default model
    embeddings_default = convert_chunks_to_embeddings(chunks)
    print(f"\nDefault model embeddings (dimension: {len(embeddings_default[0])}):")
    print(embeddings_default[0][:5], "...")  # Show first 5 elements
    
    # Using a different model
    embeddings_custom = convert_chunks_to_embeddings(chunks, model_name='paraphrase-MiniLM-L3-v2')
    print(f"\nCustom model embeddings (dimension: {len(embeddings_custom[0])}):")
    print(embeddings_custom[0][:5], "...")

"""