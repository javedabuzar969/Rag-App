"""
Embedding utilities using Cohere API.
"""

from typing import List
from cohere import Client as CohereClient


def embed_texts(co: CohereClient, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """
    Embed a list of texts using Cohere embed-english-v3.0 model.
    
    Args:
        co: Cohere client instance
        texts: List of text strings to embed
        input_type: Type of input - "search_document" or "search_query"
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    response = co.embed(model="embed-english-v3.0", texts=texts, input_type=input_type)
    return [list(vec) for vec in response.embeddings]
