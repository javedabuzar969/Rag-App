"""
Vector database operations using Chroma Cloud.
"""

from typing import Dict, List
import chromadb


CHROMA_COLLECTION_NAME = "employee-handbook"


def get_collection(api_key: str, tenant: str, database: str):
    """
    Get or create a Chroma collection.
    
    Args:
        api_key: Chroma API key
        tenant: Chroma tenant ID
        database: Database name
    
    Returns:
        Chroma collection object
    """
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )
    try:
        return client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        return client.create_collection(name=CHROMA_COLLECTION_NAME)


def ingest_chunks(
    api_key: str,
    tenant: str,
    database: str,
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict[str, str]],
    embeddings: List[List[float]],
):
    """
    Add chunks with embeddings to the Chroma vector database.
    
    Args:
        api_key: Chroma API key
        tenant: Chroma tenant ID
        database: Database name
        ids: Chunk IDs
        texts: Chunk texts
        metadatas: Chunk metadata
        embeddings: Pre-computed embeddings
    """
    collection = get_collection(api_key, tenant, database)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def query_chunks(
    api_key: str,
    tenant: str,
    database: str,
    query_embedding: List[float],
    top_k: int = 4,
) -> List[Dict]:
    """
    Query the vector database for similar chunks.
    
    Args:
        api_key: Chroma API key
        tenant: Chroma tenant ID
        database: Database name
        query_embedding: Query embedding vector
        top_k: Number of results to return
    
    Returns:
        List of result dicts with 'text', 'metadata', 'distance'
    """
    collection = get_collection(api_key, tenant, database)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "text": doc,
            "metadata": metadata,
            "distance": distance,
        })
    return docs
