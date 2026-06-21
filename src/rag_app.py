"""
Main RAG application for employee handbook Q&A.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from cohere import Client as CohereClient

from chunking import build_chunks_from_pdf
from embeddings import embed_texts
from vector_db import ingest_chunks, query_chunks

load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT_ID = os.environ.get("CHROMA_TENANT_ID")
CHROMA_DB_NAME = os.environ.get("CHROMA_DB_NAME", "ragapp")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in environment. Set it in .env.")
if not CHROMA_API_KEY or not CHROMA_TENANT_ID:
    raise RuntimeError("Missing CHROMA_API_KEY or CHROMA_TENANT_ID in environment. Set them in .env.")

co = CohereClient(api_key=COHERE_API_KEY)


def ingest(pdf_path: str):
    """Ingest PDF and store chunks with embeddings in vector DB."""
    print(f"Ingesting PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    chunks = build_chunks_from_pdf(pdf_path)
    if not chunks:
        raise RuntimeError("No text extracted from the PDF.")

    print(f"Created {len(chunks)} chunks from PDF")
    
    texts = [chunk.text for chunk in chunks]
    print("Computing embeddings...")
    embeddings = embed_texts(co, texts, input_type="search_document")
    
    ids = [chunk.id for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"Adding {len(chunks)} chunks to Chroma collection...")
    ingest_chunks(
        CHROMA_API_KEY,
        CHROMA_TENANT_ID,
        CHROMA_DB_NAME,
        ids=ids,
        texts=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print("✓ Ingestion complete.")


def chat_loop():
    """Interactive chat with the handbook."""
    print("Employee Handbook RAG Chat")
    print("Type a question about the handbook, or 'exit' to quit.")
    while True:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        print("Searching handbook...")
        query_embedding = embed_texts(co, [query], input_type="search_query")[0]
        results = query_chunks(
            CHROMA_API_KEY,
            CHROMA_TENANT_ID,
            CHROMA_DB_NAME,
            query_embedding,
            top_k=4,
        )

        print("\nRelevant handbook sections:\n")
        for idx, item in enumerate(results, start=1):
            page = item["metadata"].get("page", "?")
            print(f"--- Result {idx} (page {page}) ---")
            print(item["text"])
            print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG app for employee handbook data")
    parser.add_argument("action", choices=["ingest", "chat"], help="Action to run")
    parser.add_argument(
        "--pdf",
        default="data/Sample Employee Handbook - National Council of Nonprofits.pdf",
        help="Path to the handbook PDF"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    if args.action == "ingest":
        ingest(args.pdf)
    elif args.action == "chat":
        chat_loop()
    else:
        print("Unknown action.")
        sys.exit(1)


if __name__ == "__main__":
    main()
