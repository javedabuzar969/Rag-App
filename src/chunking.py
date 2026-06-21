"""
PDF text extraction and chunking utilities.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    id: str
    text: str
    metadata: Dict[str, str]


def load_pdf_text(pdf_path: str) -> List[tuple]:
    """
    Extract text from PDF file, returning pages with their page numbers.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of tuples: (page_number, text)
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((page_number, text))
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks_from_pdf(pdf_path: str) -> List[DocumentChunk]:
    """
    Extract and chunk text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of DocumentChunk objects with IDs and metadata
    """
    chunks: List[DocumentChunk] = []
    pages = load_pdf_text(pdf_path)
    for page_number, text in pages:
        page_chunks = chunk_text(text)
        for idx, chunk in enumerate(page_chunks, start=1):
            chunk_id = f"page-{page_number}-chunk-{idx}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=chunk,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": str(page_number),
                }
            ))
    return chunks
