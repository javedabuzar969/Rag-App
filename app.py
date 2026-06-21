#!/usr/bin/env python3
"""
Entry point for the RAG application.
Run from root: python app.py ingest OR python app.py chat
"""

import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_app import main

if __name__ == "__main__":
    main()
