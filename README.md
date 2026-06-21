# Employee Handbook RAG App

A small Retrieval-Augmented Generation app that loads the sample employee handbook PDF, indexes it in Chroma Cloud, and answers handbook questions using Cohere.

## Setup

1. Install packages:

```bash
python -m pip install -r requirements.txt
```

2. Create a `.env` file in the project root with these values:

```env
COHERE_API_KEY=your_cohere_api_key
CHROMA_API_KEY=your_chromadb_api_key
CHROMA_TENANT_ID=your_chromadb_tenant_id
CHROMA_DB_NAME=ragapp
```

3. Ingest the sample handbook:

```bash
python rag_app.py ingest --pdf "data/Sample Employee Handbook - National Council of Nonprofits.pdf"
```

4. Ask questions interactively:

```bash
python rag_app.py chat
```

## Notes

- The app uses Cohere embeddings to vectorize the handbook text.
- It stores chunks in Chroma Cloud so you can query them efficiently.
- The chat mode retrieves relevant passages and generates an answer from them.
