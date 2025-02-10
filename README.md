# RAG Application

**RAG (Retrieval-Augmented Generation)** 
combines the power of retrieval and large language models (LLMs) to produce responses that are enriched with external knowledge. This repository provides a modular pipeline that:

- **Embeds text** using a transformer model.
- **Indexes and searches** documents with FAISS.
- **Generates responses** using an LLM through Ollama.

## Overview

The pipeline is divided into several components:

- **Embedder**: Uses [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) (e.g., All-MiniLM-L6-v2) to generate vector embeddings for queries and documents.
- **Retriever**: Leverages [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search over a document corpus stored in a text file.
- **Generator**: Uses [Ollama](https://ollama.com) to run LLMs (e.g., `deepseek-r1:1.5b`) for generating responses.
- **Cleaner**: Provides utility functions to clean and normalize text.
- **Pipeline**: Orchestrates the above modules to process a query, retrieve relevant context, and generate an augmented answer.


