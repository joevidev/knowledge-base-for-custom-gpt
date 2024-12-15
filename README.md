# Knowledge Base Query API

API for querying a knowledge base stored in Pinecone using LangChain and FastAPI.

## Features

- Semantic search using Pinecone vector store
- Maximum Marginal Relevance (MMR) search for diverse results
- Configurable diversity vs relevance balance
- OpenAI embeddings with latest model
- FastAPI with automatic documentation

## Setup

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- Pinecone API key
- OpenAI API key

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:
