from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr, Field
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
import uvicorn

# Configuración de credenciales
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "english-book")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "book-chapters")


class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to be performed")
    k: int = Field(4, description="The number of results to return")
    fetch_k: int = Field(
        20, description="Number of documents to fetch before filtering"
    )
    lambda_mult: float = Field(
        0.7,
        description="Diversity of results (0.0 for max diversity, 1.0 for max relevance, 0.7 is a good balance)",
    )


app = FastAPI(
    title="API for Knowledge Base of CustomGPT",
    description="API for querying the knowledge base of CustomGPT",
    version="1.0.0",
)

if PINECONE_API_KEY and OPENAI_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(OPENAI_API_KEY), model="text-embedding-3-large"
    )

    vector_store = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text", namespace=NAMESPACE
    )


@app.post("/query")
async def query_documents(request: QueryRequest):
    try:

        results = vector_store.max_marginal_relevance_search(
            query=request.query,
            k=request.k,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
            namespace=NAMESPACE,
        )

        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "diversity_rank": idx,
            }
            for idx, doc in enumerate(results)
        ]

        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results),
            "diversity_factor": request.lambda_mult,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": "connected" if vector_store else "disconnected",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
