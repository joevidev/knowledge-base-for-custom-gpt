from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from unstructured.staging.base import elements_from_base64_gzipped_json
import os
import uvicorn
from fastapi.responses import JSONResponse

# Configuración de credenciales
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "english-book")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "book-chapters")
API_TOKEN = os.environ.get("API_TOKEN")

# Verificar token requerido
if not API_TOKEN:
    raise RuntimeError(
        "API_TOKEN no está configurado. Por favor, configure la variable de entorno API_TOKEN."
    )

security = HTTPBearer()


app = FastAPI(
    title="API for Knowledge Base of CustomGPT",
    description="API for querying the knowledge base of CustomGPT",
    version="1.0.0",
    openapi_url="/openapi.json",
)


@app.middleware("http")
async def verify_token_middleware(request, call_next):
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    try:
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authentication token not provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth.split(" ")[1]
        if token != API_TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        response = await call_next(request)
        return response

    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=e.headers,
        )


class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to be performed")
    filename: str | None = Field(
        None, description="Optional filename to filter results"
    )
    k: int = Field(4, description="The number of results to return")
    fetch_k: int = Field(
        20, description="Number of documents to fetch before filtering"
    )
    lambda_mult: float = Field(
        0.7,
        description="Diversity of results (0.0 for max diversity, 1.0 for max relevance, 0.7 is a good balance)",
    )


if PINECONE_API_KEY and OPENAI_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY, model="text-embedding-3-small"  # type: ignore
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
            filter=(
                {"filename": {"$eq": request.filename}} if request.filename else None
            ),
        )

        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "orig_elements": (
                        elements_from_base64_gzipped_json(
                            doc.metadata.get("orig_elements", "")
                        )
                        if doc.metadata.get("orig_elements")
                        else None
                    ),
                },
                "diversity_rank": idx,
            }
            for idx, doc in enumerate(results)
        ]

        return {
            "status": "success",
            "query": request.query,
            "filename": request.filename,
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
