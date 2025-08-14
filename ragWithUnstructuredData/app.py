# app.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from reterverFromDmy import Retriever  # your Retriever class file
from loadAndEmbed import TextToVectorStore  # your TextToVectorStore class file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Retriever API",
    description="API for querying FAISS vector store and generating LLM responses",
    version="1.0.0"
)

# Config
VECTORSTORE_PATH = "./vectorstores/url_store"
DEFAULT_SOURCE_URL = "https://dummyjson.com/users"

# Initialize retriever
retriever = Retriever()

@app.on_event("startup")
async def startup_event():
    try:
        # If FAISS store doesn't exist, create it
        if not os.path.exists(VECTORSTORE_PATH):
            logger.warning(f"No FAISS store found at {VECTORSTORE_PATH}. Creating one...")
            processor = TextToVectorStore()
            processor.process_source(DEFAULT_SOURCE_URL, VECTORSTORE_PATH)
            logger.info("✅ New vector store created successfully.")

        # Load vector store
        retriever.load_vector_store(VECTORSTORE_PATH)
        logger.info("✅ Vector store loaded successfully.")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# Request model
class QueryRequest(BaseModel):
    query: str
    k: int = 3
    custom_prompt: str | None = None

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query")
async def query_vectorstore(request: QueryRequest):
    try:
        response = await retriever.generate_llm_response(
            query=request.query,
            k=request.k,
            custom_prompt=request.custom_prompt
        )
        return {"query": request.query, "response": response}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
