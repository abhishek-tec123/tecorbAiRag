import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import shutil
import logging
from dotenv import load_dotenv

from textExtractor import batch_extract_text
from embed import TextToVectorStore
from reterver import Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import auth  # Import the new auth module

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

security = HTTPBearer()

@app.on_event("startup")
def load_models():
    """Load embedding model and tokenizer on startup."""
    try:
        app.state.embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Loaded model and tokenizer: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

# ============================
# 🔹 Upload and Embed Endpoint
# ============================
@app.post("/embed")
async def embed_file(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Upload a file, extract text, and embed it into the vector store."""
    # Verify JWT
    try:
        auth.verify_jwt(credentials.credentials)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    temp_filename = f"/tmp/{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        files = [temp_filename]
        extracted = batch_extract_text(files, max_workers=8)
        all_text = "\n".join(extracted.values())

        vector_handler = TextToVectorStore(
            embed_model=app.state.embed_model,
            tokenizer=app.state.tokenizer
        )
        name = os.path.basename(temp_filename)
        file_type = vector_handler.get_source_type_from_filename(name)
        sources = [{"type": file_type, "value": name}]

        result = vector_handler.process_text(
            text=all_text,
            vectorstore_path="vectorstores",
            source_value=sources,
            file_name=name
        )

        return JSONResponse({
            "message": "Embedding successful",
            "source_type": file_type,
            "vectorstore_info": result
        })

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")


# ============================
# 🔹 Query Endpoint (POST with JSON)
# ============================

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_vectorstore_json(request: QueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Query the vector store and get a response from the LLM."""
    # Verify JWT
    try:
        auth.verify_jwt(credentials.credentials)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    try:
        retriever = Retriever(embed_model=app.state.embed_model)
        retriever.load_vector_store()
        response = await retriever.generate_llm_response(request.query, k=3)
        return JSONResponse({
            "query": request.query,
            "response": response
        })
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-token")
def generate_token(username: str = "admin"):
    """
    Generate a JWT token for a given username (no password required).
    Uses the default supersecretkey from the environment if not set.
    This endpoint is for testing/demo purposes only.
    """
    token = auth.create_jwt({"sub": username})
    return {"access_token": token, "token_type": "bearer"}
