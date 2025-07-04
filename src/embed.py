import os
import time
import logging
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class TextToVectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embed_model=None,
        tokenizer=None
    ):
        self.embed_model = embed_model if embed_model is not None else HuggingFaceEmbeddings(model_name=model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.vectorstore_path = ""
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)

    def get_source_type_from_filename(self, file_name: str) -> str:
        ext = os.path.splitext(file_name)[1].lower()
        if ext == '.pdf':
            return 'pdf_file'
        elif ext == '.txt':
            return 'text_file'
        elif ext in ['.doc', '.docx']:
            return 'word_file'
        elif ext in ['.csv']:
            return 'csv_file'
        else:
            return 'unknown'

    def process_text(
        self,
        text: str,
        vectorstore_path: str = "faiss_index",
        source_type: str = None,
        source_value: str = "user_input",
        file_name: str = "input.txt"
    ) -> Dict[str, Any]:
        logger.info(f"Processing text for vector store: {file_name}")
        self.vectorstore_path = vectorstore_path
        file_name_only = os.path.basename(file_name)
        if source_type is None:
            source_type = self.get_source_type_from_filename(file_name_only)

        # Step 1: Split text
        logger.debug("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = splitter.create_documents([text])
        logger.info(f"Split into {len(split_docs)} chunks.")

        # Tokenizer for counting tokens per chunk
        total_tokens = len(self.tokenizer.encode(text))
        chunk_token_counts = [len(self.tokenizer.encode(doc.page_content)) for doc in split_docs]
        avg_tokens_per_chunk = int(sum(chunk_token_counts) / len(chunk_token_counts)) if chunk_token_counts else 0

        # Step 2: Create vector store
        try:
            logger.info("Embedding and saving to FAISS vector store...")
            start_time = time.time()
            # FAISS.from_documents already uses batch embedding internally if supported
            vector_store = FAISS.from_documents(split_docs, self.embed_model)
            vector_store.save_local(vectorstore_path)
            elapsed_time = time.time() - start_time
            status_info = {
                "index_size": len(split_docs),
                "dimension": vector_store.index.d,
                "embedding_time_seconds": elapsed_time
            }
            logger.info(f"Embedding complete in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            return {
                "status": "error",
                "message": str(e),
                "vectorstore_path": vectorstore_path
            }

        return {
            "status": "success",
            "source": {"type": source_type, "value": source_value},
            "file_name": file_name_only,
            "num_chunks": len(split_docs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vectorstore_path": vectorstore_path,
            "vectorstore_info": status_info,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens_per_chunk
        }

    def load_vector_store(self, path: str):
        if not os.path.exists(path):
            logger.error(f"No FAISS index found at: {path}")
            raise FileNotFoundError(f"No FAISS index found at: {path}")
        logger.info(f"Loading FAISS vector store from {path}")
        self.vector_store = FAISS.load_local(path, self.embed_model, allow_dangerous_deserialization=True)
        self.vectorstore_path = path

    def search(self, query: str, k: int = 3):
        if not self.vector_store:
            logger.error("Vector store not initialized.")
            raise ValueError("Vector store not initialized.")
        logger.info(f"Searching for top {k} results for query: {query}")
        return self.vector_store.similarity_search(query, k=k)
