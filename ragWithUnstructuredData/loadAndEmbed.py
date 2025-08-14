import os
import time
import logging
import requests
from typing import Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstores")

class TextExtractor:
    """
    Extract text from a TXT file or a URL.
    """
    def __init__(self, source: str):
        self.source = source
        self.is_url = source.lower().startswith(('http://', 'https://'))
        if not self.is_url and not os.path.isfile(source):
            logger.error(f"File not found: {source}")
            raise FileNotFoundError(f"File not found: {source}")

    def extract(self, preview: Optional[int] = None) -> str:
        if self.is_url:
            return self._extract_url(preview)
        else:
            return self._extract_txt(preview)

    def _extract_txt(self, preview: Optional[int]) -> str:
        with open(self.source, 'r', encoding='utf-8') as f:
            return f.read(preview) if preview else f.read()

    def _extract_url(self, preview: Optional[int]) -> str:
        response = requests.get(self.source)
        response.raise_for_status()
        text = response.text
        return text[:preview] if preview else text


class TextToVectorStore:
    """
    Process TXT file or URL text into FAISS vector store.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embed_model = HuggingFaceEmbeddings(model_name=model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.vectorstore_path = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_source(
        self,
        source: str,
        vectorstore_path: Optional[str] = None,
        preview: Optional[int] = None
    ) -> Dict[str, Any]:
        logger.info(f"Processing source: {source}")

        # Extract text
        extractor = TextExtractor(source)
        text = extractor.extract(preview)

        # Set save path
        self.vectorstore_path = vectorstore_path or DEFAULT_VECTORSTORE_PATH

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = splitter.create_documents([text])
        logger.info(f"Split into {len(split_docs)} chunks.")

        # Token counts
        total_tokens = len(self.tokenizer.encode(text))
        chunk_token_counts = [len(self.tokenizer.encode(doc.page_content)) for doc in split_docs]
        avg_tokens_per_chunk = int(sum(chunk_token_counts) / len(chunk_token_counts)) if chunk_token_counts else 0

        # Embed and save
        try:
            logger.info("Embedding and saving to FAISS...")
            start_time = time.time()
            vector_store = FAISS.from_documents(split_docs, self.embed_model)
            vector_store.save_local(self.vectorstore_path)
            elapsed_time = time.time() - start_time

            status_info = {
                "index_size": len(split_docs),
                "dimension": vector_store.index.d,
                "embedding_time_seconds": elapsed_time
            }
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return {"status": "error", "message": str(e)}

        return {
            "status": "success",
            "source": source,
            "num_chunks": len(split_docs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vectorstore_path": self.vectorstore_path,
            "vectorstore_info": status_info,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens_per_chunk
        }
    
# # Example
# if __name__ == "__main__":
#     processor = TextToVectorStore()
#     # file_path = "/Users/abhishek/Desktop/ragTecorbAI/dummJson.txt"
#     # # Example with local text file
#     # result_file = processor.process_source(file_path, "./vectorstores/text_file_store", preview=500)
#     # print(result_file)

#     # Example with URL
#     result_url = processor.process_source("/Users/abhishek/Desktop/ragTecorbAI/dummJson.txt", "./vectorstores/url_store")
#     print(result_url)
