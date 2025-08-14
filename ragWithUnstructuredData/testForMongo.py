import os
import logging
import requests
import hashlib
import json
from typing import Dict, Any, Optional, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Helpers
# ---------------------------
def generate_custom_id(file_name, length=5):
    """Deterministically generate a short ID from file name."""
    hash_digest = hashlib.sha256(file_name.encode('utf-8')).hexdigest()
    return hash_digest[:length]


def build_embedding_json_for_db(
    chunks: List[Document],
    embeddings: List[List[float]],
    embedding_model_name: str
):
    result = []
    unique_id = None

    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        source = chunk.metadata.get("source", None)
        file_type = "url" if source and source.lower().startswith(("http://", "https://")) else "text_file"
        file_name = os.path.basename(source) if source and not file_type == "url" else source

        if unique_id is None:
            unique_id = generate_custom_id(file_name, 5)

        unique_chunk_id = f"{unique_id}_{idx}"

        entry = {
            "file_name": file_name,
            "file_type": file_type,  # now only "url" or "text_file"
            "unique_id": unique_id,
            "unique_chunk_id": unique_chunk_id,
            "embedding": emb[:3],
            "embedding_size": len(emb),
            "chunk_text": chunk.page_content,
            "embedding_model": embedding_model_name,
            "metadata": chunk.metadata
        }
        result.append(entry)

    return result

# ---------------------------
# Classes
# ---------------------------
class TextExtractor:
    """Extract text from TXT file or URL."""
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


class TextEmbedder:
    """Embed TXT or URL content and return DB-ready JSON."""
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.embed_model = HuggingFaceEmbeddings(model_name=model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_source(
        self,
        source: str,
        preview: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Processing source: {source}")

        # Extract text
        extractor = TextExtractor(source)
        text = extractor.extract(preview)

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = splitter.create_documents([text], metadatas=[{"source": source}])
        logger.info(f"Split into {len(split_docs)} chunks.")

        # Embed chunks
        chunk_texts = [doc.page_content for doc in split_docs]
        embeddings = self.embed_model.embed_documents(chunk_texts)

        # Build DB JSON
        return build_embedding_json_for_db(split_docs, embeddings, self.model_name)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    embedder = TextEmbedder()

    # Example with URL
    db_json = embedder.process_source("https://dummyjson.com/users")

    # Pretty-print for review
    print(json.dumps(db_json, indent=3, ensure_ascii=False))
