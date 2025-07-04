import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from context import generate_response_from_llm
import asyncio

# Prevent tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Global Configuration ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "/Users/abhishek/Desktop/ragTecorbAI/src/vectorstores"


# === Retriever Class ===
class Retriever:
    def __init__(self, model_name: str = MODEL_NAME, embed_model=None):
        self.embed_model = embed_model if embed_model is not None else HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.vectorstore_path = None

    def load_vector_store(self, path: str = VECTORSTORE_PATH):
        if not os.path.exists(path):
            logger.error(f"No FAISS index found at: {path}")
            raise FileNotFoundError(f"No FAISS index found at: {path}")
        
        logger.info(f"Loading FAISS vector store from {path}")
        self.vector_store = FAISS.load_local(
            path,
            embeddings=self.embed_model,
            allow_dangerous_deserialization=True
        )
        self.vectorstore_path = path

    def search(self, query: str, k: int = 3):
        """Returns list of (Document, score) tuples"""
        if not self.vector_store:
            logger.error("Vector store not loaded. Call load_vector_store() first.")
            raise ValueError("Vector store not loaded.")
        
        logger.info(f"Retrieving top {k} documents with scores for query: {query}")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def simple_retrieve(self, query: str, k: int = 3):
        """Returns list of Documents using as_retriever().invoke()"""
        if not self.vector_store:
            logger.error("Vector store not loaded. Call load_vector_store() first.")
            raise ValueError("Vector store not loaded.")
        
        logger.info(f"Simple retrieving top {k} documents for query: {query}")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def combined_page_content(self, query: str, k: int = 3):
        """Returns concatenated page content from top documents"""
        results = self.search(query, k=k)
        return "\n\n".join([doc.page_content for doc, _ in results])

    async def generate_llm_response(self, query: str, k: int = 3, custom_prompt: str = None, llm_provider: str = "groq"):
        """
        Generate a response using the retrieved context and an external LLM.
        Requires an async generate_response_from_llm(context, query, ...)
        """
        logger.info(f"Generating LLM response for query: {query}")
        context_text = self.combined_page_content(query, k=k)
        return await generate_response_from_llm(
            context_text,
            query=query,
            custom_prompt=custom_prompt,
            llm_provider=llm_provider
        )


# # === Example Usage ===
# async def main():
#     # Initialize the retriever
#     retriever = Retriever()
#     retriever.load_vector_store()

#     query = "convert list of name space in uppercase"
#     print(f"\n🔍 Query: {query}")

#     print("\n=== Method 1: search() → (Document, score) ===")
#     docs_with_scores = retriever.search(query, k=3)
#     for i, (doc, score) in enumerate(docs_with_scores):
#         print(f"\nDocument {i+1} (Score: {score:.4f}):")
#         # print(doc.page_content)

#     # print("\n=== Method 2: simple_retrieve() → Document ===")
#     # docs = retriever.simple_retrieve(query, k=3)
#     # for i, doc in enumerate(docs):
#     #     print(f"\nDocument {i+1}:")
#     #     # print(doc.page_content)

#     print("\n=== generate_llm_response() ===")
#     response = await retriever.generate_llm_response(query, k=3)
#     print("\nLLM Response:\n", response)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())