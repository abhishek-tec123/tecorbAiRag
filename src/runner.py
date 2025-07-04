# # # embeddig from file---------------------------------------------------------

# # from textExtractor import batch_extract_text
# # from embed import TextToVectorStore
# # import os

# # os.environ["TOKENIZERS_PARALLELISM"] = "true"

# # files = ["/Users/abhishek/Desktop/ragTecorbAI/beginners_python_cheat_sheet_pcc_all.pdf"]

# # extracted = batch_extract_text(files, max_workers=8)

# # # Combine all extracted text into a single string
# # all_text = "\n".join(extracted.values())

# # vector_handler = TextToVectorStore()

# # # Prepare source info: list of dicts with type and value (using get_source_type_from_filename)
# # sources = []
# # file_names = []
# # for f in files:
# #     name = os.path.basename(f)
# #     file_type = vector_handler.get_source_type_from_filename(name)
# #     sources.append({"type": file_type, "value": name})
# #     file_names.append(name)

# # # Process the combined text as a single document
# # result = vector_handler.process_text(
# #     text=all_text,
# #     vectorstore_path="vectorstores",  # Folder where FAISS index will be saved
# #     source_value=sources,              # List of dicts for each file
# #     file_name=", ".join(file_names)   # Comma-separated file names
# # )

# # print("\n--- Vector Store Info (Combined) ---")
# # print(result)

# # for query---------------------------------------------------------

# from reterver import Retriever
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