from langchain_pymupdf4llm import PyMuPDF4LLMLoader
loader = PyMuPDF4LLMLoader(
    file_path="Langchain_rag/data/1409.4842v1.pdf",
    extract_images=False,  # 不提取图片
)
docs = loader.load()
print(docs)