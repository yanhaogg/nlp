from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Langchain_rag/data/1409.4842v1.pdf")
docs=loader.load()
print(docs)