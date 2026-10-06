from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("Langchain_rag/data/1.csv",encoding='utf-8')
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)