from langchain_community.document_loaders import TextLoader
loader = TextLoader("Langchain_rag/data/1.md",encoding="utf-8")
# loader = TextLoader("..../data/1.md",encoding="utf-8")
docs = loader.load()
print("="*40)
print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content[:300])
