from langchain_chroma import Chroma
import chromadb

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)

client = chromadb.PersistentClient(path = 'Langchain_rag/db')

db = Chroma(
    client = client,
    collection_name="nlp_papers",
    embedding_function=embeddings
)

print(len(db.get()["ids"]))