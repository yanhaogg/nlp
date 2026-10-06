from langchain_community.vectorstores import  Chroma 
from openai import OpenAI
import pickle
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# 1.数据处理
# 1.1 加载数据
with open('./法律条文切片.pkl','rb') as f:
    texts = pickle.load(f)
# 1.2 创建 Document 类
docs = [Document(page_content=text,
    metadata = {'source':'法律条文'})
    for idx,text in enumerate(texts)]
# 2.数据库操作
# 2.1 创建数据库(这里我们更换一下路径)
client = chromadb.PersistentClient(path="Langchain_rag/final_db")

embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)
db = Chroma.from_documents(
    documents= docs,                    
    embedding=embeddings,               
    persist_directory="./final_db",      
    collection_name="law_knowledge",
    ids=[f"law_chunk_{i}" for i in range(len(docs))] # 设置ids
)
# 2.2 数据库增删查
# 2.2.1 查看数据库
collection = db.get()
print(collection["documents"][0])
print(f'文档的数量：{len(collection["documents"])}')
new_docs = [
Document(
    page_content="第十四条 个人信息处理者应当对其处理的个人信息负责...",
    metadata={"chapter": "第二章", "article": "第14条", "source": "个人信息保护法"}
    ),
    Document(
    page_content="第十五条 处理个人信息应当取得个人同意...",
    metadata={"chapter": "第二章", "article": "第15条", "source": "个人信息保护法"}
    )
]
db.add_documents(new_docs,ids=['law_chunk_101','law_chunk_102'])
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')
# 2.2.3 删除数据
db.delete(ids=['law_chunk_1','law_chunk_2'])
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')
db.delete(
where={"chapter": "第二章"}          
# 删除所有第二章的文档
)
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')