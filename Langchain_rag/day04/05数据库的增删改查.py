import chromadb
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv() 

client=chromadb.PersistentClient(path="Langchain_rag/database")
embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)
db = Chroma(
    client=client,
    collection_name="law_knowledge",
    embedding_function=embeddings      # 必须传入
)

#加载：如果collection已经存在，则加载
#创建：如果对应collection不存在，则创建

#1.查看
#获取信息
collection=db.get()
print(collection)
print('='*50)
#print(collection['documents'],end='\n')
for item in collection['documents']:
    print(item)

print('-'*50)
print(f"文档数量：{len(collection['documents'])}")



# 3. 按 id 删除
db.delete(ids=['law_chunk_14','law_chunk_15'])
# 按 metadata 条件删除（推荐）
db.delete(
where={"chapter": "第二章"}          
# 删除所有第二章的文档
)
collection = db.get()
print(collection["documents"])
print(f'文档的数量：{len(collection["documents"])}')


#2.追加
#添加 Document 列表（推荐）
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
db.add_documents(new_docs,
                 ids=['law_chunk_14','law_chunk_15'])
print("✅ 新文档已添加")
collection = db.get()
print(collection["documents"])
print(f'文档的数量：{len(collection["documents"])}')



