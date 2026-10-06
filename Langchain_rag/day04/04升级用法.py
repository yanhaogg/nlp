from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv() 


#推理模型
# llm=ChatOpenAI(model=os.getenv('DEEPSEEK'),api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
# res2=llm.invoke('hello')
# print(res2)


from langchain_openai import OpenAIEmbeddings

embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)


#对单个句子进行embed
query_embedding=embeddings.embed_query("什么是rag")
print(query_embedding)
print('='*50)

#对多个query进行embedding
docs=["人工智能","图像识别"]
doc_embeddings=embeddings.embed_documents(docs)
print(doc_embeddings)