import chromadb
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_core.documents import Document


#已经经过04的升级，明确embedding_function怎么定义
load_dotenv() 




#1.准备数据库
from langchain_openai import OpenAIEmbeddings

client=chromadb.PersistentClient(path="Langchain_rag/database")
embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)

# 创建 Chroma 向量库（LangChain 封装版本）
db = Chroma(
    client=client,
    collection_name="law_knowledge",
    embedding_function=embeddings      # 必须传入
)


#2.获取文本
texts = [
'''第一条 为了保护个人信息权益，规范个人信息处理活动，促进个人信息合理利用，根据宪法，制定本法。''',
'''第二条 自然人的个人信息受法律保护，任何组织、个人不得侵害自然人的个人信息权益。''',
'''第三条 在中华人民共和国境内处理自然人个人信息的活动，适用本法。
在中华人民共和国境外处理中华人民共和国境内自然人个人信息的活动，有下列情形之一的，也适用本法：
（一）以向境内自然人提供产品或者服务为目的；
（二）分析、评估境内自然人的行为；
（三）法律、行政法规规定的其他情形。''',
'''第四条 个人信息是以电子或者其他方式记录的与已识别或者可识别的自然人有关的各种信息，不包括匿名化处理后
的信息。
个人信息的处理包括个人信息的收集、存储、使用、加工、传输、提供、公开、删除等。'''
]

docs = [Document(page_content=text,
                 metadata = {'source':'法律条文','chunk_ids':idx})
        for idx,text in enumerate(texts)]

db = Chroma.from_documents(
    documents= docs,                    # List [Document]
    embedding=embeddings,               # 你的 Embedding 模型
    persist_directory="Langchain_rag/database",     # 持久化路径
    collection_name="law_knowledge",    # 集合名称（类似表名）
    ids=[f"law_chunk_{i}" for i in range(len(docs))] # 设置ids
)
#ids是官方指定数据库中的参数，代表传入数据库的chunk(切片)的唯一编号(必要)
#chunk_id是什么自己设置的一个元数据(不必要)


'''
如果 collection_name 不存在 → 创建（Create）一个新的 Collection 并写入数据
如果 collection_name 已存在 → 追加（Append） 数据到已有 Collection
'''