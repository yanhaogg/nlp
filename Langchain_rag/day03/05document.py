from langchain_core.documents import Document
# 一个典型的 Document 示例
doc = Document(
    page_content="第一条 个人信息处理应当遵循合法、正当、必要原则……",   # 核心文本内容
    metadata={
        "source": "中华人民共和国个人信息保护法",
        "chapter": "第一章 总则",
        "article_num": "第1条",
        "chunk_id": "doc_001_001"
    }
)
print(doc.page_content[:20])   # 输出文本内容
print(doc.metadata)            # 输出元数据字典


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
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
    print('---')